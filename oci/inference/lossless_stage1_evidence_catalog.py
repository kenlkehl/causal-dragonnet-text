"""Role-neutral, lossless Stage-1 evidence catalog and architecture chunks.

Each source adapter below is a closed projection of an existing Stage-1 handoff.
It emits semantically complete atoms for exactly one canonical architecture.
Large member collections are partitioned into complete member batches, never
rank-truncated or split into arbitrary JSON leaves.  Architecture chunking then
delivers every atom exactly once and rejects an oversize atom before any model
call.

Fold-level numerical summaries are authenticated separately and are never
rendered as concept-grounding evidence.  Row-aligned upstream values remain the
responsibility of the direct numerical integration boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    EXTRACTION_SUPPORT_AXIS,
    HETEROGENEITY_AXIS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    OBSERVABLE_AXES,
    OUTCOME_AXIS,
    PAIR_UPLIFT_AXIS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
    TREATMENT_AXIS,
    DiscoveryEvidenceItem,
    canonical_json,
)
from .all_evidence_fusion import (
    LEGACY_ALL_SOURCE,
    NEURAL_QUERY_SOURCE,
    SPARSE_QUERY_SOURCE,
    TFIDF_TOPIC_SOURCE,
    FoldEvidenceInput,
)
from .htr_attention_evidence_schema import (
    ROLE_NEUTRAL_HTR_CHUNK_EVIDENCE_SCHEMA,
    ROLE_NEUTRAL_HTR_NATIVE_EVIDENCE_SCHEMA,
    ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA,
    ROLE_NEUTRAL_HTR_TOKEN_EVIDENCE_PACKAGE_SCHEMA,
)
from .htr_stage2_complete_semantic_aggregation import (
    HTR_STAGE2_AGGREGATE_BATCH_SCHEMA,
    HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA,
    HTR_STAGE2_MODEL_AGGREGATE_SCHEMA,
)

ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION = "role_neutral_stage1_evidence_catalog_v8"
ARCHITECTURE_CHUNK_SCHEMA_VERSION = "role_neutral_architecture_chunk_v8"
ARCHITECTURE_CHUNK_PLAN_SCHEMA_VERSION = "complete_architecture_chunk_plan_v8"
NON_GROUNDING_SUMMARY_SCHEMA_VERSION = "separated_non_grounding_summary_v1"
NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION = "native_stage1_family_concept_evidence_v1"
NATIVE_ROLE_NEUTRAL_PAYLOAD_ADAPTER_SCHEMA_VERSION = (
    "native_role_neutral_family_payload_adapter_v4"
)
NATIVE_ROLE_NEUTRAL_UNIT_SCHEMA_VERSION = "native_role_neutral_evidence_unit_v1"
SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION = (
    "configured_lossless_semantic_member_batching_v1"
)
SEMANTIC_RETRIEVAL_DERIVATION = "tfidf_ngrams_contrasting_frozen_embedding_retrieval_tails"

DEFAULT_MAX_ATOMS_PER_ARCHITECTURE_CHUNK = 2
DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK = 48_000
DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK = 3

_REQUIRED_SOURCE_KINDS = frozenset({LEGACY_ALL_SOURCE, TFIDF_TOPIC_SOURCE, NEURAL_QUERY_SOURCE})
_FAMILY_ORDER = {family: index for index, family in enumerate(ACTIVE_STAGE1_CONCEPT_FAMILIES)}
_AXIS_ORDER = {axis: index for index, axis in enumerate(OBSERVABLE_AXES)}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CUMULATIVE_SCOPE_ID = re.compile(r"^outer_[0-9]{3}_hierarchy_epoch_[0-9]{3}$")
_BOW_NUISANCE_TYPES = frozenset(
    {
        "confounder_overlap",
        "treatment_positive",
        "treatment_negative",
        "outcome_positive",
        "outcome_negative",
    }
)
_BOW_R_LOSS_TYPES = frozenset({"pseudo_target_positive", "pseudo_target_negative"})
_MATCHED_PAIR_TYPES = frozenset(
    {
        "uplift_pair_features",
        "uplift_delta_logit_positive",
        "uplift_delta_logit_negative",
        "ridge_delta_probability_positive",
        "ridge_delta_probability_negative",
    }
)

_CUMULATIVE_FAMILY_SOURCE_KIND = {
    BOW_NUISANCE: LEGACY_ALL_SOURCE,
    BOW_R_LOSS: LEGACY_ALL_SOURCE,
    HTR_NEURAL: LEGACY_ALL_SOURCE,
    MATCHED_PAIR_UPLIFT: LEGACY_ALL_SOURCE,
    EMBEDDING_WHOLE_COHORT: LEGACY_ALL_SOURCE,
    EMBEDDING_CLUSTERED: LEGACY_ALL_SOURCE,
    TFIDF_SEMANTIC_RETRIEVAL: LEGACY_ALL_SOURCE,
    TFIDF_TOPICS: TFIDF_TOPIC_SOURCE,
    TFIDF_ORPHAN_NGRAMS: TFIDF_TOPIC_SOURCE,
    NEURAL_QUERY_MOMENTS: NEURAL_QUERY_SOURCE,
}
_CUMULATIVE_FAMILY_ATOM_KINDS = {
    BOW_NUISANCE: frozenset({"bow_term_group"}),
    BOW_R_LOSS: frozenset({"bow_term_group"}),
    HTR_NEURAL: frozenset({"htr_semantic_aggregate_batch"}),
    MATCHED_PAIR_UPLIFT: frozenset({"bow_term_group", "matched_pair_htr_phrase"}),
    EMBEDDING_WHOLE_COHORT: frozenset({"embedding_contrast"}),
    EMBEDDING_CLUSTERED: frozenset({"embedding_contrast"}),
    TFIDF_SEMANTIC_RETRIEVAL: frozenset({"tfidf_semantic_retrieval_contrast"}),
    TFIDF_TOPICS: frozenset({"tfidf_topic"}),
    TFIDF_ORPHAN_NGRAMS: frozenset({"tfidf_orphan_ngram_cluster"}),
    NEURAL_QUERY_MOMENTS: frozenset({"neural_query_semantic_witnesses"}),
}
_CUMULATIVE_ATOM_CONTENT_KEYS = {
    "bow_term_group": frozenset(
        {
            "architecture_encoder",
            "group",
            "terms",
            "member_batch_index",
            "member_batch_count",
            "full_member_count",
        }
    ),
    "embedding_contrast": frozenset(
        {
            "architecture_view",
            "contrast",
            "concept_witnesses",
            "member_batch_index",
            "member_batch_count",
            "full_member_count",
        }
    ),
    "tfidf_semantic_retrieval_contrast": frozenset(
        {
            "architecture_view",
            "source_passages_removed",
            "contrast",
            "concept_witnesses",
            "member_batch_index",
            "member_batch_count",
            "full_member_count",
        }
    ),
    "tfidf_topic": frozenset(
        {
            "bank",
            "topic_id",
            "terms",
            "member_batch_index",
            "member_batch_count",
            "full_member_count",
        }
    ),
    "tfidf_orphan_ngram_cluster": frozenset(
        {
            "cluster_id",
            "terms",
            "member_batch_index",
            "member_batch_count",
            "full_member_count",
        }
    ),
    "neural_query_semantic_witnesses": frozenset(
        {
            "bank",
            "query_id",
            "statistical_gate_applied",
            "semantic_witnesses",
            "member_batch_index",
            "member_batch_count",
            "full_member_count",
        }
    ),
    "htr_phrase": frozenset({"architecture_encoder", "group", "phrase_evidence"}),
    "htr_semantic_aggregate_batch": frozenset(
        {"architecture_encoder", "group", "aggregate_batch"}
    ),
    "matched_pair_htr_phrase": frozenset({"architecture_encoder", "group", "phrase_evidence"}),
}
_CUMULATIVE_ATOM_COLLECTION_KEY = {
    "bow_term_group": "terms",
    "embedding_contrast": "concept_witnesses",
    "tfidf_semantic_retrieval_contrast": "concept_witnesses",
    "tfidf_topic": "terms",
    "tfidf_orphan_ngram_cluster": "terms",
    "neural_query_semantic_witnesses": "semantic_witnesses",
}
_CUMULATIVE_ATOM_SINGULAR_KEY = {
    "htr_phrase": "phrase_evidence",
    "htr_semantic_aggregate_batch": "aggregate_batch",
    "matched_pair_htr_phrase": "phrase_evidence",
}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _semantic_member_batching_identity(
    semantic_member_batch_size: Any,
) -> dict[str, Any]:
    if (
        isinstance(semantic_member_batch_size, bool)
        or not isinstance(semantic_member_batch_size, int)
        or semantic_member_batch_size < 1
    ):
        raise ValueError(
            "semantic_member_batch_size must be an explicitly configured "
            "positive integer"
        )
    return {
        "schema_version": SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION,
        "semantic_member_batch_size": int(semantic_member_batch_size),
        "selection_or_truncation_authorized": False,
        "complete_member_coverage_required": True,
    }


def _clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be one JSON object")
    return value


def _sequence(value: Any, *, path: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list")
    return tuple(value)


def _finite_number(value: Any, *, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    return value


def _short_string(
    value: Any,
    *,
    path: str,
    allow_empty: bool = False,
    max_chars: int = 600,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    if normalized != value:
        raise ValueError(f"{path} must use normalized Unicode")
    if not allow_empty and not value.strip():
        raise ValueError(f"{path} cannot be empty")
    if len(value) > max_chars:
        raise ValueError(f"{path} exceeds its closed adapter length")
    return value


def _concept_phrase(value: Any, *, path: str) -> str:
    """Validate one complete semantic witness without a hidden length cap.

    Architecture-level configured byte/member capacities decide whether an
    atom can be transported and fail closed when it cannot.  This adapter must
    not shorten or discard an otherwise valid phrase before that accounting.
    """

    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    phrase = unicodedata.normalize("NFKC", value)
    if phrase != value:
        raise ValueError(f"{path} must use normalized Unicode")
    if not phrase.strip():
        raise ValueError(f"{path} cannot be empty")
    if phrase != " ".join(phrase.split()):
        raise ValueError(f"{path} must use normalized whitespace")
    if re.search(r"https?://|\b[^\s@]+@[^\s@]+\.[^\s@]+\b", phrase):
        raise ValueError(f"{path} contains URL or email-like text")
    if re.search(
        r"\b(?:patient id|medical record number|mrn|account number|social security)\b",
        phrase,
        flags=re.IGNORECASE,
    ):
        raise ValueError(f"{path} contains identifier-like text")
    return phrase


def _closed_keys(value: Mapping[str, Any], *, allowed: Iterable[str], path: str) -> None:
    unexpected = set(map(str, value)) - set(allowed)
    if unexpected:
        raise ValueError(f"{path} contains unhandled fields: {sorted(unexpected)}")


def _ordered_axes(values: Iterable[str]) -> tuple[str, ...]:
    unique = set(values)
    unknown = unique - set(OBSERVABLE_AXES)
    if unknown or not unique:
        raise ValueError(f"observable axes are invalid: {sorted(unknown)}")
    return tuple(sorted(unique, key=_AXIS_ORDER.__getitem__))


def _descriptor_axes(descriptor: str, *, path: str) -> tuple[str, ...]:
    text = descriptor.casefold()
    if "pair" in text or "uplift" in text:
        return (PAIR_UPLIFT_AXIS,)
    if any(
        marker in text
        for marker in (
            "pseudo_target",
            "pseudo-target",
            "r_score",
            "r-stage",
            "residual",
            "interaction",
            "heterogeneity",
            "treated_outcome",
            "untreated_outcome",
            "within_treatment_arm",
        )
    ):
        return (HETEROGENEITY_AXIS,)
    if "nuisance" in text or "confounder_overlap" in text or "confounder_vector" in text:
        return (TREATMENT_AXIS, OUTCOME_AXIS)
    axes: list[str] = []
    if "treatment" in text:
        axes.append(TREATMENT_AXIS)
    if "outcome" in text:
        axes.append(OUTCOME_AXIS)
    if not axes:
        raise ValueError(f"{path} cannot be mapped to an observable Stage-1 axis")
    return _ordered_axes(axes)


def _classify_bow(metadata: Mapping[str, str]) -> tuple[str, tuple[str, ...]]:
    view = metadata["view_name"]
    evidence_type = metadata["evidence_type"]
    source = metadata["source"]
    if not view:
        raise ValueError("BOW producer view name cannot be empty")
    if evidence_type in _BOW_NUISANCE_TYPES:
        if source != f"{view}.{evidence_type}":
            raise ValueError("BOW nuisance producer tuple is not canonical")
        if evidence_type == "confounder_overlap":
            axes = (TREATMENT_AXIS, OUTCOME_AXIS)
        elif evidence_type.startswith("treatment_"):
            axes = (TREATMENT_AXIS,)
        else:
            axes = (OUTCOME_AXIS,)
        return BOW_NUISANCE, axes
    if evidence_type in _BOW_R_LOSS_TYPES:
        if view.startswith("ensemble_r__"):
            if not view.removeprefix("ensemble_r__"):
                raise ValueError("BOW R-loss ensemble view has no configured base view")
            expected_source = f"ensemble_r.{view}.{evidence_type}"
        else:
            expected_source = f"{view}.{evidence_type}"
        if source != expected_source:
            raise ValueError("BOW R-loss source does not match its view and evidence type")
        return BOW_R_LOSS, (HETEROGENEITY_AXIS,)
    if evidence_type in _MATCHED_PAIR_TYPES:
        prefix = "pair_uplift__"
        if not view.startswith(prefix) or not view.removeprefix(prefix):
            raise ValueError("matched-pair BOW view is not canonical")
        if source != f"matched_pair_uplift.{view}.{evidence_type}":
            raise ValueError("matched-pair BOW source does not match its producer tuple")
        return MATCHED_PAIR_UPLIFT, (PAIR_UPLIFT_AXIS,)
    raise ValueError(f"unknown BOW evidence type: {evidence_type!r}")


def _classify_embedding(metadata: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    contrast_family = str(metadata["contrast_family"])
    name = str(metadata["name"])
    if contrast_family == "marginal":
        if name == "treatment":
            return EMBEDDING_WHOLE_COHORT, (TREATMENT_AXIS,)
        if name == "outcome":
            return EMBEDDING_WHOLE_COHORT, (OUTCOME_AXIS,)
        raise ValueError("marginal embedding contrast name must be treatment or outcome")
    if contrast_family == "marginal_confounder_average":
        return EMBEDDING_WHOLE_COHORT, (TREATMENT_AXIS, OUTCOME_AXIS)
    if contrast_family == "within_treatment_arm_outcome":
        return EMBEDDING_WHOLE_COHORT, (OUTCOME_AXIS,)
    if contrast_family in {
        "treatment_outcome_cell_interaction",
        "r_pseudo_target",
        "orthogonal_r_score",
        "residualized_treatment_outcome_cell_interaction",
    }:
        return EMBEDDING_WHOLE_COHORT, (HETEROGENEITY_AXIS,)
    if contrast_family == "cluster_local_treatment_contrast_basis":
        return EMBEDDING_CLUSTERED, (TREATMENT_AXIS,)
    if contrast_family == "cluster_local_residualized_interaction_contrast_basis":
        return EMBEDDING_CLUSTERED, (HETEROGENEITY_AXIS,)
    raise ValueError(f"unknown embedding contrast family: {contrast_family!r}")


def _bank_axis(bank: str, *, path: str) -> tuple[str, ...]:
    normalized = bank.strip().casefold()
    mapping = {
        "treatment": (TREATMENT_AXIS,),
        "outcome": (OUTCOME_AXIS,),
        "effect": (HETEROGENEITY_AXIS,),
    }
    if normalized not in mapping:
        raise ValueError(f"{path} must be treatment, outcome, or effect")
    return mapping[normalized]


def _member_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    semantic_member_batch_size: int,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    _semantic_member_batching_identity(semantic_member_batch_size)
    canonical = sorted((_clone(row) for row in rows), key=canonical_json)
    if not canonical:
        # An empty adapter record carries no lexical or semantic grounding.  It
        # must not become a placeholder evidence atom merely to make a family
        # look present in the all-architecture audit.
        return ()
    return tuple(
        tuple(canonical[start : start + semantic_member_batch_size])
        for start in range(0, len(canonical), semantic_member_batch_size)
    )


def _attach_member_ids(
    content: Mapping[str, Any],
    *,
    source_family: str,
    atom_kind: str,
    parent_collection_sha256: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Attach stable IDs to the one closed member collection in an atom."""

    projected = _clone(content)
    collection_key = {
        "bow_term_group": "terms",
        "embedding_contrast": "concept_witnesses",
        "tfidf_semantic_retrieval_contrast": "concept_witnesses",
        "tfidf_topic": "terms",
        "tfidf_orphan_ngram_cluster": "terms",
        "neural_query_semantic_witnesses": "semantic_witnesses",
        "exact_inner_recurrent_terms": "terms",
    }.get(atom_kind)
    singular_key = {
        "htr_phrase": "phrase_evidence",
        "matched_pair_htr_phrase": "phrase_evidence",
        "htr_semantic_aggregate_batch": "aggregate_batch",
    }.get(atom_kind)
    if collection_key is None and singular_key is None:
        raise ValueError(f"atom kind {atom_kind!r} lacks a closed member-ID adapter")
    if atom_kind == "htr_semantic_aggregate_batch":
        batch = projected.get("aggregate_batch")
        if not isinstance(batch, Mapping):
            raise ValueError(
                "htr_semantic_aggregate_batch.aggregate_batch must be an object"
            )
        raw_members = batch.get("aggregates")
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError(
                "htr_semantic_aggregate_batch must contain semantic aggregates"
            )
        members = raw_members
    elif collection_key is not None:
        raw_members = projected.get(collection_key)
        if not isinstance(raw_members, list):
            raise ValueError(f"{atom_kind}.{collection_key} must be a list")
        members = raw_members
    else:
        raw_member = projected.get(str(singular_key))
        if not isinstance(raw_member, Mapping):
            raise ValueError(f"{atom_kind}.{singular_key} must be an object")
        members = [raw_member]
    member_ids: list[str] = []
    identified: list[dict[str, Any]] = []
    duplicate_counts: Counter[str] = Counter()
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError(f"{atom_kind} members must be objects")
        detached = _clone(member)
        if "member_id" in detached:
            raise ValueError("source member records cannot predeclare member_id")
        member_signature = canonical_json(detached)
        duplicate_counts[member_signature] += 1
        identity = {
            "source_family": source_family,
            "atom_kind": atom_kind,
            "parent_collection_sha256": parent_collection_sha256,
            "member_batch_index": projected.get("member_batch_index"),
            "member_content": detached,
            "duplicate_ordinal_in_batch": duplicate_counts[member_signature],
        }
        member_id = f"member_{_sha256_json(identity)}"
        detached["member_id"] = member_id
        identified.append(detached)
        member_ids.append(member_id)
    if atom_kind == "htr_semantic_aggregate_batch":
        projected_batch = _clone(projected["aggregate_batch"])
        projected_batch["aggregates"] = identified
        projected["aggregate_batch"] = projected_batch
    elif collection_key is not None:
        projected[collection_key] = identified
    else:
        projected[str(singular_key)] = identified[0]
    return projected, tuple(member_ids)


def _bind_member_ids_to_evidence_instance(
    content: Mapping[str, Any],
    *,
    template_member_ids: Sequence[str],
    origin_sha256: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Make duplicate semantic records separately accountable per instance."""

    replacements = {
        template: f"member_{_sha256_json({'template_member_id': template, 'origin_sha256': origin_sha256})}"
        for template in template_member_ids
    }
    observed: list[str] = []

    def replace(value: Any) -> Any:
        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            for key, child in value.items():
                if key == "member_id":
                    template = str(child)
                    if template not in replacements:
                        raise ValueError("atom content contains an unregistered template member ID")
                    output[key] = replacements[template]
                    observed.append(template)
                else:
                    output[str(key)] = replace(child)
            return output
        if isinstance(value, list):
            return [replace(child) for child in value]
        return _clone(value)

    bound = replace(content)
    if set(observed) != set(template_member_ids) or len(observed) != len(template_member_ids):
        raise ValueError("atom member IDs do not bind exactly to its member records")
    return bound, tuple(replacements[value] for value in template_member_ids)


def _validate_partition(item: FoldEvidenceInput) -> None:
    payload = item.payload
    provenance = item.provenance
    if "outer_fold" in payload:
        try:
            outer_fold = int(payload["outer_fold"])
        except (TypeError, ValueError) as exc:
            raise ValueError("payload outer_fold must be an integer") from exc
        if outer_fold != provenance.outer_fold:
            raise ValueError("payload outer_fold does not match fold provenance")
    if "scope" in payload:
        normalized_scope = {
            "outer_train": "outer_train",
            "full_outer_train": "outer_train",
            "inner_train": "inner_train",
            "candidate_selection_inner_fit": "inner_train",
            "candidate_consistency_inner_train": "inner_train",
        }.get(str(payload.get("scope") or "").strip().casefold())
        if normalized_scope != provenance.scope:
            raise ValueError("payload scope does not match fold provenance")
    if "inner_fold" in payload:
        try:
            inner_fold = int(payload["inner_fold"])
        except (TypeError, ValueError) as exc:
            raise ValueError("payload inner_fold must be an integer") from exc
        if inner_fold != provenance.inner_fold:
            raise ValueError("payload inner_fold does not match fold provenance")


def _htr_semantic_aggregate_prompt_content(
    content: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only clinically interpretable HTR aggregate data.

    The catalog retains the authenticated batch, raw-package hashes, reverse
    index references, and schema identities.  Those machine fields are not
    model evidence and therefore never cross the discovery prompt boundary.
    Each aggregate retains its catalog-local member ID, so the ordinary
    exhaustive member-disposition contract accounts for every aggregate.
    """

    batch = content.get("aggregate_batch")
    if not isinstance(batch, Mapping):
        raise ValueError("HTR semantic aggregate prompt source is malformed")
    raw_aggregates = batch.get("aggregates")
    if not isinstance(raw_aggregates, list) or not raw_aggregates:
        raise ValueError("HTR semantic aggregate prompt source is empty")
    aggregates: list[dict[str, Any]] = []
    for raw in raw_aggregates:
        if not isinstance(raw, Mapping):
            raise ValueError("HTR semantic aggregate prompt member is malformed")
        member_id = str(raw.get("member_id") or "")
        fold_support = raw.get("fold_support")
        if (
            not member_id.startswith("member_")
            or not isinstance(fold_support, list)
            or not fold_support
        ):
            raise ValueError("HTR semantic aggregate prompt member is unbound")
        aggregates.append(
            {
                "member_id": member_id,
                "stage": str(raw["stage"]),
                "objective": str(raw["objective"]),
                "normalized_focus_text": str(
                    raw["normalized_focus_text"]
                ),
                "wordpiece_kind": str(raw["wordpiece_kind"]),
                "occurrence_count": int(raw["occurrence_count"]),
                "unique_note_count": int(raw["unique_note_count"]),
                "unique_chunk_count": int(raw["unique_chunk_count"]),
                "attention_summaries": _clone(
                    raw["attention_summaries"]
                ),
                "fold_support": [
                    {
                        "fold": int(row["fold"]),
                        "occurrence_count": int(
                            row["occurrence_count"]
                        ),
                        "unique_note_count": int(
                            row["unique_note_count"]
                        ),
                        "unique_chunk_count": int(
                            row["unique_chunk_count"]
                        ),
                    }
                    for row in fold_support
                ],
                "display_text_variant_count": int(
                    raw["display_text_variant_count"]
                ),
                "context_windows": _clone(raw["context_windows"]),
                "hierarchical_attention_interpretation": (
                    "ranking_heuristic_not_causal_attribution"
                ),
            }
        )
    return {
        "atom_kind": "htr_semantic_aggregate_batch",
        "architecture_encoder": (
            "learned_token_attention_then_document_chunk_transformer"
        ),
        "group": _clone(content["group"]),
        "aggregate_batch": {
            "stage": str(batch["stage"]),
            "objective": str(batch["objective"]),
            "aggregate_count": len(aggregates),
            "aggregates": aggregates,
            "complete_semantic_aggregate_delivery": True,
            "raw_occurrence_inventory_location": (
                "authenticated_handoff_sidecars_not_copied_into_prompt"
            ),
            "special_token_accounting": (
                "retained_in_authenticated_raw_package_and_excluded_from_"
                "readable_phrases"
            ),
        },
    }


@dataclass(frozen=True)
class Stage1EvidenceAtom:
    evidence_id: str
    atom_kind: str
    source_kind: str
    source_family: str
    observable_axes: tuple[str, ...]
    member_ids: tuple[str, ...]
    split_fingerprint: str
    origin_sha256: str
    content_sha256: str
    _origin_json: str = field(repr=False)
    _content_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.evidence_id.startswith("evidence_"):
            raise ValueError("evidence_id must be content addressed")
        if self.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("atom source_family is inactive or unknown")
        if not self.member_ids:
            raise ValueError("concept-bearing evidence atom must contain semantic members")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("evidence atom member_ids cannot contain duplicates")
        _ordered_axes(self.observable_axes)
        for label, value in (
            ("split_fingerprint", self.split_fingerprint),
            ("origin_sha256", self.origin_sha256),
            ("content_sha256", self.content_sha256),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256")

    @property
    def origin(self) -> dict[str, Any]:
        return json.loads(self._origin_json)

    @property
    def content(self) -> dict[str, Any]:
        return json.loads(self._content_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
            "evidence_id": self.evidence_id,
            "atom_kind": self.atom_kind,
            "source_kind": self.source_kind,
            "source_family": self.source_family,
            "observable_axes": list(self.observable_axes),
            "member_ids": list(self.member_ids),
            "split_fingerprint": self.split_fingerprint,
            "origin_sha256": self.origin_sha256,
            "content_sha256": self.content_sha256,
            "origin": self.origin,
            "content": self.content,
        }

    def as_discovery_item(self) -> DiscoveryEvidenceItem:
        content = (
            _htr_semantic_aggregate_prompt_content(self.content)
            if self.atom_kind == "htr_semantic_aggregate_batch"
            else {"atom_kind": self.atom_kind, **self.content}
        )
        return DiscoveryEvidenceItem(
            evidence_id=self.evidence_id,
            source_family=self.source_family,
            observable_axes=self.observable_axes,
            content=content,
            member_ids=self.member_ids,
        )


@dataclass(frozen=True)
class NonGroundingNumericalSummary:
    summary_id: str
    source_kind: str
    source_family: str
    observable_axes: tuple[str, ...]
    split_fingerprint: str
    _metrics_json: str = field(repr=False)

    @property
    def metrics(self) -> dict[str, int | float]:
        return json.loads(self._metrics_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NON_GROUNDING_SUMMARY_SCHEMA_VERSION,
            "summary_id": self.summary_id,
            "source_kind": self.source_kind,
            "source_family": self.source_family,
            "observable_axes": list(self.observable_axes),
            "split_fingerprint": self.split_fingerprint,
            "metrics": self.metrics,
            "concept_grounding_allowed": False,
        }


@dataclass(frozen=True)
class RoleNeutralEvidenceCatalog:
    outer_fold: int
    scope: str
    inner_fold: int | None
    split_fingerprint: str
    atoms: tuple[Stage1EvidenceAtom, ...]
    non_grounding_numerical_summaries: tuple[NonGroundingNumericalSummary, ...]
    catalog_sha256: str
    _audit_json: str = field(repr=False)

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
            "outer_fold": self.outer_fold,
            "scope": self.scope,
            "inner_fold": self.inner_fold,
            "split_fingerprint": self.split_fingerprint,
            "catalog_sha256": self.catalog_sha256,
            "atoms": [atom.as_dict() for atom in self.atoms],
            "non_grounding_numerical_summaries": [
                summary.as_dict() for summary in self.non_grounding_numerical_summaries
            ],
            "audit": self.audit,
        }

    def family_atoms(self, source_family: str) -> tuple[Stage1EvidenceAtom, ...]:
        if source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("source_family is inactive or unknown")
        return tuple(atom for atom in self.atoms if atom.source_family == source_family)


@dataclass(frozen=True)
class _Prototype:
    atom_kind: str
    source_kind: str
    source_family: str
    observable_axes: tuple[str, ...]
    member_ids: tuple[str, ...]
    split_fingerprint: str
    _origin_base_json: str
    _content_json: str

    @property
    def signature(self) -> str:
        return canonical_json(
            {
                "atom_kind": self.atom_kind,
                "source_kind": self.source_kind,
                "source_family": self.source_family,
                "observable_axes": self.observable_axes,
                "member_ids": self.member_ids,
                "split_fingerprint": self.split_fingerprint,
                "origin_base": json.loads(self._origin_base_json),
                "content": json.loads(self._content_json),
            }
        )


class _CatalogBuilder:
    def __init__(
        self,
        inputs: Sequence[FoldEvidenceInput],
        *,
        semantic_member_batch_size: int,
    ) -> None:
        self.inputs = tuple(inputs)
        self.semantic_member_batching = _semantic_member_batching_identity(
            semantic_member_batch_size
        )
        self.semantic_member_batch_size = int(semantic_member_batch_size)
        self.prototypes: list[_Prototype] = []
        self.summaries: list[NonGroundingNumericalSummary] = []
        self.semantic_source_counts: Counter[str] = Counter()
        self.upstream_truncations: list[dict[str, Any]] = []
        self.empty_lexical_query_count = 0

    def emit(
        self,
        *,
        item: FoldEvidenceInput,
        atom_kind: str,
        source_family: str,
        observable_axes: Sequence[str],
        branch: str,
        parent_collection_sha256: str,
        content: Mapping[str, Any],
        view_of_parent: str | None = None,
    ) -> None:
        if source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("cannot emit an inactive architecture")
        safe_content, member_ids = _attach_member_ids(
            content,
            source_family=source_family,
            atom_kind=atom_kind,
            parent_collection_sha256=parent_collection_sha256,
        )
        # Constructing the discovery item here applies the prompt-boundary
        # forbidden-key scan before an atom can enter the catalog.
        DiscoveryEvidenceItem(
            evidence_id="evidence_precommit",
            source_family=source_family,
            observable_axes=_ordered_axes(observable_axes),
            content={"atom_kind": atom_kind, **safe_content},
            member_ids=member_ids,
        )
        origin = {
            "source_kind": item.source_kind,
            "artifact_id_sha256": hashlib.sha256(
                item.provenance.artifact_id.encode("utf-8")
            ).hexdigest(),
            "branch": branch,
            "parent_collection_sha256": parent_collection_sha256,
        }
        if view_of_parent is not None:
            origin["architecture_view_of_parent"] = view_of_parent
        self.prototypes.append(
            _Prototype(
                atom_kind=atom_kind,
                source_kind=item.source_kind,
                source_family=source_family,
                observable_axes=_ordered_axes(observable_axes),
                member_ids=member_ids,
                split_fingerprint=item.provenance.split_fingerprint,
                _origin_base_json=canonical_json(origin),
                _content_json=canonical_json(safe_content),
            )
        )
        self.semantic_source_counts[source_family] += 1

    def emit_summary(
        self,
        *,
        item: FoldEvidenceInput,
        source_family: str,
        observable_axes: Sequence[str],
        coordinate: str,
        metrics: Mapping[str, Any],
    ) -> None:
        parsed: dict[str, int | float] = {}
        for key, value in metrics.items():
            _short_string(key, path="summary metric name", max_chars=120)
            parsed[key] = _finite_number(value, path=f"summary.metrics.{key}")
        identity = {
            "schema_version": NON_GROUNDING_SUMMARY_SCHEMA_VERSION,
            "source_kind": item.source_kind,
            "source_family": source_family,
            "observable_axes": _ordered_axes(observable_axes),
            "split_fingerprint": item.provenance.split_fingerprint,
            "coordinate": coordinate,
            "metrics": parsed,
        }
        self.summaries.append(
            NonGroundingNumericalSummary(
                summary_id=f"summary_{_sha256_json(identity)}",
                source_kind=item.source_kind,
                source_family=source_family,
                observable_axes=_ordered_axes(observable_axes),
                split_fingerprint=item.provenance.split_fingerprint,
                _metrics_json=canonical_json(parsed),
            )
        )

    def build(
        self,
        *,
        require_all_source_kinds: bool,
        require_all_architecture_families: bool,
        require_upstream_completeness: bool,
    ) -> RoleNeutralEvidenceCatalog:
        if not self.inputs:
            raise ValueError("at least one FoldEvidenceInput is required")
        if not all(isinstance(item, FoldEvidenceInput) for item in self.inputs):
            raise TypeError("catalog inputs must be FoldEvidenceInput objects")
        source_kinds = [item.source_kind for item in self.inputs]
        if len(source_kinds) != len(set(source_kinds)):
            raise ValueError("each source_kind can appear at most once")
        if SPARSE_QUERY_SOURCE in source_kinds:
            raise ValueError("the inactive sparse-query fallback cannot enter this catalog")
        if require_all_source_kinds and set(source_kinds) != _REQUIRED_SOURCE_KINDS:
            raise ValueError(
                "strict all-architecture catalog requires legacy, TF-IDF, and neural-query inputs"
            )
        reference = self.inputs[0].provenance
        for item in self.inputs:
            if item.provenance.split_fingerprint != reference.split_fingerprint:
                raise ValueError("all catalog sources must have identical fold provenance")
            _validate_partition(item)
        for item in sorted(self.inputs, key=lambda row: row.source_kind):
            if item.source_kind == LEGACY_ALL_SOURCE:
                self._legacy(item)
            elif item.source_kind == TFIDF_TOPIC_SOURCE:
                self._tfidf(item)
            elif item.source_kind == NEURAL_QUERY_SOURCE:
                self._neural_queries(item)
        if require_upstream_completeness and self.upstream_truncations:
            raise ValueError(
                "upstream evidence declares truncation before the lossless catalog: "
                f"{self.upstream_truncations}"
            )

        grouped: dict[str, list[_Prototype]] = defaultdict(list)
        for prototype in self.prototypes:
            grouped[prototype.signature].append(prototype)
        atoms: list[Stage1EvidenceAtom] = []
        for signature in sorted(grouped):
            rows = grouped[signature]
            prototype = rows[0]
            for ordinal in range(1, len(rows) + 1):
                origin = json.loads(prototype._origin_base_json)
                origin.update({"multiplicity_ordinal": ordinal, "multiplicity_count": len(rows)})
                origin_sha = _sha256_json(origin)
                content, member_ids = _bind_member_ids_to_evidence_instance(
                    json.loads(prototype._content_json),
                    template_member_ids=prototype.member_ids,
                    origin_sha256=origin_sha,
                )
                content_sha = _sha256_json(content)
                identity = {
                    "atom_kind": prototype.atom_kind,
                    "source_kind": prototype.source_kind,
                    "source_family": prototype.source_family,
                    "observable_axes": prototype.observable_axes,
                    "member_ids": member_ids,
                    "split_fingerprint": prototype.split_fingerprint,
                    "origin_sha256": origin_sha,
                    "content_sha256": content_sha,
                }
                atoms.append(
                    Stage1EvidenceAtom(
                        evidence_id=f"evidence_{_sha256_json(identity)}",
                        atom_kind=prototype.atom_kind,
                        source_kind=prototype.source_kind,
                        source_family=prototype.source_family,
                        observable_axes=prototype.observable_axes,
                        member_ids=member_ids,
                        split_fingerprint=prototype.split_fingerprint,
                        origin_sha256=origin_sha,
                        content_sha256=content_sha,
                        _origin_json=canonical_json(origin),
                        _content_json=canonical_json(content),
                    )
                )
        atoms.sort(key=lambda row: row.evidence_id)
        summaries = sorted(self.summaries, key=lambda row: row.summary_id)
        if len({row.evidence_id for row in atoms}) != len(atoms):
            raise RuntimeError("evidence atom IDs collided")
        if len({row.summary_id for row in summaries}) != len(summaries):
            raise ValueError("duplicate non-grounding numerical summaries")
        identity = {
            "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
            "semantic_member_batching": self.semantic_member_batching,
            "outer_fold": reference.outer_fold,
            "scope": reference.scope,
            "inner_fold": reference.inner_fold,
            "split_fingerprint": reference.split_fingerprint,
            "atoms": [atom.as_dict() for atom in atoms],
            "non_grounding_numerical_summaries": [row.as_dict() for row in summaries],
        }
        catalog_sha = _sha256_json(identity)
        family_counts = Counter(atom.source_family for atom in atoms)
        family_member_counts = Counter(
            {
                family: sum(len(atom.member_ids) for atom in atoms if atom.source_family == family)
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            }
        )
        missing_families = [
            family
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            if family_counts.get(family, 0) == 0 or family_member_counts.get(family, 0) == 0
        ]
        if require_all_architecture_families and missing_families:
            raise ValueError(
                "strict all-architecture catalog has no concept-bearing evidence for: "
                f"{missing_families}"
            )
        audit = {
            "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
            "catalog_sha256": catalog_sha,
            "source_kinds": sorted(source_kinds),
            "inactive_sparse_query_present": False,
            "role_fields_emitted": False,
            "extraction_contracts_emitted": False,
            "temporal_policy_emitted": False,
            "global_top_k_applied": False,
            "semantic_member_batching": self.semantic_member_batching,
            "semantic_member_batch_size": self.semantic_member_batch_size,
            "semantic_member_batches_truncated": False,
            "atom_count": len(atoms),
            "atom_count_by_family": {
                family: family_counts.get(family, 0) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
            "semantic_member_count_by_family": {
                family: family_member_counts.get(family, 0)
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            },
            "all_architecture_families_required": require_all_architecture_families,
            "missing_architecture_families": missing_families,
            "non_grounding_numerical_summary_count": len(summaries),
            "non_grounding_summaries_visible_to_discovery": False,
            "upstream_completeness_required": require_upstream_completeness,
            "upstream_truncation_count": len(self.upstream_truncations),
            "upstream_truncations": self.upstream_truncations,
            "empty_neural_query_lexical_witness_count": self.empty_lexical_query_count,
        }
        catalog = RoleNeutralEvidenceCatalog(
            outer_fold=reference.outer_fold,
            scope=reference.scope,
            inner_fold=reference.inner_fold,
            split_fingerprint=reference.split_fingerprint,
            atoms=tuple(atoms),
            non_grounding_numerical_summaries=tuple(summaries),
            catalog_sha256=catalog_sha,
            _audit_json=canonical_json(audit),
        )
        validate_role_neutral_catalog(catalog)
        return catalog

    def _legacy(self, item: FoldEvidenceInput) -> None:
        context = item.payload.get("context")
        context = context if isinstance(context, Mapping) else item.payload
        digest = _mapping(context.get("evidence_digest"), path="legacy.evidence_digest")
        _closed_keys(digest, allowed={"confounders", "effect_modifiers"}, path="legacy.digest")
        for section_name in ("confounders", "effect_modifiers"):
            section = digest.get(section_name)
            if section is None:
                continue
            section = _mapping(section, path=f"legacy.{section_name}")
            _closed_keys(
                section,
                allowed={
                    "bow_blurbs",
                    "embedding_chunks",
                    "htr_blurbs",
                    "role",
                    "role_definition",
                },
                path=f"legacy.{section_name}",
            )
            for raw in _sequence(section.get("bow_blurbs"), path="legacy.bow_blurbs"):
                self._bow_group(item, _mapping(raw, path="legacy.bow_group"), section_name)
            for raw in _sequence(section.get("embedding_chunks"), path="legacy.embedding_chunks"):
                self._embedding(item, _mapping(raw, path="legacy.embedding"), section_name)
            for raw in _sequence(section.get("htr_blurbs"), path="legacy.htr_blurbs"):
                self._htr_group(item, _mapping(raw, path="legacy.htr_group"), section_name)
        recurrence = context.get("exact_inner_recurrence")
        if recurrence is not None:
            self._recurrence(
                item,
                _mapping(recurrence, path="legacy.exact_inner_recurrence"),
                branch="exact_inner_recurrence",
            )

    def _bow_group(
        self, item: FoldEvidenceInput, group: Mapping[str, Any], section_name: str
    ) -> None:
        _closed_keys(
            group,
            allowed={"source", "view_name", "bow_model", "evidence_type", "meaning", "rows"},
            path="legacy.bow_group",
        )
        metadata = {
            key: _short_string(group.get(key), path=f"legacy.bow_group.{key}")
            for key in ("source", "view_name", "bow_model", "evidence_type", "meaning")
        }
        family, axes = _classify_bow(metadata)
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(_sequence(group.get("rows"), path="legacy.bow_group.rows")):
            row = _mapping(raw, path=f"legacy.bow_group.rows[{index}]")
            _closed_keys(row, allowed={"feature", "term", "phrase", "score"}, path="bow.row")
            phrase_field = next(
                (key for key in ("feature", "term", "phrase") if row.get(key) is not None),
                None,
            )
            if phrase_field is None:
                raise ValueError("BOW evidence row has no concept phrase")
            projected = {
                "term": _concept_phrase(row[phrase_field], path=f"bow.rows[{index}].term"),
                "score": _finite_number(row.get("score"), path=f"bow.rows[{index}].score"),
            }
            rows.append(projected)
        parent = _sha256_json({"metadata": metadata, "rows": sorted(rows, key=canonical_json)})
        batches = _member_batches(
            rows,
            semantic_member_batch_size=self.semantic_member_batch_size,
        )
        for batch_index, batch in enumerate(batches, start=1):
            self.emit(
                item=item,
                atom_kind="bow_term_group",
                source_family=family,
                observable_axes=axes,
                branch=f"{section_name}.bow_blurbs",
                parent_collection_sha256=parent,
                content={
                    "architecture_encoder": "bow",
                    "group": metadata,
                    "terms": list(batch),
                    "member_batch_index": batch_index,
                    "member_batch_count": len(batches),
                    "full_member_count": len(rows),
                },
            )

    def _embedding(
        self, item: FoldEvidenceInput, contrast: Mapping[str, Any], section_name: str
    ) -> None:
        excerpt_keys = {
            "positive_aligned_chunks",
            "negative_aligned_chunks",
            "positive_external_chunks",
            "negative_external_chunks",
        }
        _closed_keys(
            contrast,
            allowed={
                "name",
                "contrast_family",
                "direction_source",
                "role_hint",
                "cluster_component_index",
                "concept_derivation",
                "raw_retrieved_excerpts_retained",
                "concept_probe_scores",
                *excerpt_keys,
            },
            path="legacy.embedding",
        )
        for key in excerpt_keys:
            if _sequence(contrast.get(key), path=f"legacy.embedding.{key}"):
                raise ValueError("raw embedding retrieval excerpts cannot enter discovery")
        metadata: dict[str, Any] = {}
        for key in ("name", "contrast_family", "direction_source"):
            metadata[key] = _short_string(contrast.get(key), path=f"embedding.{key}")
        if contrast.get("cluster_component_index") is not None:
            metadata["cluster_component_index"] = int(contrast["cluster_component_index"])
        structural_family, axes = _classify_embedding(metadata)
        probes: list[dict[str, Any]] = []
        for index, raw in enumerate(
            _sequence(contrast.get("concept_probe_scores"), path="embedding.concept_probe_scores")
        ):
            row = _mapping(raw, path=f"embedding.concept_probe_scores[{index}]")
            _closed_keys(row, allowed={"concept", "phrase", "label", "score"}, path="probe")
            phrase_field = next(
                (key for key in ("concept", "phrase", "label") if row.get(key) is not None),
                None,
            )
            if phrase_field is None:
                raise ValueError("embedding concept probe has no phrase")
            probes.append(
                {
                    "concept": _concept_phrase(
                        row[phrase_field], path=f"embedding.probes[{index}].concept"
                    ),
                    "score": _finite_number(
                        row.get("score"), path=f"embedding.probes[{index}].score"
                    ),
                }
            )
        parent = _sha256_json({"metadata": metadata, "probes": sorted(probes, key=canonical_json)})
        batches = _member_batches(
            probes,
            semantic_member_batch_size=self.semantic_member_batch_size,
        )
        semantic = contrast.get("concept_derivation") == SEMANTIC_RETRIEVAL_DERIVATION
        if semantic and contrast.get("raw_retrieved_excerpts_retained") is not False:
            raise ValueError("semantic retrieval view must attest that raw excerpts were removed")
        for batch_index, batch in enumerate(batches, start=1):
            common = {
                "contrast": metadata,
                "concept_witnesses": list(batch),
                "member_batch_index": batch_index,
                "member_batch_count": len(batches),
                "full_member_count": len(probes),
            }
            self.emit(
                item=item,
                atom_kind="embedding_contrast",
                source_family=structural_family,
                observable_axes=axes,
                branch=f"{section_name}.embedding_chunks",
                parent_collection_sha256=parent,
                content={"architecture_view": "embedding_contrast", **common},
            )
            if semantic:
                self.emit(
                    item=item,
                    atom_kind="tfidf_semantic_retrieval_contrast",
                    source_family=TFIDF_SEMANTIC_RETRIEVAL,
                    observable_axes=axes,
                    branch=f"{section_name}.embedding_chunks.semantic_projection",
                    parent_collection_sha256=parent,
                    view_of_parent=structural_family,
                    content={
                        "architecture_view": SEMANTIC_RETRIEVAL_DERIVATION,
                        "source_passages_removed": True,
                        **common,
                    },
                )

    def _htr_group(
        self, item: FoldEvidenceInput, group: Mapping[str, Any], section_name: str
    ) -> None:
        _closed_keys(
            group,
            allowed={"stage", "meaning", "metrics", "rows", "exact_attention_phrases"},
            path="legacy.htr_group",
        )
        stage = _short_string(group.get("stage"), path="htr.stage")
        meaning = _short_string(group.get("meaning"), path="htr.meaning")
        try:
            axes = {
                "nuisance": (TREATMENT_AXIS, OUTCOME_AXIS),
                "effect": (HETEROGENEITY_AXIS,),
                "pair_uplift": (PAIR_UPLIFT_AXIS,),
            }[stage]
        except KeyError as exc:
            raise ValueError(f"unknown HTR stage: {stage!r}") from exc
        metadata = {"stage": stage, "meaning": meaning}
        phrase_rows: list[dict[str, Any]] = []
        allowed_row = {
            "attended_token_summary",
            "attention_score",
            "token",
            "phrase",
            "concept",
            "feature",
            "evidence_snippet",
            "top_token_spans",
        }
        for row_index, raw in enumerate(_sequence(group.get("rows"), path="htr.rows")):
            row = _mapping(raw, path=f"htr.rows[{row_index}]")
            _closed_keys(row, allowed=allowed_row, path=f"htr.rows[{row_index}]")
            score = row.get("attention_score")
            if score is not None:
                score = _finite_number(score, path=f"htr.rows[{row_index}].attention_score")
            for field_name in (
                "attended_token_summary",
                "token",
                "phrase",
                "concept",
                "feature",
                "evidence_snippet",
            ):
                if row.get(field_name) is not None:
                    phrase_rows.append(
                        {
                            "phrase": _concept_phrase(
                                row[field_name], path=f"htr.rows[{row_index}].{field_name}"
                            ),
                            "phrase_source_field": field_name,
                            "attention_score": score,
                        }
                    )
            for span_index, raw_span in enumerate(
                _sequence(row.get("top_token_spans"), path="htr.top_token_spans")
            ):
                span = _mapping(raw_span, path=f"htr.spans[{span_index}]")
                _closed_keys(
                    span,
                    allowed={"token", "phrase", "concept", "attention_score", "score"},
                    path="htr.span",
                )
                field_name = next(
                    (key for key in ("token", "phrase", "concept") if span.get(key) is not None),
                    None,
                )
                if field_name is None:
                    raise ValueError("HTR token span has no phrase")
                raw_score = span.get("attention_score", span.get("score", score))
                phrase_rows.append(
                    {
                        "phrase": _concept_phrase(span[field_name], path="htr.span.phrase"),
                        "phrase_source_field": f"top_token_spans.{field_name}",
                        "attention_score": (
                            None
                            if raw_score is None
                            else _finite_number(raw_score, path="htr.span.attention_score")
                        ),
                    }
                )
        for index, raw in enumerate(
            _sequence(group.get("exact_attention_phrases"), path="htr.exact_attention_phrases")
        ):
            row = _mapping(raw, path=f"htr.exact_attention_phrases[{index}]")
            _closed_keys(
                row,
                allowed={"token", "phrase", "concept", "attention_score", "score"},
                path="htr.exact_phrase",
            )
            field_name = next(
                (key for key in ("token", "phrase", "concept") if row.get(key) is not None),
                None,
            )
            if field_name is None:
                raise ValueError("HTR exact phrase has no phrase")
            raw_score = row.get("attention_score", row.get("score"))
            phrase_rows.append(
                {
                    "phrase": _concept_phrase(row[field_name], path="htr.exact_phrase.phrase"),
                    "phrase_source_field": f"exact_attention_phrases.{field_name}",
                    "attention_score": (
                        None
                        if raw_score is None
                        else _finite_number(raw_score, path="htr.exact_phrase.attention_score")
                    ),
                }
            )
        if not phrase_rows:
            raise ValueError("HTR group has no exact phrase evidence")
        parent = _sha256_json(
            {"metadata": metadata, "phrases": sorted(phrase_rows, key=canonical_json)}
        )
        for phrase in sorted(phrase_rows, key=canonical_json):
            content = {
                "architecture_encoder": "htr",
                "group": metadata,
                "phrase_evidence": phrase,
            }
            if PAIR_UPLIFT_AXIS in axes:
                self.emit(
                    item=item,
                    atom_kind="matched_pair_htr_phrase",
                    source_family=MATCHED_PAIR_UPLIFT,
                    observable_axes=axes,
                    branch=f"{section_name}.htr_blurbs",
                    parent_collection_sha256=parent,
                    view_of_parent=HTR_NEURAL,
                    content=content,
                )
            else:
                self.emit(
                    item=item,
                    atom_kind="htr_phrase",
                    source_family=HTR_NEURAL,
                    observable_axes=axes,
                    branch=f"{section_name}.htr_blurbs",
                    parent_collection_sha256=parent,
                    content=content,
                )

    def _tfidf(self, item: FoldEvidenceInput) -> None:
        discovery = item.payload.get("discovery")
        discovery = discovery if isinstance(discovery, Mapping) else item.payload
        banks = _mapping(discovery.get("topic_banks"), path="tfidf.topic_banks")
        _closed_keys(banks, allowed={"treatment", "outcome", "effect"}, path="tfidf.banks")
        for bank in ("treatment", "outcome", "effect"):
            raw_bank = banks.get(bank)
            if raw_bank is None:
                continue
            bank_payload = _mapping(raw_bank, path=f"tfidf.topic_banks.{bank}")
            _closed_keys(
                bank_payload,
                allowed={
                    "bank",
                    "requested_topic_count",
                    "actual_topic_count",
                    "terms_per_topic",
                    "component_reduction_reason",
                    "seeds",
                    "selected_term_count",
                    "selected_terms",
                    "eligible_candidate_count",
                    "selected_candidate_count",
                    "discarded_candidate_count",
                    "selection_rule",
                    "feature_weights",
                    "alignments",
                    "weak_or_unstable_raw_evidence",
                    "topics",
                },
                path=f"tfidf.{bank}",
            )
            terms_per_topic = int(bank_payload.get("terms_per_topic") or 0)
            if terms_per_topic < 1 and (bank_payload.get("topics") or []):
                raise ValueError(
                    f"tfidf.{bank}.terms_per_topic must be a positive "
                    "configured evidence capacity"
                )
            for raw in _sequence(bank_payload.get("topics"), path=f"tfidf.{bank}.topics"):
                self._tfidf_topic(
                    item,
                    _mapping(raw, path="tfidf.topic"),
                    bank,
                    terms_per_topic=terms_per_topic,
                )
        orphan = discovery.get("effect_orphan_ngram_branch")
        if not isinstance(orphan, Mapping):
            for key in ("topic_score_tests", "topic_score_selection", "score_tests"):
                nested = discovery.get(key)
                if isinstance(nested, Mapping) and isinstance(
                    nested.get("effect_orphan_ngram_branch"), Mapping
                ):
                    orphan = nested["effect_orphan_ngram_branch"]
                    break
        if isinstance(orphan, Mapping):
            _closed_keys(
                orphan,
                allowed={
                    "schema_version",
                    "status",
                    "candidate_definition",
                    "uses_outer_heldout_labels",
                    "uses_heldout_treatment_and_outcome",
                    "fits_patient_level_cate_model",
                    "topic_term_exclusion_is_fit_side",
                    "cluster_construction_uses_heldout_rows_or_labels",
                    "candidate_count_before_topic_exclusion",
                    "represented_topic_term_exclusion_count",
                    "candidate_count_before_nested_deduplication",
                    "deduplicated_alias_count",
                    "representative_count",
                    "cluster_count",
                    "lexical_overlap_threshold",
                    "selected_cluster_ids",
                    "selected_clusters",
                    "clusters",
                    "selection_count",
                    "selection_rule",
                    "minimum_selected_clusters",
                    "maximum_selected_clusters",
                    "source_artifact_audit",
                },
                path="tfidf.effect_orphan_ngram_branch",
            )
            selected_ids = tuple(str(value) for value in (orphan.get("selected_cluster_ids") or []))
            if len(selected_ids) != len(set(selected_ids)):
                raise ValueError("orphan selected_cluster_ids cannot contain duplicates")
            seen: set[str] = set()
            for collection_key in ("selected_clusters", "clusters"):
                for raw in _sequence(orphan.get(collection_key), path=f"tfidf.{collection_key}"):
                    cluster = _mapping(raw, path="tfidf.orphan.cluster")
                    cluster_hash = _sha256_json(_clone(cluster))
                    if cluster_hash in seen:
                        continue
                    seen.add(cluster_hash)
                    self._orphan_cluster(item, cluster, collection_key)
            if selected_ids:
                observed_ids = {
                    str(_mapping(raw, path="tfidf.orphan.cluster").get("cluster_id") or "")
                    for raw in _sequence(
                        orphan.get("selected_clusters"), path="tfidf.selected_clusters"
                    )
                }
                if observed_ids != set(selected_ids):
                    raise ValueError(
                        "orphan selected_cluster_ids must exactly match selected_clusters"
                    )
        recurrence = discovery.get("exact_inner_recurrence")
        if recurrence is not None:
            self._recurrence(
                item,
                _mapping(recurrence, path="tfidf.exact_inner_recurrence"),
                branch="exact_inner_recurrence",
            )

    def _recurrence(
        self,
        item: FoldEvidenceInput,
        recurrence: Mapping[str, Any],
        *,
        branch: str,
    ) -> None:
        _closed_keys(
            recurrence,
            allowed={
                "schema_version",
                "normalization",
                "inner_fold_count",
                "latent_topic_ids_compared_across_folds",
                "minimum_inner_fold_support",
                "groups",
            },
            path="exact_inner_recurrence",
        )
        role_axes = {
            "confounder": (TREATMENT_AXIS, OUTCOME_AXIS),
            "effect_modifier": (HETEROGENEITY_AXIS,),
            "prognostic": (OUTCOME_AXIS,),
            "extraction_support": (EXTRACTION_SUPPORT_AXIS,),
        }
        for group_index, raw_group in enumerate(
            _sequence(recurrence.get("groups"), path="exact_inner_recurrence.groups")
        ):
            group = _mapping(raw_group, path=f"exact_inner_recurrence.groups[{group_index}]")
            _closed_keys(
                group,
                allowed={
                    "source_family",
                    "role",
                    "discovered_recurrent_term_count",
                    "retained_term_count",
                    "terms",
                },
                path="exact_inner_recurrence.group",
            )
            family = str(group.get("source_family") or "")
            if family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
                raise ValueError("recurrence group has an inactive or unknown source family")
            legacy_role = str(group.get("role") or "")
            if legacy_role not in role_axes:
                raise ValueError("recurrence group role cannot be mapped to observable axes")
            discovered = int(group.get("discovered_recurrent_term_count", 0))
            retained = int(group.get("retained_term_count", 0))
            if discovered < 0 or retained < 0 or retained > discovered:
                raise ValueError("recurrence group counts are inconsistent")
            if discovered > retained:
                self.upstream_truncations.append(
                    {
                        "source_kind": item.source_kind,
                        "source_family": family,
                        "discovered_recurrent_term_count": discovered,
                        "retained_term_count": retained,
                    }
                )
            terms: list[dict[str, Any]] = []
            for term_index, raw_term in enumerate(
                _sequence(group.get("terms"), path="exact_inner_recurrence.terms")
            ):
                term = _mapping(raw_term, path=f"exact_inner_recurrence.terms[{term_index}]")
                _closed_keys(
                    term,
                    allowed={
                        "term",
                        "inner_fold_support_count",
                        "inner_fold_support_fraction",
                        "occurrence_count",
                    },
                    path="exact_inner_recurrence.term",
                )
                projected: dict[str, Any] = {
                    "term": _concept_phrase(
                        term.get("term"), path="exact_inner_recurrence.term.term"
                    )
                }
                for key in (
                    "inner_fold_support_count",
                    "inner_fold_support_fraction",
                    "occurrence_count",
                ):
                    if term.get(key) is not None:
                        projected[key] = _finite_number(
                            term[key], path=f"exact_inner_recurrence.term.{key}"
                        )
                terms.append(projected)
            if len(terms) != retained:
                raise ValueError("recurrence retained_term_count does not match supplied terms")
            parent = _sha256_json(
                {
                    "source_family": family,
                    "observable_axes": role_axes[legacy_role],
                    "terms": sorted(terms, key=canonical_json),
                }
            )
            batches = _member_batches(
                terms,
                semantic_member_batch_size=self.semantic_member_batch_size,
            )
            for batch_index, batch in enumerate(batches, start=1):
                self.emit(
                    item=item,
                    atom_kind="exact_inner_recurrent_terms",
                    source_family=family,
                    observable_axes=role_axes[legacy_role],
                    branch=branch,
                    parent_collection_sha256=parent,
                    content={
                        "terms": list(batch),
                        "member_batch_index": batch_index,
                        "member_batch_count": len(batches),
                        "full_member_count": len(terms),
                    },
                )

    def _tfidf_topic(
        self,
        item: FoldEvidenceInput,
        topic: Mapping[str, Any],
        bank: str,
        *,
        terms_per_topic: int,
    ) -> None:
        _closed_keys(
            topic,
            allowed={"topic_id", "bank", "terms_per_topic", "terms"},
            path="tfidf.topic",
        )
        topic_id = _short_string(topic.get("topic_id"), path="tfidf.topic.topic_id")
        if topic.get("bank") is not None and str(topic.get("bank")).casefold() != bank:
            raise ValueError("TF-IDF topic bank field does not match its containing bank")
        if int(topic.get("terms_per_topic") or 0) != int(terms_per_topic):
            raise ValueError(
                "TF-IDF topic term capacity does not match its containing bank"
            )
        terms: list[dict[str, Any]] = []
        for index, raw in enumerate(_sequence(topic.get("terms"), path="tfidf.topic.terms")):
            row = {"term": raw} if isinstance(raw, str) else _mapping(raw, path="tfidf.term")
            _closed_keys(
                row,
                allowed={"term", "loading", "screen_rank", "signed_score"},
                path="tfidf.term",
            )
            projected: dict[str, Any] = {
                "term": _concept_phrase(row.get("term"), path=f"tfidf.terms[{index}].term")
            }
            for key in ("loading", "screen_rank", "signed_score"):
                if row.get(key) is not None:
                    projected[key] = _finite_number(row[key], path=f"tfidf.term.{key}")
            terms.append(projected)
        if len(terms) != int(terms_per_topic):
            raise ValueError(
                f"TF-IDF topic supplied {len(terms)} terms; expected the "
                f"complete configured capacity {int(terms_per_topic)}"
            )
        parent = _sha256_json(
            {"bank": bank, "topic_id": topic_id, "terms": sorted(terms, key=canonical_json)}
        )
        batches = _member_batches(
            terms,
            semantic_member_batch_size=self.semantic_member_batch_size,
        )
        for batch_index, batch in enumerate(batches, start=1):
            self.emit(
                item=item,
                atom_kind="tfidf_topic",
                source_family=TFIDF_TOPICS,
                observable_axes=_bank_axis(bank, path="tfidf.topic.bank"),
                branch=f"topic_banks.{bank}.topics",
                parent_collection_sha256=parent,
                content={
                    "bank": bank,
                    "topic_id": topic_id,
                    "terms": list(batch),
                    "member_batch_index": batch_index,
                    "member_batch_count": len(batches),
                    "full_member_count": len(terms),
                },
            )

    def _orphan_cluster(
        self, item: FoldEvidenceInput, cluster: Mapping[str, Any], collection_key: str
    ) -> None:
        _closed_keys(
            cluster,
            allowed={
                "cluster_id",
                "evidence_kind",
                "terms",
                "member_terms",
                "supporting_terms",
                "seed_term",
                "fit_rank",
                "maximum_abs_fit_signed_score",
                "grouping_method",
            },
            path="tfidf.orphan.cluster",
        )
        cluster_id = _short_string(cluster.get("cluster_id"), path="tfidf.orphan.cluster_id")
        raw_terms = cluster.get("terms")
        if raw_terms is None:
            raw_terms = cluster.get("member_terms", cluster.get("supporting_terms"))
        terms: list[dict[str, Any]] = []
        allowed_term = {
            "term",
            "feature",
            "ngram",
            "combined_importance",
            "fit_rank",
            "fit_signed_score",
            "lexical_similarity_to_seed",
            "signed_score",
            "support_control",
            "support_treated",
        }
        for index, raw in enumerate(_sequence(raw_terms, path="tfidf.orphan.terms")):
            row = {"term": raw} if isinstance(raw, str) else _mapping(raw, path="orphan.term")
            _closed_keys(row, allowed=allowed_term, path="orphan.term")
            phrase_field = next(
                (key for key in ("term", "feature", "ngram") if row.get(key) is not None),
                None,
            )
            if phrase_field is None:
                raise ValueError("orphan term has no phrase")
            projected: dict[str, Any] = {
                "term": _concept_phrase(row[phrase_field], path=f"orphan.terms[{index}].term")
            }
            for key in sorted(allowed_term - {"term", "feature", "ngram"}):
                if row.get(key) is not None:
                    projected[key] = _finite_number(row[key], path=f"orphan.term.{key}")
            terms.append(projected)
        parent = _sha256_json(
            {"cluster_id": cluster_id, "terms": sorted(terms, key=canonical_json)}
        )
        batches = _member_batches(
            terms,
            semantic_member_batch_size=self.semantic_member_batch_size,
        )
        for batch_index, batch in enumerate(batches, start=1):
            self.emit(
                item=item,
                atom_kind="tfidf_orphan_ngram_cluster",
                source_family=TFIDF_ORPHAN_NGRAMS,
                observable_axes=(HETEROGENEITY_AXIS,),
                branch=f"effect_orphan_ngram_branch.{collection_key}",
                parent_collection_sha256=parent,
                content={
                    "cluster_id": cluster_id,
                    "terms": list(batch),
                    "member_batch_index": batch_index,
                    "member_batch_count": len(batches),
                    "full_member_count": len(terms),
                },
            )

    def _neural_queries(self, item: FoldEvidenceInput) -> None:
        raw_queries = item.payload.get("query_evidence")
        if raw_queries is None:
            raw_queries = item.payload.get("queries", item.payload.get("evidence"))
        allowed_query = {
            "bank",
            "mechanical_role",
            "query_id",
            "statistical_gate_applied",
            "top_chunks",
            "top_contrastive_ngrams",
            "contrastive_ngrams",
            "fit_standardized_score",
            "member_count",
            "member_subfolds",
        }
        for query_index, raw in enumerate(
            _sequence(raw_queries, path="neural_query.query_evidence")
        ):
            query = _mapping(raw, path=f"neural_query[{query_index}]")
            _closed_keys(query, allowed=allowed_query, path=f"neural_query[{query_index}]")
            if _sequence(query.get("top_chunks"), path="neural_query.top_chunks"):
                raise ValueError("row-level neural-query excerpts cannot enter discovery")
            bank = _short_string(query.get("bank"), path="neural_query.bank").casefold()
            axes = _bank_axis(bank, path="neural_query.bank")
            expected_mechanical_role = "effect_modifier" if bank == "effect" else "confounder"
            if (
                query.get("mechanical_role") is not None
                and str(query.get("mechanical_role")) != expected_mechanical_role
            ):
                raise ValueError("neural-query mechanical role does not match its bank")
            query_id = _short_string(query.get("query_id"), path="neural_query.query_id")
            gate = query.get("statistical_gate_applied")
            if not isinstance(gate, bool):
                raise ValueError("neural_query.statistical_gate_applied must be boolean")
            witnesses: list[dict[str, Any]] = []
            for branch in ("top_contrastive_ngrams", "contrastive_ngrams"):
                for term_index, raw_term in enumerate(
                    _sequence(query.get(branch), path=f"neural_query.{branch}")
                ):
                    row = _mapping(raw_term, path=f"neural_query.{branch}[{term_index}]")
                    _closed_keys(
                        row,
                        allowed={
                            "term",
                            "feature",
                            "ngram",
                            "tfidf_contrast",
                            "loading",
                            "signed_score",
                            "fit_signed_score",
                            "standardized_score",
                            "rank",
                            "fit_rank",
                        },
                        path="neural_query.witness",
                    )
                    phrase_field = next(
                        (key for key in ("term", "feature", "ngram") if row.get(key) is not None),
                        None,
                    )
                    if phrase_field is None:
                        raise ValueError("neural-query witness has no phrase")
                    projected = {
                        "branch": branch,
                        "term": _concept_phrase(
                            row[phrase_field], path="neural_query.witness.term"
                        ),
                    }
                    for score_key in (
                        "tfidf_contrast",
                        "loading",
                        "signed_score",
                        "fit_signed_score",
                        "standardized_score",
                        "rank",
                        "fit_rank",
                    ):
                        if row.get(score_key) is not None:
                            projected[score_key] = _finite_number(
                                row[score_key], path=f"neural_query.witness.{score_key}"
                            )
                    witnesses.append(projected)
            parent = _sha256_json(
                {
                    "bank": bank,
                    "query_id": query_id,
                    "statistical_gate_applied": gate,
                    "witnesses": sorted(witnesses, key=canonical_json),
                }
            )
            if witnesses:
                batches = _member_batches(
                    witnesses,
                    semantic_member_batch_size=self.semantic_member_batch_size,
                )
                for batch_index, batch in enumerate(batches, start=1):
                    self.emit(
                        item=item,
                        atom_kind="neural_query_semantic_witnesses",
                        source_family=NEURAL_QUERY_MOMENTS,
                        observable_axes=axes,
                        branch="query_evidence.semantic_witnesses",
                        parent_collection_sha256=parent,
                        content={
                            "bank": bank,
                            "query_id": query_id,
                            "statistical_gate_applied": gate,
                            "semantic_witnesses": list(batch),
                            "member_batch_index": batch_index,
                            "member_batch_count": len(batches),
                            "full_member_count": len(witnesses),
                        },
                    )
            else:
                self.empty_lexical_query_count += 1
            metrics = {
                key: query[key]
                for key in ("fit_standardized_score", "member_count")
                if query.get(key) is not None
            }
            if metrics:
                self.emit_summary(
                    item=item,
                    source_family=NEURAL_QUERY_MOMENTS,
                    observable_axes=axes,
                    coordinate=query_id,
                    metrics=metrics,
                )


def build_role_neutral_evidence_catalog(
    evidence_inputs: Sequence[FoldEvidenceInput],
    *,
    semantic_member_batch_size: int = (
        DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK
    ),
    require_all_source_kinds: bool = True,
    require_all_architecture_families: bool = True,
    require_upstream_completeness: bool = True,
) -> RoleNeutralEvidenceCatalog:
    """Build a complete catalog with no role-first or numerical grounding path."""

    return _CatalogBuilder(
        evidence_inputs,
        semantic_member_batch_size=semantic_member_batch_size,
    ).build(
        require_all_source_kinds=require_all_source_kinds,
        require_all_architecture_families=require_all_architecture_families,
        require_upstream_completeness=require_upstream_completeness,
    )


def _contains_member_id(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) == "member_id" or _contains_member_id(child) for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_member_id(child) for child in value)
    return False


def _native_semantic_projection(
    value: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> dict[str, Any]:
    """Project readable scalar labels while the canonical native JSON stays authoritative."""

    projected: dict[str, Any] = {}
    for key in keys:
        child = value.get(key)
        if child is None or isinstance(child, (str, bool, int, float)):
            if child is not None:
                projected[key] = _clone(child)
    return projected


def _native_evidence_unit(
    *,
    source_record: Mapping[str, Any],
    source_record_index: int,
    native_record: Mapping[str, Any],
    native_record_index: int,
    semantic_projection: Mapping[str, Any],
    proof_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal one native producer record inside one catalog semantic member.

    Native JSON is stored canonically as a string.  That keeps upstream
    catalog-local ``member_id`` fields distinguishable from the new catalog's
    member IDs while retaining the exact producer object without deletion or
    key rewriting.
    """

    source = _clone(source_record)
    native = _clone(native_record)
    context = None if proof_context is None else _clone(proof_context)
    body = {
        "schema_version": NATIVE_ROLE_NEUTRAL_UNIT_SCHEMA_VERSION,
        "source_record_index": int(source_record_index),
        "source_record_sha256": _sha256_json(source),
        "native_record_index": int(native_record_index),
        "native_record_sha256": _sha256_json(native),
        "native_record_json": canonical_json(native),
        "native_proof_context_json": (
            None if context is None else canonical_json(context)
        ),
        "semantic_projection": _clone(semantic_projection),
    }
    return {**body, "native_unit_sha256": _sha256_json(body)}


def _validate_native_evidence_unit(value: Mapping[str, Any]) -> dict[str, Any]:
    unit = _clone(value)
    expected = {
        "schema_version",
        "source_record_index",
        "source_record_sha256",
        "native_record_index",
        "native_record_sha256",
        "native_record_json",
        "native_proof_context_json",
        "semantic_projection",
        "native_unit_sha256",
    }
    if set(unit) != expected:
        raise ValueError("native role-neutral evidence unit is not a closed schema")
    if unit.get("schema_version") != NATIVE_ROLE_NEUTRAL_UNIT_SCHEMA_VERSION:
        raise ValueError("native role-neutral evidence unit changed schema")
    for key in ("source_record_index", "native_record_index"):
        if (
            isinstance(unit.get(key), bool)
            or not isinstance(unit.get(key), int)
            or int(unit[key]) < 0
        ):
            raise ValueError(f"native role-neutral evidence unit has invalid {key}")
    for key in (
        "source_record_sha256",
        "native_record_sha256",
        "native_unit_sha256",
    ):
        if _SHA256.fullmatch(str(unit.get(key) or "")) is None:
            raise ValueError(f"native role-neutral evidence unit has invalid {key}")
    native_json = unit.get("native_record_json")
    if not isinstance(native_json, str):
        raise ValueError("native role-neutral evidence unit lacks canonical native JSON")
    try:
        native_record = json.loads(native_json)
    except json.JSONDecodeError as exc:
        raise ValueError("native role-neutral evidence unit contains invalid JSON") from exc
    if (
        not isinstance(native_record, dict)
        or canonical_json(native_record) != native_json
        or _sha256_json(native_record) != unit["native_record_sha256"]
    ):
        raise ValueError("native role-neutral evidence unit record does not authenticate")
    proof_json = unit.get("native_proof_context_json")
    if proof_json is not None:
        if not isinstance(proof_json, str):
            raise ValueError("native role-neutral proof context must be canonical JSON or null")
        try:
            proof = json.loads(proof_json)
        except json.JSONDecodeError as exc:
            raise ValueError("native role-neutral proof context is invalid JSON") from exc
        if not isinstance(proof, dict) or canonical_json(proof) != proof_json:
            raise ValueError("native role-neutral proof context is not canonical")
    if not isinstance(unit.get("semantic_projection"), Mapping):
        raise ValueError("native role-neutral evidence unit lacks a semantic projection")
    body = {key: child for key, child in unit.items() if key != "native_unit_sha256"}
    if _sha256_json(body) != unit["native_unit_sha256"]:
        raise ValueError("native role-neutral evidence unit does not self-authenticate")
    return unit


def _native_units_in(value: Any) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if value.get("schema_version") == NATIVE_ROLE_NEUTRAL_UNIT_SCHEMA_VERSION:
            units.append(_validate_native_evidence_unit(value))
            return units
        for child in value.values():
            units.extend(_native_units_in(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            units.extend(_native_units_in(child))
    return units


def _positive_native_integer(value: Any, *, path: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    minimum = 0 if allow_zero else 1
    if int(value) < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return int(value)


def _native_finite(value: Any, *, path: str) -> int | float:
    return _finite_number(value, path=path)


def _native_batched_atoms(
    *,
    family: str,
    atom_kind: str,
    source_kind: str,
    axes: tuple[str, ...],
    base_content: Mapping[str, Any],
    collection_key: str,
    members: Sequence[Mapping[str, Any]],
    semantic_member_batch_size: int,
) -> list[dict[str, Any]]:
    batches = _member_batches(
        members,
        semantic_member_batch_size=semantic_member_batch_size,
    )
    if not batches:
        raise ValueError(f"{family} native evidence group has no semantic members")
    output: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches, start=1):
        output.append(
            {
                "atom_kind": atom_kind,
                "source_kind": source_kind,
                "observable_axes": list(axes),
                "content": {
                    **_clone(base_content),
                    collection_key: list(batch),
                    "member_batch_index": batch_index,
                    "member_batch_count": len(batches),
                    "full_member_count": len(members),
                },
            }
        )
    return output


def _adapt_native_bow_payload(
    evidence: Sequence[Mapping[str, Any]],
    *,
    family: str,
    semantic_member_batch_size: int,
) -> list[dict[str, Any]]:
    objectives = {
        BOW_NUISANCE: {
            "treatment_nuisance": "treatment_positive",
            "outcome_nuisance": "outcome_positive",
        },
        BOW_R_LOSS: {
            "effect_pseudo_target": "pseudo_target_positive",
            "effect_weighted_r": "pseudo_target_positive",
        },
    }[family]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for source_index, raw in enumerate(evidence):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{family} native BoW record must be an object")
        row = _clone(raw)
        witness_kind = str(row.get("witness_kind") or "")
        expected_keys = {
            "objective",
            "view_name",
            "fold",
            "witness_kind",
            *(
                {"feature_index", "term", "idf"}
                if witness_kind == "fitted_tfidf_term"
                else {"constant_prediction"}
                if witness_kind == "constant_fit"
                else {"__unknown_witness_kind__"}
            ),
        }
        if set(row) != expected_keys:
            raise ValueError(f"{family} native BoW evidence schema changed")
        objective = str(row.get("objective") or "")
        view_name = str(row.get("view_name") or "")
        fold = _positive_native_integer(row.get("fold"), path="native BoW fold")
        if objective not in objectives or not view_name.strip():
            raise ValueError(f"{family} native BoW objective/view changed")
        projection = _native_semantic_projection(
            row,
            keys=(
                "witness_kind",
                "objective",
                "view_name",
                "fold",
                "term",
                "idf",
                "constant_prediction",
            ),
        )
        if witness_kind == "fitted_tfidf_term":
            _positive_native_integer(
                row.get("feature_index"),
                path="native BoW feature_index",
                allow_zero=True,
            )
            if not isinstance(row.get("term"), str) or not row["term"].strip():
                raise ValueError("native BoW term must be a non-empty string")
            _native_finite(row.get("idf"), path="native BoW idf")
        else:
            _native_finite(
                row.get("constant_prediction"),
                path="native BoW constant_prediction",
            )
        groups[(objective, view_name, fold)].append(
            _native_evidence_unit(
                source_record=row,
                source_record_index=source_index,
                native_record=row,
                native_record_index=0,
                semantic_projection=projection,
            )
        )
    output: list[dict[str, Any]] = []
    for (objective, view_name, fold), members in sorted(groups.items()):
        evidence_type = objectives[objective]
        source = (
            f"ensemble_r.{view_name}.{evidence_type}"
            if family == BOW_R_LOSS and view_name.startswith("ensemble_r__")
            else f"{view_name}.{evidence_type}"
        )
        group = {
            "view_name": view_name,
            "evidence_type": evidence_type,
            "source": source,
            "native_objective": objective,
            "native_fold": fold,
        }
        observed_family, axes = _classify_bow(group)
        if observed_family != family:
            raise RuntimeError("native BoW adapter changed the source family")
        output.extend(
            _native_batched_atoms(
                family=family,
                atom_kind="bow_term_group",
                source_kind=LEGACY_ALL_SOURCE,
                axes=axes,
                base_content={
                    "architecture_encoder": "bow",
                    "group": group,
                },
                collection_key="terms",
                members=members,
                semantic_member_batch_size=semantic_member_batch_size,
            )
        )
    return output


def _adapt_native_htr_payload(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = {
        "witness_kind",
        "schema_version",
        "stage",
        "objective",
        "fold",
        "fit_note_position",
        "fit_row_id",
        "chunk_index",
        "chunk_text",
        "chunk_sha256",
        "attention",
        "readable_token_spans",
        "readable_span_policy",
        "token_inventory_content_sha256",
    }
    output: list[dict[str, Any]] = []
    for source_index, raw in enumerate(evidence):
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError("native HTR attention evidence schema changed")
        row = _clone(raw)
        stage = str(row.get("stage") or "")
        if (
            row.get("witness_kind") != "complete_htr_chunk_attention"
            or row.get("schema_version")
            != ROLE_NEUTRAL_HTR_CHUNK_EVIDENCE_SCHEMA
            or stage not in {"nuisance", "effect_modifier"}
            or not isinstance(row.get("objective"), str)
            or not row["objective"].strip()
        ):
            raise ValueError("native HTR evidence changed its witness semantics")
        for key in ("fold",):
            _positive_native_integer(row.get(key), path=f"native HTR {key}")
        for key in ("fit_note_position", "fit_row_id", "chunk_index"):
            _positive_native_integer(
                row.get(key),
                path=f"native HTR {key}",
                allow_zero=True,
            )
        chunk_text = row.get("chunk_text")
        if not isinstance(chunk_text, str) or not chunk_text:
            raise ValueError("native HTR chunk text must be non-empty")
        if hashlib.sha256(chunk_text.encode("utf-8")).hexdigest() != row.get(
            "chunk_sha256"
        ):
            raise ValueError("native HTR chunk digest does not authenticate")
        _native_finite(row.get("attention"), path="native HTR attention")
        spans = row.get("readable_token_spans")
        policy = row.get("readable_span_policy")
        token_inventory_sha256 = str(
            row.get("token_inventory_content_sha256") or ""
        )
        if (
            not isinstance(spans, list)
            or not isinstance(policy, Mapping)
            or policy.get("special_tokens_excluded") is not True
            or policy.get("complete_raw_inventory_retained") is not True
            or policy.get("overlapping_chunk_occurrences_retained") is not True
            or _SHA256.fullmatch(token_inventory_sha256) is None
        ):
            raise ValueError("native HTR token-span evidence changed")
        for rank, span in enumerate(spans, start=1):
            if (
                not isinstance(span, Mapping)
                or span.get("schema_version")
                != ROLE_NEUTRAL_HTR_READABLE_SPAN_SCHEMA
                or int(span.get("selection_rank", 0)) != rank
                or not isinstance(span.get("text"), str)
                or not span["text"]
                or span.get(
                    "special_tokens_excluded_from_readable_projection"
                )
                is not True
                or span.get("raw_special_token_mass_retained_in_sidecar")
                is not True
                or not math.isclose(
                    _native_finite(
                        span.get("hierarchical_attention_score"),
                        path="native HTR hierarchical attention",
                    ),
                    _native_finite(
                        span.get("chunk_attention"),
                        path="native HTR span chunk attention",
                    )
                    * _native_finite(
                        span.get("token_attention"),
                        path="native HTR span token attention",
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError("native HTR readable token span changed")
        normalized_stage = "nuisance" if stage == "nuisance" else "effect"
        axes = (
            (TREATMENT_AXIS, OUTCOME_AXIS)
            if normalized_stage == "nuisance"
            else (HETEROGENEITY_AXIS,)
        )
        unit = _native_evidence_unit(
            source_record=row,
            source_record_index=source_index,
            native_record=row,
            native_record_index=0,
            semantic_projection={
                "chunk_text": chunk_text,
                "attention": row["attention"],
                "objective": row["objective"],
                "fold": row["fold"],
                "fit_note_position": row["fit_note_position"],
                "chunk_index": row["chunk_index"],
                "readable_token_spans": spans,
                "token_inventory_content_sha256": (
                    token_inventory_sha256
                ),
                "attention_interpretation": (
                    "attention_based_ranking_heuristic_not_causal_"
                    "contribution"
                ),
                "raw_token_inventory_location": (
                    "authenticated_zero_copy_htr_fit_state_sidecars"
                ),
            },
        )
        output.append(
            {
                "atom_kind": "htr_phrase",
                "source_kind": LEGACY_ALL_SOURCE,
                "observable_axes": list(axes),
                "content": {
                    "architecture_encoder": "htr",
                    "group": {
                        "stage": normalized_stage,
                        "meaning": row["objective"],
                    },
                    "phrase_evidence": unit,
                },
            }
        )
    return output


def _adapt_native_embedding_payload(
    evidence: Sequence[Mapping[str, Any]],
    *,
    family: str,
    semantic_member_batch_size: int,
) -> list[dict[str, Any]]:
    expected_atom_kind = (
        "tfidf_semantic_retrieval_contrast"
        if family == TFIDF_SEMANTIC_RETRIEVAL
        else "embedding_contrast"
    )
    output: list[dict[str, Any]] = []
    for source_index, raw in enumerate(evidence):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{family} native embedding evidence must be an object")
        row = _clone(raw)
        if set(row) not in (
            {"atom_kind", "source_kind", "observable_axes", "content"},
            {
                "atom_kind",
                "source_kind",
                "observable_axes",
                "content",
                "canonical_preflight_scope_reused",
                "canonical_preflight_atom_index",
            },
        ):
            raise ValueError(f"{family} native embedding evidence schema changed")
        if (
            row.get("atom_kind") != expected_atom_kind
            or row.get("source_kind") != LEGACY_ALL_SOURCE
            or not isinstance(row.get("content"), Mapping)
            or not isinstance(row.get("observable_axes"), list)
        ):
            raise ValueError(f"{family} native embedding architecture binding changed")
        axes = tuple(map(str, row["observable_axes"]))
        if axes != _ordered_axes(axes):
            raise ValueError(f"{family} native embedding axes are not canonical")
        content = row["content"]
        contrast = content.get("contrast")
        witnesses = content.get("concept_witnesses")
        if (
            not isinstance(contrast, Mapping)
            or not isinstance(witnesses, list)
            or not witnesses
            or any(not isinstance(member, Mapping) for member in witnesses)
        ):
            raise ValueError(f"{family} native embedding witnesses are malformed")
        structural_family, expected_axes = _classify_embedding(contrast)
        if expected_axes != axes or (
            family != TFIDF_SEMANTIC_RETRIEVAL
            and structural_family != family
        ):
            raise ValueError(f"{family} native embedding contrast changed family/axes")
        if family == TFIDF_SEMANTIC_RETRIEVAL:
            if (
                content.get("architecture_view")
                != SEMANTIC_RETRIEVAL_DERIVATION
                or content.get("source_passages_removed") is not True
            ):
                raise ValueError("native semantic retrieval evidence retains passages")
        elif content.get("architecture_view") != "embedding_contrast":
            raise ValueError("native embedding evidence changed architecture view")
        if content.get("all_source_chunks_accounted_once") not in (None, True):
            raise ValueError("native embedding source-chunk coverage is incomplete")
        if content.get("all_configured_semantic_terms_accounted_once") not in (
            None,
            True,
        ):
            raise ValueError("native embedding semantic-term coverage is incomplete")
        unit = _native_evidence_unit(
            source_record=row,
            source_record_index=source_index,
            native_record=row,
            native_record_index=0,
            semantic_projection={
                "contrast": _clone(contrast),
                "concept_witnesses_json": canonical_json(witnesses),
                "producer_member_batch_index": content.get("member_batch_index"),
                "producer_member_batch_count": content.get("member_batch_count"),
                "producer_full_member_count": content.get("full_member_count"),
            },
        )
        normalized_contrast = {
            **_clone(contrast),
            "native_source_record_index": source_index,
            "native_source_record_sha256": _sha256_json(row),
        }
        base = {
            "architecture_view": (
                SEMANTIC_RETRIEVAL_DERIVATION
                if family == TFIDF_SEMANTIC_RETRIEVAL
                else "embedding_contrast"
            ),
            "contrast": normalized_contrast,
        }
        if family == TFIDF_SEMANTIC_RETRIEVAL:
            base["source_passages_removed"] = True
        output.extend(
            _native_batched_atoms(
                family=family,
                atom_kind=expected_atom_kind,
                source_kind=LEGACY_ALL_SOURCE,
                axes=axes,
                base_content=base,
                collection_key="concept_witnesses",
                members=[unit],
                semantic_member_batch_size=semantic_member_batch_size,
            )
        )
    return output


def _adapt_native_tfidf_topic_payload(
    evidence: Sequence[Mapping[str, Any]],
    *,
    semantic_member_batch_size: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source_index, raw in enumerate(evidence):
        if not isinstance(raw, Mapping):
            raise ValueError("native TF-IDF topic evidence must be an object")
        row = _clone(raw)
        witness_kind = str(row.get("witness_kind") or "")
        if witness_kind not in {
            "fitted_consensus_nmf_topic_term",
            "fitted_topic_without_rendered_terms",
            "no_feasible_fitted_topic",
        }:
            raise ValueError("native TF-IDF topic witness kind changed")
        if witness_kind == "no_feasible_fitted_topic":
            if set(row) != {"witness_kind", "reason"}:
                raise ValueError("native no-topic witness schema changed")
            bank = "unavailable"
            topic_id = "no_feasible_fitted_topic"
        else:
            bank = str(row.get("bank") or "").casefold()
            topic_id = str(row.get("topic_id") or "")
            _bank_axis(bank, path="native TF-IDF topic bank")
            if not topic_id:
                raise ValueError("native TF-IDF topic ID is empty")
            if witness_kind == "fitted_topic_without_rendered_terms":
                if set(row) != {"bank", "topic_id", "witness_kind"}:
                    raise ValueError("native empty-topic witness schema changed")
            else:
                required = {
                    "bank",
                    "topic_id",
                    "topic_position",
                    "term_position",
                    "witness_kind",
                }
                if not required.issubset(row):
                    raise ValueError("native TF-IDF topic-term witness is incomplete")
                if not isinstance(row.get("term"), str) or not row["term"].strip():
                    raise ValueError("native TF-IDF topic-term witness has no term")
                _positive_native_integer(
                    row.get("topic_position"),
                    path="native topic_position",
                    allow_zero=True,
                )
                _positive_native_integer(
                    row.get("term_position"),
                    path="native term_position",
                    allow_zero=True,
                )
        projection = _native_semantic_projection(
            row,
            keys=(
                "witness_kind",
                "bank",
                "topic_id",
                "term",
                "feature",
                "loading",
                "signed_score",
                "reason",
            ),
        )
        groups[(bank, topic_id)].append(
            _native_evidence_unit(
                source_record=row,
                source_record_index=source_index,
                native_record=row,
                native_record_index=0,
                semantic_projection=projection,
            )
        )
    output: list[dict[str, Any]] = []
    for (bank, topic_id), members in sorted(groups.items()):
        axes = (
            _ordered_axes((TREATMENT_AXIS, OUTCOME_AXIS, HETEROGENEITY_AXIS))
            if bank == "unavailable"
            else _bank_axis(bank, path="native TF-IDF topic bank")
        )
        output.extend(
            _native_batched_atoms(
                family=TFIDF_TOPICS,
                atom_kind="tfidf_topic",
                source_kind=TFIDF_TOPIC_SOURCE,
                axes=axes,
                base_content={"bank": bank, "topic_id": topic_id},
                collection_key="terms",
                members=members,
                semantic_member_batch_size=semantic_member_batch_size,
            )
        )
    return output


def _adapt_native_tfidf_orphan_payload(
    evidence: Sequence[Mapping[str, Any]],
    *,
    semantic_member_batch_size: int,
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for source_index, raw in enumerate(evidence):
        if not isinstance(raw, Mapping):
            raise ValueError("native residual TF-IDF evidence must be an object")
        row = _clone(raw)
        witness_kind = str(row.get("witness_kind") or "")
        if witness_kind == "fit_side_residual_tfidf_ngram":
            required = {
                "witness_kind",
                "fit_rank",
                "represented_in_effect_topic",
                "feature",
            }
            if (
                not required.issubset(row)
                or row.get("represented_in_effect_topic") is not False
                or not isinstance(row.get("feature"), str)
                or not row["feature"].strip()
            ):
                raise ValueError("native residual TF-IDF n-gram witness changed")
            _positive_native_integer(row.get("fit_rank"), path="native residual fit_rank")
        elif witness_kind == "no_eligible_residual_tfidf_ngram":
            if set(row) != {"witness_kind", "reason"}:
                raise ValueError("native empty residual TF-IDF witness schema changed")
        else:
            raise ValueError("native residual TF-IDF witness kind changed")
        members.append(
            _native_evidence_unit(
                source_record=row,
                source_record_index=source_index,
                native_record=row,
                native_record_index=0,
                semantic_projection=_native_semantic_projection(
                    row,
                    keys=(
                        "witness_kind",
                        "fit_rank",
                        "feature",
                        "signed_score",
                        "combined_importance",
                        "reason",
                    ),
                ),
            )
        )
    return _native_batched_atoms(
        family=TFIDF_ORPHAN_NGRAMS,
        atom_kind="tfidf_orphan_ngram_cluster",
        source_kind=TFIDF_TOPIC_SOURCE,
        axes=(HETEROGENEITY_AXIS,),
        base_content={
            "cluster_id": "complete_native_fit_side_residual_tfidf_ngrams"
        },
        collection_key="terms",
        members=members,
        semantic_member_batch_size=semantic_member_batch_size,
    )


def _adapt_native_neural_query_payload(
    evidence: Sequence[Mapping[str, Any]],
    *,
    semantic_member_batch_size: int,
) -> list[dict[str, Any]]:
    expected = {
        "query_id",
        "bank",
        "mechanical_role",
        "statistical_gate_applied",
        "member_count",
        "fit_standardized_score",
        "top_chunks",
        "top_contrastive_ngrams",
    }
    output: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for source_index, raw in enumerate(evidence):
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError("native neural-query evidence schema changed")
        row = _clone(raw)
        bank = str(row.get("bank") or "").casefold()
        axes = _bank_axis(bank, path="native neural-query bank")
        query_id = str(row.get("query_id") or "")
        expected_role = "effect_modifier" if bank == "effect" else "confounder"
        if (
            not query_id
            or query_id in query_ids
            or row.get("mechanical_role") != expected_role
            or row.get("statistical_gate_applied") is not False
            or row.get("top_chunks") != []
            or not isinstance(row.get("top_contrastive_ngrams"), list)
        ):
            raise ValueError("native neural-query evidence changed safe query semantics")
        query_ids.add(query_id)
        unit = _native_evidence_unit(
            source_record=row,
            source_record_index=source_index,
            native_record=row,
            native_record_index=0,
            semantic_projection={
                **_native_semantic_projection(
                    row,
                    keys=(
                        "query_id",
                        "bank",
                        "mechanical_role",
                        "member_count",
                        "fit_standardized_score",
                    ),
                ),
                "top_contrastive_ngrams_json": canonical_json(
                    row["top_contrastive_ngrams"]
                ),
            },
        )
        output.extend(
            _native_batched_atoms(
                family=NEURAL_QUERY_MOMENTS,
                atom_kind="neural_query_semantic_witnesses",
                source_kind=NEURAL_QUERY_SOURCE,
                axes=axes,
                base_content={
                    "bank": bank,
                    "query_id": query_id,
                    "statistical_gate_applied": False,
                },
                collection_key="semantic_witnesses",
                members=[unit],
                semantic_member_batch_size=semantic_member_batch_size,
            )
        )
    return output


def _adapt_native_matched_pair_payload(
    evidence: Sequence[Mapping[str, Any]],
    *,
    semantic_member_batch_size: int,
) -> list[dict[str, Any]]:
    bow_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    htr_rows: list[dict[str, Any]] = []
    seen_subproducers: set[str] = set()
    source_seals: set[str] = set()
    for source_index, raw in enumerate(evidence):
        if not isinstance(raw, Mapping):
            raise ValueError("native matched-pair proof must be an object")
        source = _clone(raw)
        if set(source) != {
            "source_family_seal_content_sha256",
            "subproducer",
            "evidence_payload_sha256",
            "evidence_payload",
        }:
            raise ValueError("native matched-pair proof schema changed")
        source_seal = str(source.get("source_family_seal_content_sha256") or "")
        subproducer = str(source.get("subproducer") or "")
        payload = source.get("evidence_payload")
        if (
            _SHA256.fullmatch(source_seal) is None
            or subproducer not in {"bow", "htr"}
            or subproducer in seen_subproducers
            or _SHA256.fullmatch(str(source.get("evidence_payload_sha256") or ""))
            is None
            or not isinstance(payload, Mapping)
            or _sha256_json(payload) != source["evidence_payload_sha256"]
        ):
            raise ValueError("native matched-pair proof does not authenticate")
        seen_subproducers.add(subproducer)
        source_seals.add(source_seal)
        nested = _clone(payload)
        expected_kind = {
            "bow": "complete_fold_vocabulary_coefficients_v1",
            "htr": "complete_validation_pair_witnesses_v1",
        }[subproducer]
        if (
            set(nested)
            != {
                "subproducer",
                "evidence_kind",
                "top_k_applied",
                "text_truncation_applied",
                "atoms",
            }
            or nested.get("subproducer") != subproducer
            or nested.get("evidence_kind") != expected_kind
            or nested.get("top_k_applied") is not False
            or nested.get("text_truncation_applied") is not False
            or not isinstance(nested.get("atoms"), list)
            or not nested["atoms"]
        ):
            raise ValueError("native matched-pair subproducer evidence changed")
        proof_context = {
            "source_family_seal_content_sha256": source_seal,
            "subproducer": subproducer,
            "evidence_payload_sha256": source["evidence_payload_sha256"],
            "evidence_payload_without_atoms": {
                key: child for key, child in nested.items() if key != "atoms"
            },
        }
        for native_index, raw_atom in enumerate(nested["atoms"]):
            if not isinstance(raw_atom, Mapping):
                raise ValueError("native matched-pair atom must be an object")
            atom = _clone(raw_atom)
            if subproducer == "bow":
                if set(atom) != {
                    "fold",
                    "view_name",
                    "feature_index",
                    "term",
                    "control_delta_logit_coefficient",
                    "treated_delta_logit_coefficient",
                }:
                    raise ValueError("native matched-pair BoW atom schema changed")
                fold = _positive_native_integer(
                    atom.get("fold"),
                    path="native matched-pair BoW fold",
                )
                view_name = str(atom.get("view_name") or "")
                _positive_native_integer(
                    atom.get("feature_index"),
                    path="native matched-pair feature_index",
                    allow_zero=True,
                )
                if (
                    not view_name
                    or not isinstance(atom.get("term"), str)
                    or not atom["term"].strip()
                ):
                    raise ValueError("native matched-pair BoW term/view is invalid")
                for key in (
                    "control_delta_logit_coefficient",
                    "treated_delta_logit_coefficient",
                ):
                    _native_finite(atom.get(key), path=f"native matched-pair {key}")
                bow_groups[(fold, view_name)].append(
                    _native_evidence_unit(
                        source_record=source,
                        source_record_index=source_index,
                        native_record=atom,
                        native_record_index=native_index,
                        proof_context=proof_context,
                        semantic_projection=_clone(atom),
                    )
                )
            else:
                if set(atom) != {
                    "fold",
                    "pair_index",
                    "candidate_row_id",
                    "control_row_id",
                    "propensity_abs_diff",
                    "outcome_abs_diff",
                    "delta_logit",
                }:
                    raise ValueError("native matched-pair HTR atom schema changed")
                _positive_native_integer(
                    atom.get("fold"),
                    path="native matched-pair HTR fold",
                )
                for key in (
                    "pair_index",
                    "candidate_row_id",
                    "control_row_id",
                ):
                    _positive_native_integer(
                        atom.get(key),
                        path=f"native matched-pair {key}",
                        allow_zero=True,
                    )
                for key in (
                    "propensity_abs_diff",
                    "outcome_abs_diff",
                    "delta_logit",
                ):
                    _native_finite(atom.get(key), path=f"native matched-pair {key}")
                # Candidate/control row identifiers authenticate in the
                # upstream source record but are operational coordinates, not
                # clinical concept evidence.  The discovery-facing native
                # unit therefore carries the complete non-identifying pair
                # witness and the source-record hash, never the row IDs.
                safe_atom = {
                    key: child
                    for key, child in atom.items()
                    if key not in {"candidate_row_id", "control_row_id"}
                }
                htr_rows.append(
                    _native_evidence_unit(
                        source_record=source,
                        source_record_index=source_index,
                        native_record=safe_atom,
                        native_record_index=native_index,
                        proof_context=proof_context,
                        semantic_projection=_clone(safe_atom),
                    )
                )
    if seen_subproducers != {"bow", "htr"} or len(source_seals) != 1:
        raise ValueError("native matched-pair payload lacks one common subproducer proof")
    output: list[dict[str, Any]] = []
    for (fold, view_name), members in sorted(bow_groups.items()):
        normalized_view = f"pair_uplift__{view_name}"
        evidence_type = "uplift_pair_features"
        group = {
            "view_name": normalized_view,
            "evidence_type": evidence_type,
            "source": (
                f"matched_pair_uplift.{normalized_view}.{evidence_type}"
            ),
            "native_fold": fold,
            "native_view_name": view_name,
        }
        observed_family, axes = _classify_bow(group)
        if observed_family != MATCHED_PAIR_UPLIFT:
            raise RuntimeError("matched-pair native adapter changed family")
        output.extend(
            _native_batched_atoms(
                family=MATCHED_PAIR_UPLIFT,
                atom_kind="bow_term_group",
                source_kind=LEGACY_ALL_SOURCE,
                axes=axes,
                base_content={
                    "architecture_encoder": "bow",
                    "group": group,
                },
                collection_key="terms",
                members=members,
                semantic_member_batch_size=semantic_member_batch_size,
            )
        )
    for unit in htr_rows:
        output.append(
            {
                "atom_kind": "matched_pair_htr_phrase",
                "source_kind": LEGACY_ALL_SOURCE,
                "observable_axes": [PAIR_UPLIFT_AXIS],
                "content": {
                    "architecture_encoder": "htr",
                    "group": {
                        "stage": "pair_uplift",
                        "meaning": "complete_validation_pair_witnesses_v1",
                    },
                    "phrase_evidence": unit,
                },
            }
        )
    return output


def _normalize_cumulative_family_payload(
    raw_payload: Mapping[str, Any],
    *,
    family: str,
    semantic_member_batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _clone(raw_payload)
    if (
        family == HTR_NEURAL
        and payload.get("schema_version") != HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA
    ):
        raise ValueError(
            "raw per-chunk HTR evidence cannot enter Stage 2; "
            "the authenticated semantic aggregate payload is required"
        )
    expected_payload_keys = (
        {
            "schema_version",
            "family",
            "architecture_evidence",
            "semantic_aggregation",
            "content_sha256",
        }
        if family == HTR_NEURAL
        else {
            "schema_version",
            "family",
            "architecture_evidence",
        }
    )
    if set(payload) != expected_payload_keys:
        raise ValueError(f"{family} cumulative family payload is not a closed schema")
    evidence = payload.get("architecture_evidence")
    expected_schema = (
        HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA
        if family == HTR_NEURAL
        else NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION
    )
    semantic_aggregation = payload.get("semantic_aggregation")
    if (
        payload.get("schema_version") != expected_schema
        or payload.get("family") != family
        or not isinstance(evidence, list)
        or not evidence
        or (
            family == HTR_NEURAL
            and (
                not isinstance(semantic_aggregation, Mapping)
                or semantic_aggregation.get(
                    "no_top_k_sampling_or_truncation"
                )
                is not True
                or semantic_aggregation.get(
                    "every_semantic_aggregate_delivered_exactly_once"
                )
                is not True
                or semantic_aggregation.get("raw_token_arrays_copied")
                is not False
                or payload.get("content_sha256")
                != _sha256_json(
                    {
                        key: child
                        for key, child in payload.items()
                        if key != "content_sha256"
                    }
                )
            )
        )
    ):
        raise ValueError(f"{family} cumulative family payload is empty or misbound")
    def is_catalog_shaped(item: Any) -> bool:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {"atom_kind", "source_kind", "observable_axes", "content"}
            or item.get("atom_kind") not in _CUMULATIVE_FAMILY_ATOM_KINDS[family]
            or item.get("source_kind") != _CUMULATIVE_FAMILY_SOURCE_KIND[family]
            or not isinstance(item.get("content"), Mapping)
        ):
            return False
        content_keys = set(item["content"])
        expected_content_keys = set(
            _CUMULATIVE_ATOM_CONTENT_KEYS[str(item["atom_kind"])]
        )
        native_embedding_proof_keys = {
            "source_chunk_count",
            "all_source_chunks_accounted_once",
            "all_configured_semantic_terms_accounted_once",
        }
        if (
            family
            in {
                EMBEDDING_WHOLE_COHORT,
                EMBEDDING_CLUSTERED,
                TFIDF_SEMANTIC_RETRIEVAL,
            }
            and content_keys != expected_content_keys
            and native_embedding_proof_keys.intersection(content_keys)
        ):
            return False
        return True

    catalog_shaped = [is_catalog_shaped(item) for item in evidence]
    if any(catalog_shaped) and not all(catalog_shaped):
        raise ValueError(f"{family} cumulative payload mixes catalog and native schemas")
    if all(catalog_shaped):
        normalized = {
            "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
            "family": family,
            "architecture_evidence": _clone(evidence),
        }
        adapter_applied = False
    else:
        adapters = {
            BOW_NUISANCE: lambda: _adapt_native_bow_payload(
                evidence,
                family=BOW_NUISANCE,
                semantic_member_batch_size=semantic_member_batch_size,
            ),
            BOW_R_LOSS: lambda: _adapt_native_bow_payload(
                evidence,
                family=BOW_R_LOSS,
                semantic_member_batch_size=semantic_member_batch_size,
            ),
            HTR_NEURAL: lambda: (_ for _ in ()).throw(
                ValueError(
                    "raw per-chunk HTR evidence cannot enter Stage 2; "
                    "the authenticated semantic aggregate payload is required"
                )
            ),
            MATCHED_PAIR_UPLIFT: lambda: _adapt_native_matched_pair_payload(
                evidence,
                semantic_member_batch_size=semantic_member_batch_size,
            ),
            EMBEDDING_WHOLE_COHORT: lambda: _adapt_native_embedding_payload(
                evidence,
                family=EMBEDDING_WHOLE_COHORT,
                semantic_member_batch_size=semantic_member_batch_size,
            ),
            EMBEDDING_CLUSTERED: lambda: _adapt_native_embedding_payload(
                evidence,
                family=EMBEDDING_CLUSTERED,
                semantic_member_batch_size=semantic_member_batch_size,
            ),
            TFIDF_SEMANTIC_RETRIEVAL: lambda: _adapt_native_embedding_payload(
                evidence,
                family=TFIDF_SEMANTIC_RETRIEVAL,
                semantic_member_batch_size=semantic_member_batch_size,
            ),
            TFIDF_TOPICS: lambda: _adapt_native_tfidf_topic_payload(
                evidence,
                semantic_member_batch_size=semantic_member_batch_size,
            ),
            TFIDF_ORPHAN_NGRAMS: lambda: _adapt_native_tfidf_orphan_payload(
                evidence,
                semantic_member_batch_size=semantic_member_batch_size,
            ),
            NEURAL_QUERY_MOMENTS: lambda: _adapt_native_neural_query_payload(
                evidence,
                semantic_member_batch_size=semantic_member_batch_size,
            ),
        }
        normalized_evidence = adapters[family]()
        if not normalized_evidence:
            raise ValueError(f"{family} native payload normalization emitted no evidence")
        normalized = {
            "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
            "family": family,
            "architecture_evidence": sorted(
                normalized_evidence,
                key=canonical_json,
            ),
        }
        adapter_applied = True

    units = _native_units_in(normalized["architecture_evidence"])
    source_hashes = [_sha256_json(item) for item in evidence]
    if adapter_applied:
        observed_source_counts = Counter(
            int(unit["source_record_index"]) for unit in units
        )
        covered_source_indices = {
            int(unit["source_record_index"]) for unit in units
        }
        if (
            not units
            or covered_source_indices != set(range(len(evidence)))
            or any(
                unit["source_record_sha256"]
                != source_hashes[int(unit["source_record_index"])]
                for unit in units
            )
        ):
            raise RuntimeError(f"{family} native payload normalization lost a source record")
        if family == MATCHED_PAIR_UPLIFT:
            for source_index, source_record in enumerate(evidence):
                nested = source_record.get("evidence_payload")
                native_atoms = (
                    nested.get("atoms") if isinstance(nested, Mapping) else None
                )
                expected_count = len(native_atoms) if isinstance(native_atoms, list) else 0
                observed_indices = {
                    int(unit["native_record_index"])
                    for unit in units
                    if int(unit["source_record_index"]) == source_index
                }
                if (
                    observed_source_counts[source_index] != expected_count
                    or observed_indices != set(range(expected_count))
                ):
                    raise RuntimeError(
                        "matched-pair native payload normalization omitted or "
                        "duplicated a nested atom"
                    )
        elif (
            observed_source_counts
            != Counter({index: 1 for index in range(len(evidence))})
            or any(int(unit["native_record_index"]) != 0 for unit in units)
        ):
            raise RuntimeError(
                f"{family} native payload normalization omitted or duplicated a record"
            )
    audit = {
        "schema_version": NATIVE_ROLE_NEUTRAL_PAYLOAD_ADAPTER_SCHEMA_VERSION,
        "family": family,
        "adapter_applied": adapter_applied,
        "source_payload_sha256": _sha256_json(payload),
        "normalized_payload_sha256": _sha256_json(normalized),
        "source_record_count": len(evidence),
        "source_ordered_record_sha256": _sha256_json(source_hashes),
        "native_unit_count": len(units),
        "native_unit_multiset_sha256": _sha256_json(
            sorted(unit["native_unit_sha256"] for unit in units)
        ),
        "all_source_records_accounted_once_or_by_complete_nested_units": True,
        "native_units_self_authenticated": True,
        "selection_or_truncation_applied": False,
        "complete_token_attention_evidence": (
            None
            if family != HTR_NEURAL
            else {
                "schema_version": HTR_STAGE2_AGGREGATE_PAYLOAD_SCHEMA,
                "source_payload_content_sha256": (
                    semantic_aggregation["raw_evidence_reference"][
                        "source_payload_content_sha256"
                    ]
                ),
                "token_attention_package_content_sha256": (
                    semantic_aggregation["raw_evidence_reference"][
                        "token_attention_package_content_sha256"
                    ]
                ),
                "token_occurrence_count": int(
                    semantic_aggregation["raw_evidence_reference"][
                        "token_occurrence_count"
                    ]
                ),
                "chunk_interpretation_count": int(
                    semantic_aggregation[
                        "source_chunk_interpretation_count"
                    ]
                ),
                "readable_token_occurrence_count": int(
                    semantic_aggregation[
                        "eligible_readable_token_occurrence_count"
                    ]
                ),
                "special_token_occurrence_count": int(
                    semantic_aggregation[
                        "special_token_accounting_bucket"
                    ][
                        "occurrence_count"
                    ]
                ),
                "raw_sidecars_authenticated_in_zero_copy_source_graph": True,
                "aggregate_reverse_index_content_sha256": (
                    semantic_aggregation["reverse_index_reference"][
                        "reverse_index_manifest_content_sha256"
                    ]
                ),
                "fold_local_aggregate_count": int(
                    semantic_aggregation["fold_local_aggregate_count"]
                ),
                "cross_fold_aggregate_count": int(
                    semantic_aggregation["cross_fold_aggregate_count"]
                ),
                "model_facing_batch_count": int(
                    semantic_aggregation["model_facing_batch_count"]
                ),
                "selection_or_truncation_applied_to_raw_inventory": False,
                "selection_or_truncation_applied_to_semantic_aggregates": False,
            }
        ),
    }
    return normalized, audit


def _validate_cumulative_payload_item(
    value: Any,
    *,
    family: str,
    semantic_member_batch_size: int,
    batch_groups: dict[str, list[tuple[int, int, int, int]]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{family} cumulative family evidence item must be a mapping")
    item = _clone(value)
    if set(item) != {"atom_kind", "source_kind", "observable_axes", "content"}:
        raise ValueError(f"{family} cumulative family evidence item is not a closed schema")
    atom_kind = str(item.get("atom_kind") or "")
    source_kind = str(item.get("source_kind") or "")
    if atom_kind not in _CUMULATIVE_FAMILY_ATOM_KINDS[family]:
        raise ValueError(f"{family} cumulative payload contains another architecture's atom")
    if source_kind != _CUMULATIVE_FAMILY_SOURCE_KIND[family]:
        raise ValueError(f"{family} cumulative payload changed its canonical source kind")
    raw_axes = item.get("observable_axes")
    if not isinstance(raw_axes, list):
        raise TypeError(f"{family} cumulative observable_axes must be a list")
    axes = tuple(map(str, raw_axes))
    if axes != _ordered_axes(axes):
        raise ValueError(f"{family} cumulative observable_axes are not canonical")
    raw_content = item.get("content")
    if not isinstance(raw_content, Mapping):
        raise TypeError(f"{family} cumulative atom content must be a mapping")
    content = _clone(raw_content)
    if set(content) != set(_CUMULATIVE_ATOM_CONTENT_KEYS[atom_kind]):
        raise ValueError(f"{family} cumulative atom content is not a closed schema")
    if _contains_member_id(content):
        raise ValueError("cumulative family payloads cannot predeclare catalog-local member IDs")

    if atom_kind == "bow_term_group":
        if content.get("architecture_encoder") != "bow" or not isinstance(
            content.get("group"), Mapping
        ):
            raise ValueError("cumulative BoW atom changed its architecture binding")
        observed_family, observed_axes = _classify_bow(content["group"])
        if observed_family != family or observed_axes != axes:
            raise ValueError("cumulative BoW atom changed its family or observable axes")
    elif atom_kind == "embedding_contrast":
        if content.get("architecture_view") != "embedding_contrast" or not isinstance(
            content.get("contrast"), Mapping
        ):
            raise ValueError("cumulative embedding atom changed its architecture binding")
        observed_family, observed_axes = _classify_embedding(content["contrast"])
        if observed_family != family or observed_axes != axes:
            raise ValueError("cumulative embedding atom changed its family or observable axes")
    elif atom_kind == "tfidf_semantic_retrieval_contrast":
        if (
            content.get("architecture_view") != SEMANTIC_RETRIEVAL_DERIVATION
            or content.get("source_passages_removed") is not True
            or not isinstance(content.get("contrast"), Mapping)
        ):
            raise ValueError("cumulative semantic-retrieval atom is not the safe projection")
        _structural_family, observed_axes = _classify_embedding(content["contrast"])
        if observed_axes != axes:
            raise ValueError("cumulative semantic-retrieval observable axes changed")
    elif atom_kind in {
        "htr_phrase",
        "htr_semantic_aggregate_batch",
        "matched_pair_htr_phrase",
    }:
        expected_encoder = (
            "htr_token_attention_semantic_aggregation"
            if atom_kind == "htr_semantic_aggregate_batch"
            else "htr"
        )
        if content.get("architecture_encoder") != expected_encoder or not isinstance(
            content.get("group"), Mapping
        ):
            raise ValueError("cumulative HTR atom changed its architecture binding")
        group = content["group"]
        if set(group) != {"stage", "meaning"}:
            raise ValueError("cumulative HTR group is not a closed schema")
        stage = str(group.get("stage") or "")
        expected = {
            "nuisance": (HTR_NEURAL, (TREATMENT_AXIS, OUTCOME_AXIS)),
            "effect": (HTR_NEURAL, (HETEROGENEITY_AXIS,)),
            "pair_uplift": (MATCHED_PAIR_UPLIFT, (PAIR_UPLIFT_AXIS,)),
        }.get(stage)
        if expected != (family, axes):
            raise ValueError("cumulative HTR atom changed its family or observable axes")
        if atom_kind == "htr_semantic_aggregate_batch":
            batch = content.get("aggregate_batch")
            expected_batch_keys = {
                "schema_version",
                "stage",
                "objective",
                "batch_index",
                "batch_count",
                "aggregate_count",
                "aggregates",
                "raw_evidence_reference",
                "reverse_index_reference",
                "hierarchical_attention_interpretation",
                "complete_semantic_aggregate_delivery",
                "content_sha256",
            }
            if (
                not isinstance(batch, Mapping)
                or set(batch) != expected_batch_keys
                or batch.get("schema_version")
                != HTR_STAGE2_AGGREGATE_BATCH_SCHEMA
                or (
                    "nuisance"
                    if batch.get("stage") == "nuisance"
                    else "effect"
                    if batch.get("stage") == "effect_modifier"
                    else None
                )
                != stage
                or batch.get("objective") != group.get("meaning")
                or batch.get("hierarchical_attention_interpretation")
                != "ranking_heuristic_not_causal_attribution"
                or batch.get("complete_semantic_aggregate_delivery") is not True
                or batch.get("content_sha256")
                != _sha256_json(
                    {
                        key: child
                        for key, child in batch.items()
                        if key != "content_sha256"
                    }
                )
            ):
                raise ValueError("cumulative HTR semantic aggregate batch changed")
            aggregates = batch.get("aggregates")
            if (
                not isinstance(aggregates, list)
                or not aggregates
                or batch.get("aggregate_count") != len(aggregates)
                or isinstance(batch.get("batch_index"), bool)
                or not isinstance(batch.get("batch_index"), int)
                or int(batch["batch_index"]) < 1
                or isinstance(batch.get("batch_count"), bool)
                or not isinstance(batch.get("batch_count"), int)
                or int(batch["batch_count"]) < int(batch["batch_index"])
            ):
                raise ValueError("cumulative HTR aggregate batch accounting changed")
            aggregate_ids: set[str] = set()
            for aggregate in aggregates:
                if not isinstance(aggregate, Mapping):
                    raise ValueError("cumulative HTR semantic aggregate is malformed")
                aggregate_body = {
                    key: child
                    for key, child in aggregate.items()
                    if key != "content_sha256"
                }
                expected_aggregate_keys = {
                    "schema_version",
                    "aggregate_id",
                    "source_aggregate_content_sha256",
                    "stage",
                    "objective",
                    "normalized_focus_text",
                    "wordpiece_kind",
                    "semantic_occurrence_definition",
                    "occurrence_count",
                    "raw_token_occurrence_count",
                    "unique_note_count",
                    "unique_chunk_count",
                    "attention_summaries",
                    "fold_support",
                    "display_text_variant_count",
                    "display_text_variant_content_sha256",
                    "display_text_variants_authenticated_reference",
                    "context_windows",
                    "architecture_chunk_schema_version",
                    "hierarchical_attention_interpretation",
                    "complete_semantic_accounting",
                    "content_sha256",
                }
                fold_support = aggregate.get("fold_support")
                if (
                    set(aggregate) != expected_aggregate_keys
                    or aggregate.get("schema_version")
                    != HTR_STAGE2_MODEL_AGGREGATE_SCHEMA
                    or aggregate.get("stage") != batch.get("stage")
                    or aggregate.get("objective") != batch.get("objective")
                    or aggregate.get("content_sha256")
                    != _sha256_json(aggregate_body)
                    or aggregate.get("complete_semantic_accounting")
                    is not True
                    or aggregate.get("semantic_occurrence_definition")
                    != "every_eligible_non_special_raw_token_occurrence_v2"
                    or aggregate.get("raw_token_occurrence_count")
                    != aggregate.get("occurrence_count")
                    or not isinstance(
                        aggregate.get(
                            "display_text_variants_authenticated_reference"
                        ),
                        Mapping,
                    )
                    or not isinstance(
                        aggregate.get("architecture_chunk_schema_version"),
                        str,
                    )
                    or aggregate.get(
                        "hierarchical_attention_interpretation"
                    )
                    != "ranking_heuristic_not_causal_attribution"
                    or not isinstance(
                        aggregate.get("attention_summaries"),
                        Mapping,
                    )
                    or not isinstance(
                        aggregate.get("context_windows"),
                        list,
                    )
                    or not isinstance(fold_support, list)
                    or not fold_support
                    or any(
                        not isinstance(row, Mapping)
                        or set(row)
                        != {
                            "fold",
                            "fold_aggregate_id",
                            "fold_aggregate_content_sha256",
                            "occurrence_count",
                            "raw_token_occurrence_count",
                            "unique_note_count",
                            "unique_chunk_count",
                        }
                        for row in fold_support
                    )
                ):
                    raise ValueError("cumulative HTR semantic aggregate changed")
                aggregate_id = str(aggregate.get("aggregate_id") or "")
                if not aggregate_id or aggregate_id in aggregate_ids:
                    raise ValueError("cumulative HTR aggregate IDs are invalid")
                aggregate_ids.add(aggregate_id)
    elif atom_kind == "tfidf_topic":
        topic_bank = str(content.get("bank") or "")
        expected_topic_axes = (
            _ordered_axes((TREATMENT_AXIS, OUTCOME_AXIS, HETEROGENEITY_AXIS))
            if topic_bank == "unavailable"
            else _bank_axis(topic_bank, path="cumulative.tfidf.bank")
        )
        if expected_topic_axes != axes:
            raise ValueError("cumulative TF-IDF topic observable axes changed")
    elif atom_kind == "tfidf_orphan_ngram_cluster":
        if axes != (HETEROGENEITY_AXIS,):
            raise ValueError("cumulative orphan-ngram observable axes changed")
    elif atom_kind == "neural_query_semantic_witnesses":
        if _bank_axis(str(content.get("bank") or ""), path="cumulative.query.bank") != axes:
            raise ValueError("cumulative neural-query observable axes changed")

    collection_key = _CUMULATIVE_ATOM_COLLECTION_KEY.get(atom_kind)
    if collection_key is not None:
        members = content.get(collection_key)
        if (
            not isinstance(members, list)
            or not members
            or len(members) > semantic_member_batch_size
            or not all(isinstance(member, Mapping) for member in members)
        ):
            raise ValueError(f"{family} cumulative semantic-member batch is invalid")
        for member in members:
            if (
                member.get("schema_version")
                == NATIVE_ROLE_NEUTRAL_UNIT_SCHEMA_VERSION
            ):
                _validate_native_evidence_unit(member)
        raw_numbers = {
            name: content.get(name)
            for name in ("member_batch_index", "member_batch_count", "full_member_count")
        }
        if any(
            isinstance(number, bool) or not isinstance(number, int)
            for number in raw_numbers.values()
        ):
            raise ValueError(f"{family} cumulative member-batch metadata must be integer")
        batch_index = int(raw_numbers["member_batch_index"])
        batch_count = int(raw_numbers["member_batch_count"])
        full_count = int(raw_numbers["full_member_count"])
        if (
            batch_index < 1
            or batch_count < 1
            or full_count < 1
            or batch_index > batch_count
        ):
            raise ValueError(f"{family} cumulative member-batch accounting is incomplete")
        collection_identity = {
            key: child
            for key, child in content.items()
            if key not in {collection_key, "member_batch_index"}
        }
        group_sha256 = _sha256_json(
            {
                "family": family,
                "atom_kind": atom_kind,
                "source_kind": source_kind,
                "observable_axes": axes,
                "collection_identity": collection_identity,
            }
        )
        batch_groups[group_sha256].append(
            (batch_index, batch_count, full_count, len(members))
        )
    else:
        singular_key = _CUMULATIVE_ATOM_SINGULAR_KEY[atom_kind]
        if not isinstance(content.get(singular_key), Mapping):
            raise ValueError(f"{family} cumulative singular semantic member is invalid")
        singular = content[singular_key]
        if (
            singular.get("schema_version")
            == NATIVE_ROLE_NEUTRAL_UNIT_SCHEMA_VERSION
        ):
            _validate_native_evidence_unit(singular)

    item["observable_axes"] = list(axes)
    item["content"] = content
    return item


def _without_catalog_member_ids(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_catalog_member_ids(child)
            for key, child in value.items()
            if str(key) != "member_id"
        }
    if isinstance(value, (list, tuple)):
        return [_without_catalog_member_ids(child) for child in value]
    return _clone(value)


def _assembled_family_payload(
    catalog: RoleNeutralEvidenceCatalog,
    *,
    family: str,
) -> dict[str, Any]:
    items = [
        {
            "atom_kind": atom.atom_kind,
            "source_kind": atom.source_kind,
            "observable_axes": list(atom.observable_axes),
            "content": _without_catalog_member_ids(atom.content),
        }
        for atom in catalog.family_atoms(family)
    ]
    items.sort(key=canonical_json)
    return {
        "schema_version": NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION,
        "family": family,
        "architecture_evidence": items,
    }


def assemble_cumulative_spent_role_neutral_catalog(
    *,
    family_payload_by_family: Mapping[str, Mapping[str, Any]],
    family_artifact_sha256_by_family: Mapping[str, str],
    scope_binding_sha256: str,
    scope_id: str,
    outer_fold: int,
    provider_inner_fold: int,
    split_fingerprint: str,
    semantic_member_batch_size: int = (
        DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK
    ),
) -> RoleNeutralEvidenceCatalog:
    """Assemble one lossless hierarchy catalog from ten authenticated payloads.

    The cumulative producers independently regenerate and authenticate their
    native family payloads.  This boundary adds catalog-local evidence/member
    identifiers without changing any semantic item, then verifies an exact
    ten-way projection roundtrip before returning the hierarchy catalog.
    """

    batching = _semantic_member_batching_identity(
        semantic_member_batch_size
    )
    if (
        not isinstance(family_payload_by_family, Mapping)
        or set(family_payload_by_family) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
    ):
        raise ValueError("cumulative catalog assembly requires exactly ten family payloads")
    if (
        not isinstance(family_artifact_sha256_by_family, Mapping)
        or set(family_artifact_sha256_by_family) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET
    ):
        raise ValueError("cumulative catalog assembly requires exactly ten artifact hashes")
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int) or outer_fold < 1:
        raise ValueError("cumulative catalog outer_fold must be a positive integer")
    if (
        isinstance(provider_inner_fold, bool)
        or not isinstance(provider_inner_fold, int)
        or provider_inner_fold < 1
    ):
        raise ValueError("cumulative catalog provider_inner_fold must be a positive integer")
    expected_scope_id = f"outer_{outer_fold:03d}_hierarchy_epoch_{provider_inner_fold - 1:03d}"
    if scope_id != expected_scope_id or _CUMULATIVE_SCOPE_ID.fullmatch(scope_id) is None:
        raise ValueError("cumulative catalog scope_id is not canonical")
    if _SHA256.fullmatch(str(scope_binding_sha256 or "")) is None:
        raise ValueError("cumulative catalog scope binding must be a lowercase SHA-256")
    if _SHA256.fullmatch(str(split_fingerprint or "")) is None:
        raise ValueError("cumulative catalog split fingerprint must be a lowercase SHA-256")

    artifact_hashes = {
        family: str(family_artifact_sha256_by_family[family])
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    if any(_SHA256.fullmatch(value) is None for value in artifact_hashes.values()):
        raise ValueError("cumulative family artifact hashes must be lowercase SHA-256 values")
    if len(set(artifact_hashes.values())) != len(ACTIVE_STAGE1_CONCEPT_FAMILIES):
        raise ValueError("cumulative family artifacts must have distinct identities")

    batch_groups: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    payloads: dict[str, dict[str, Any]] = {}
    source_payloads: dict[str, dict[str, Any]] = {}
    native_adapter_audits: dict[str, dict[str, Any]] = {}
    rows: list[tuple[str, dict[str, Any]]] = []
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        raw_payload = family_payload_by_family[family]
        if not isinstance(raw_payload, Mapping):
            raise TypeError(f"{family} cumulative family payload must be a mapping")
        source_payloads[family] = _clone(raw_payload)
        payload, adapter_audit = _normalize_cumulative_family_payload(
            raw_payload,
            family=family,
            semantic_member_batch_size=semantic_member_batch_size,
        )
        native_adapter_audits[family] = adapter_audit
        evidence = payload.get("architecture_evidence")
        validated = [
            _validate_cumulative_payload_item(
                item,
                family=family,
                semantic_member_batch_size=semantic_member_batch_size,
                batch_groups=batch_groups,
            )
            for item in evidence
        ]
        if validated != sorted(validated, key=canonical_json):
            raise ValueError(f"{family} cumulative family payload is not canonically ordered")
        payload["architecture_evidence"] = validated
        payloads[family] = payload
        rows.extend((family, item) for item in validated)

    for group_rows in batch_groups.values():
        declared_counts = {
            batch_count
            for _batch_index, batch_count, _full_count, _member_count in group_rows
        }
        declared_full_counts = {
            full_count
            for _batch_index, _batch_count, full_count, _member_count in group_rows
        }
        indices = Counter(
            batch_index
            for batch_index, _batch_count, _full_count, _member_count in group_rows
        )
        if len(declared_counts) != 1 or len(declared_full_counts) != 1:
            raise ValueError(
                "cumulative semantic collection changed its batch or member count"
            )
        batch_count = next(iter(declared_counts))
        full_count = next(iter(declared_full_counts))
        observed_member_count = sum(
            member_count
            for _batch_index, _batch_count, _full_count, member_count in group_rows
        )
        if (
            set(indices) != set(range(1, batch_count + 1))
            or any(count != 1 for count in indices.values())
            or len(group_rows) != batch_count
            or observed_member_count != full_count
        ):
            raise ValueError("cumulative semantic collection omitted or duplicated a batch")

    multiplicities = Counter(
        canonical_json({"family": family, "item": item}) for family, item in rows
    )
    seen: Counter[str] = Counter()
    atoms: list[Stage1EvidenceAtom] = []
    for family, item in rows:
        signature = canonical_json({"family": family, "item": item})
        seen[signature] += 1
        atom_kind = str(item["atom_kind"])
        source_kind = str(item["source_kind"])
        axes = tuple(map(str, item["observable_axes"]))
        content = _clone(item["content"])
        collection_key = _CUMULATIVE_ATOM_COLLECTION_KEY.get(atom_kind)
        collection_identity = {
            key: child
            for key, child in content.items()
            if key not in {collection_key, "member_batch_index"}
        }
        parent_sha256 = _sha256_json(
            {
                "scope_binding_sha256": scope_binding_sha256,
                "family_artifact_sha256": artifact_hashes[family],
                "family": family,
                "atom_kind": atom_kind,
                "source_kind": source_kind,
                "observable_axes": axes,
                "collection_identity": collection_identity,
            }
        )
        identified_content, template_member_ids = _attach_member_ids(
            content,
            source_family=family,
            atom_kind=atom_kind,
            parent_collection_sha256=parent_sha256,
        )
        DiscoveryEvidenceItem(
            evidence_id="evidence_precommit",
            source_family=family,
            observable_axes=axes,
            content={"atom_kind": atom_kind, **identified_content},
            member_ids=template_member_ids,
        )
        origin = {
            "source_kind": source_kind,
            "artifact_id_sha256": artifact_hashes[family],
            "branch": "authenticated_cumulative_spent_family_payload",
            "parent_collection_sha256": parent_sha256,
            "scope_id": scope_id,
            "scope_binding_sha256": scope_binding_sha256,
            "multiplicity_ordinal": seen[signature],
            "multiplicity_count": multiplicities[signature],
        }
        origin_sha256 = _sha256_json(origin)
        bound_content, member_ids = _bind_member_ids_to_evidence_instance(
            identified_content,
            template_member_ids=template_member_ids,
            origin_sha256=origin_sha256,
        )
        content_sha256 = _sha256_json(bound_content)
        identity = {
            "atom_kind": atom_kind,
            "source_kind": source_kind,
            "source_family": family,
            "observable_axes": axes,
            "member_ids": member_ids,
            "split_fingerprint": split_fingerprint,
            "origin_sha256": origin_sha256,
            "content_sha256": content_sha256,
        }
        atoms.append(
            Stage1EvidenceAtom(
                evidence_id=f"evidence_{_sha256_json(identity)}",
                atom_kind=atom_kind,
                source_kind=source_kind,
                source_family=family,
                observable_axes=axes,
                member_ids=member_ids,
                split_fingerprint=split_fingerprint,
                origin_sha256=origin_sha256,
                content_sha256=content_sha256,
                _origin_json=canonical_json(origin),
                _content_json=canonical_json(bound_content),
            )
        )
    atoms.sort(key=lambda atom: atom.evidence_id)
    if len({atom.evidence_id for atom in atoms}) != len(atoms):
        raise RuntimeError("cumulative catalog evidence identifiers collided")
    catalog_identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "semantic_member_batching": batching,
        "outer_fold": outer_fold,
        "scope": "inner_train",
        "inner_fold": provider_inner_fold,
        "split_fingerprint": split_fingerprint,
        "atoms": [atom.as_dict() for atom in atoms],
        "non_grounding_numerical_summaries": [],
    }
    catalog_sha256 = _sha256_json(catalog_identity)
    family_atom_counts = Counter(atom.source_family for atom in atoms)
    family_member_counts = {
        family: sum(len(atom.member_ids) for atom in atoms if atom.source_family == family)
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    if any(
        family_atom_counts.get(family, 0) < 1 or family_member_counts[family] < 1
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise ValueError("cumulative catalog assembly lost an active architecture")
    audit = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "catalog_sha256": catalog_sha256,
        "assembly_kind": "authenticated_cumulative_spent_family_payloads_v1",
        "scope_id": scope_id,
        "scope_binding_sha256": scope_binding_sha256,
        "source_kinds": sorted({atom.source_kind for atom in atoms}),
        "family_artifact_sha256_by_family": artifact_hashes,
        "family_payload_sha256_by_family": {
            family: _sha256_json(payloads[family]) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        "source_family_payload_sha256_by_family": {
            family: _sha256_json(source_payloads[family])
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        "native_payload_adapter_by_family": native_adapter_audits,
        "native_payload_adapter_selection_or_truncation_applied": False,
        "inactive_sparse_query_present": False,
        "role_fields_emitted": False,
        "extraction_contracts_emitted": False,
        "temporal_policy_emitted": False,
        "global_top_k_applied": False,
        "semantic_member_batching": batching,
        "semantic_member_batch_size": semantic_member_batch_size,
        "semantic_member_batches_truncated": False,
        "atom_count": len(atoms),
        "atom_count_by_family": {
            family: family_atom_counts.get(family, 0) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        "semantic_member_count_by_family": family_member_counts,
        "all_architecture_families_required": True,
        "missing_architecture_families": [],
        "non_grounding_numerical_summary_count": 0,
        "non_grounding_summaries_visible_to_discovery": False,
        "upstream_completeness_required": True,
        "upstream_truncation_count": 0,
        "upstream_truncations": [],
        "empty_neural_query_lexical_witness_count": None,
        "family_payload_roundtrip_verified": True,
    }
    catalog = RoleNeutralEvidenceCatalog(
        outer_fold=outer_fold,
        scope="inner_train",
        inner_fold=provider_inner_fold,
        split_fingerprint=split_fingerprint,
        atoms=tuple(atoms),
        non_grounding_numerical_summaries=(),
        catalog_sha256=catalog_sha256,
        _audit_json=canonical_json(audit),
    )
    validate_role_neutral_catalog(catalog)
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        if _assembled_family_payload(catalog, family=family) != payloads[family]:
            raise RuntimeError(f"{family} changed during cumulative catalog assembly")
    return catalog


def validate_role_neutral_catalog(catalog: RoleNeutralEvidenceCatalog) -> None:
    if not isinstance(catalog, RoleNeutralEvidenceCatalog):
        raise TypeError("catalog must be RoleNeutralEvidenceCatalog")
    batching = catalog.audit.get("semantic_member_batching")
    if (
        not isinstance(batching, Mapping)
        or batching
        != _semantic_member_batching_identity(
            batching.get("semantic_member_batch_size")
        )
        or catalog.audit.get("semantic_member_batch_size")
        != batching["semantic_member_batch_size"]
    ):
        raise ValueError(
            "catalog lacks its configured semantic-member batching identity"
        )
    seen: set[str] = set()
    seen_members: set[str] = set()
    for atom in catalog.atoms:
        if atom.evidence_id in seen:
            raise ValueError("catalog contains duplicate evidence IDs")
        if not atom.member_ids:
            raise ValueError("catalog contains a concept-bearing atom with no semantic members")
        seen.add(atom.evidence_id)
        overlap = seen_members.intersection(atom.member_ids)
        if overlap:
            raise ValueError("catalog contains member IDs shared across evidence instances")
        seen_members.update(atom.member_ids)
        if atom.split_fingerprint != catalog.split_fingerprint:
            raise ValueError("atom split fingerprint differs from catalog")
        if _sha256_json(atom.origin) != atom.origin_sha256:
            raise ValueError("atom origin hash does not authenticate")
        if _sha256_json(atom.content) != atom.content_sha256:
            raise ValueError("atom content hash does not authenticate")
        identity = {
            "atom_kind": atom.atom_kind,
            "source_kind": atom.source_kind,
            "source_family": atom.source_family,
            "observable_axes": atom.observable_axes,
            "member_ids": atom.member_ids,
            "split_fingerprint": atom.split_fingerprint,
            "origin_sha256": atom.origin_sha256,
            "content_sha256": atom.content_sha256,
        }
        if f"evidence_{_sha256_json(identity)}" != atom.evidence_id:
            raise ValueError("atom ID does not authenticate its identity")
        atom.as_discovery_item()
    identity = {
        "schema_version": ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION,
        "semantic_member_batching": batching,
        "outer_fold": catalog.outer_fold,
        "scope": catalog.scope,
        "inner_fold": catalog.inner_fold,
        "split_fingerprint": catalog.split_fingerprint,
        "atoms": [atom.as_dict() for atom in catalog.atoms],
        "non_grounding_numerical_summaries": [
            row.as_dict() for row in catalog.non_grounding_numerical_summaries
        ],
    }
    if _sha256_json(identity) != catalog.catalog_sha256:
        raise ValueError("catalog SHA-256 does not authenticate")


@dataclass(frozen=True)
class ArchitectureEvidenceChunk:
    source_family: str
    chunk_index: int
    chunk_count: int
    chunk_id: str
    canonical_size_bytes: int
    _evidence_json: str = field(repr=False)

    @property
    def evidence(self) -> list[dict[str, Any]]:
        return json.loads(self._evidence_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARCHITECTURE_CHUNK_SCHEMA_VERSION,
            "source_family": self.source_family,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "chunk_id": self.chunk_id,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ArchitectureChunkPlan:
    catalog_sha256: str
    max_atoms_per_chunk: int
    max_bytes_per_chunk: int
    max_semantic_member_ids_per_chunk: int
    chunks: tuple[ArchitectureEvidenceChunk, ...]
    plan_sha256: str
    _audit_json: str = field(repr=False)

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARCHITECTURE_CHUNK_PLAN_SCHEMA_VERSION,
            "catalog_sha256": self.catalog_sha256,
            "max_atoms_per_chunk": self.max_atoms_per_chunk,
            "max_bytes_per_chunk": self.max_bytes_per_chunk,
            "max_semantic_member_ids_per_chunk": self.max_semantic_member_ids_per_chunk,
            "plan_sha256": self.plan_sha256,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
            "audit": self.audit,
        }


_PLACEHOLDER_CHUNK_ID = "chunk_" + ("0" * 64)
_SIZE_SENTINEL = 999_999_999


def _chunk_wire(
    *,
    source_family: str,
    index: int,
    count: int,
    chunk_id: str,
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": ARCHITECTURE_CHUNK_SCHEMA_VERSION,
        "source_family": source_family,
        "chunk_index": index,
        "chunk_count": count,
        "chunk_id": chunk_id,
        "evidence": list(evidence),
    }


def _conservative_chunk_size(source_family: str, evidence: Sequence[Mapping[str, Any]]) -> int:
    return len(
        canonical_json(
            _chunk_wire(
                source_family=source_family,
                index=_SIZE_SENTINEL,
                count=_SIZE_SENTINEL,
                chunk_id=_PLACEHOLDER_CHUNK_ID,
                evidence=evidence,
            )
        ).encode("utf-8")
    )


def _semantic_member_ids(evidence: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    member_ids: list[str] = []
    for item in evidence:
        raw_member_ids = item.get("member_ids")
        if not isinstance(raw_member_ids, list) or not raw_member_ids:
            raise ValueError("architecture evidence atom must contain a non-empty member_ids list")
        for member_id in raw_member_ids:
            if not isinstance(member_id, str) or not member_id.strip():
                raise ValueError("architecture evidence member IDs must be non-empty strings")
            member_ids.append(member_id)
    if len(member_ids) != len(set(member_ids)):
        raise ValueError("architecture chunk contains a semantic member ID more than once")
    return tuple(member_ids)


def build_complete_architecture_chunks(
    catalog: RoleNeutralEvidenceCatalog,
    *,
    max_atoms_per_chunk: int = DEFAULT_MAX_ATOMS_PER_ARCHITECTURE_CHUNK,
    max_bytes_per_chunk: int = DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK,
    max_semantic_member_ids_per_chunk: int = (
        DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK
    ),
) -> ArchitectureChunkPlan:
    """Pack complete atoms one architecture at a time, with zero truncation."""

    validate_role_neutral_catalog(catalog)
    if isinstance(max_atoms_per_chunk, bool) or not isinstance(max_atoms_per_chunk, int):
        raise ValueError("max_atoms_per_chunk must be a positive integer")
    if isinstance(max_bytes_per_chunk, bool) or not isinstance(max_bytes_per_chunk, int):
        raise ValueError("max_bytes_per_chunk must be a positive integer")
    if isinstance(max_semantic_member_ids_per_chunk, bool) or not isinstance(
        max_semantic_member_ids_per_chunk, int
    ):
        raise ValueError("max_semantic_member_ids_per_chunk must be a positive integer")
    if max_atoms_per_chunk < 1 or max_bytes_per_chunk < 1 or max_semantic_member_ids_per_chunk < 1:
        raise ValueError("chunk limits must be positive")
    provisional: list[tuple[str, list[list[dict[str, Any]]]]] = []
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        prompt_items = [
            atom.as_discovery_item().as_prompt_item() for atom in catalog.family_atoms(family)
        ]
        packed: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for item in prompt_items:
            item_member_count = len(_semantic_member_ids([item]))
            if item_member_count > max_semantic_member_ids_per_chunk:
                raise ValueError(
                    "one semantic atom exceeds max_semantic_member_ids_per_chunk; its "
                    "architecture adapter must emit smaller complete member batches before "
                    "any model call"
                )
            if _conservative_chunk_size(family, [item]) > max_bytes_per_chunk:
                raise ValueError(
                    "one semantic atom exceeds max_bytes_per_chunk; its architecture adapter "
                    "must emit smaller complete member batches before any model call"
                )
            candidate = [*current, item]
            if current and (
                family == HTR_NEURAL
                or len(candidate) > max_atoms_per_chunk
                or len(_semantic_member_ids(candidate)) > max_semantic_member_ids_per_chunk
                or _conservative_chunk_size(family, candidate) > max_bytes_per_chunk
            ):
                packed.append(current)
                current = [item]
            else:
                current = candidate
        if current:
            packed.append(current)
        provisional.append((family, packed))

    chunks: list[ArchitectureEvidenceChunk] = []
    for family, packed in provisional:
        for index, evidence in enumerate(packed, start=1):
            identity = {
                "schema_version": ARCHITECTURE_CHUNK_SCHEMA_VERSION,
                "catalog_sha256": catalog.catalog_sha256,
                "source_family": family,
                "chunk_index": index,
                "chunk_count": len(packed),
                "evidence": evidence,
            }
            chunk_id = f"chunk_{_sha256_json(identity)}"
            wire = _chunk_wire(
                source_family=family,
                index=index,
                count=len(packed),
                chunk_id=chunk_id,
                evidence=evidence,
            )
            size = len(canonical_json(wire).encode("utf-8"))
            if size > max_bytes_per_chunk:
                raise RuntimeError("canonical architecture chunk exceeds its byte bound")
            if len(_semantic_member_ids(evidence)) > max_semantic_member_ids_per_chunk:
                raise RuntimeError("architecture chunk exceeds its semantic-member bound")
            chunks.append(
                ArchitectureEvidenceChunk(
                    source_family=family,
                    chunk_index=index,
                    chunk_count=len(packed),
                    chunk_id=chunk_id,
                    canonical_size_bytes=size,
                    _evidence_json=canonical_json(evidence),
                )
            )
    chunks.sort(key=lambda row: (_FAMILY_ORDER[row.source_family], row.chunk_index))
    identity = {
        "schema_version": ARCHITECTURE_CHUNK_PLAN_SCHEMA_VERSION,
        "catalog_sha256": catalog.catalog_sha256,
        "max_atoms_per_chunk": max_atoms_per_chunk,
        "max_bytes_per_chunk": max_bytes_per_chunk,
        "max_semantic_member_ids_per_chunk": max_semantic_member_ids_per_chunk,
        "chunks": [chunk.as_dict() for chunk in chunks],
    }
    plan_sha = _sha256_json(identity)
    unaudited = ArchitectureChunkPlan(
        catalog_sha256=catalog.catalog_sha256,
        max_atoms_per_chunk=max_atoms_per_chunk,
        max_bytes_per_chunk=max_bytes_per_chunk,
        max_semantic_member_ids_per_chunk=max_semantic_member_ids_per_chunk,
        chunks=tuple(chunks),
        plan_sha256=plan_sha,
        _audit_json="{}",
    )
    audit = audit_complete_architecture_delivery(catalog, unaudited)
    return ArchitectureChunkPlan(
        catalog_sha256=catalog.catalog_sha256,
        max_atoms_per_chunk=max_atoms_per_chunk,
        max_bytes_per_chunk=max_bytes_per_chunk,
        max_semantic_member_ids_per_chunk=max_semantic_member_ids_per_chunk,
        chunks=tuple(chunks),
        plan_sha256=plan_sha,
        _audit_json=canonical_json(audit),
    )


def audit_complete_architecture_delivery(
    catalog: RoleNeutralEvidenceCatalog, plan: ArchitectureChunkPlan
) -> dict[str, Any]:
    validate_role_neutral_catalog(catalog)
    if plan.catalog_sha256 != catalog.catalog_sha256:
        raise ValueError("chunk plan is bound to another catalog")
    identity = {
        "schema_version": ARCHITECTURE_CHUNK_PLAN_SCHEMA_VERSION,
        "catalog_sha256": catalog.catalog_sha256,
        "max_atoms_per_chunk": plan.max_atoms_per_chunk,
        "max_bytes_per_chunk": plan.max_bytes_per_chunk,
        "max_semantic_member_ids_per_chunk": plan.max_semantic_member_ids_per_chunk,
        "chunks": [chunk.as_dict() for chunk in plan.chunks],
    }
    if _sha256_json(identity) != plan.plan_sha256:
        raise ValueError("chunk plan SHA-256 does not authenticate")
    expected = {
        (atom.evidence_id, atom.source_family): atom.as_discovery_item().as_prompt_item()
        for atom in catalog.atoms
    }
    delivered: dict[tuple[str, str], dict[str, Any]] = {}
    delivered_member_ids: list[str] = []
    member_count_by_chunk: list[dict[str, Any]] = []
    chunks_by_family: dict[str, list[ArchitectureEvidenceChunk]] = defaultdict(list)
    for chunk in plan.chunks:
        if chunk.source_family not in ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
            raise ValueError("chunk has an inactive architecture")
        if len(chunk.evidence) > plan.max_atoms_per_chunk:
            raise ValueError("chunk exceeds max_atoms_per_chunk")
        chunk_member_ids = _semantic_member_ids(chunk.evidence)
        if len(chunk_member_ids) > plan.max_semantic_member_ids_per_chunk:
            raise ValueError("chunk exceeds max_semantic_member_ids_per_chunk")
        delivered_member_ids.extend(chunk_member_ids)
        member_count_by_chunk.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_family": chunk.source_family,
                "semantic_member_id_count": len(chunk_member_ids),
            }
        )
        size = len(canonical_json(chunk.as_dict()).encode("utf-8"))
        if size != chunk.canonical_size_bytes or size > plan.max_bytes_per_chunk:
            raise ValueError("chunk canonical byte accounting is invalid")
        chunk_identity = {
            "schema_version": ARCHITECTURE_CHUNK_SCHEMA_VERSION,
            "catalog_sha256": catalog.catalog_sha256,
            "source_family": chunk.source_family,
            "chunk_index": chunk.chunk_index,
            "chunk_count": chunk.chunk_count,
            "evidence": chunk.evidence,
        }
        if f"chunk_{_sha256_json(chunk_identity)}" != chunk.chunk_id:
            raise ValueError("chunk ID does not authenticate its evidence")
        chunks_by_family[chunk.source_family].append(chunk)
        for row in chunk.evidence:
            if row.get("source_family") != chunk.source_family:
                raise ValueError("one architecture chunk contains mixed architectures")
            evidence_id = str(row.get("evidence_id") or "")
            key = (evidence_id, chunk.source_family)
            if key in delivered:
                raise ValueError("an evidence atom was delivered more than once")
            delivered[key] = _clone(row)
    for family, rows in chunks_by_family.items():
        ordered = sorted(rows, key=lambda row: row.chunk_index)
        if [row.chunk_index for row in ordered] != list(range(1, len(rows) + 1)):
            raise ValueError("architecture chunk indices are not contiguous")
        if any(row.chunk_count != len(rows) for row in ordered):
            raise ValueError("architecture chunk count metadata is inconsistent")
    if set(delivered) != set(expected):
        missing = sorted(set(expected) - set(delivered))
        extra = sorted(set(delivered) - set(expected))
        raise ValueError(f"chunk delivery differs from catalog; missing={missing}, extra={extra}")
    for key, expected_item in expected.items():
        if canonical_json(delivered[key]) != canonical_json(expected_item):
            raise ValueError("an evidence atom changed during architecture delivery")
    expected_member_ids = [member_id for atom in catalog.atoms for member_id in atom.member_ids]
    if len(delivered_member_ids) != len(set(delivered_member_ids)):
        raise ValueError("a semantic member ID was delivered more than once")
    if set(delivered_member_ids) != set(expected_member_ids):
        missing = sorted(set(expected_member_ids) - set(delivered_member_ids))
        extra = sorted(set(delivered_member_ids) - set(expected_member_ids))
        raise ValueError(
            "semantic member delivery differs from catalog; " f"missing={missing}, extra={extra}"
        )
    counts = Counter(family for _evidence_id, family in delivered)
    return {
        "schema_version": ARCHITECTURE_CHUNK_PLAN_SCHEMA_VERSION,
        "catalog_sha256": catalog.catalog_sha256,
        "plan_sha256": plan.plan_sha256,
        "catalog_atom_count": len(catalog.atoms),
        "observed_delivery_count": len(delivered),
        "catalog_semantic_member_id_count": len(expected_member_ids),
        "observed_semantic_member_id_delivery_count": len(delivered_member_ids),
        "max_semantic_member_ids_per_chunk": plan.max_semantic_member_ids_per_chunk,
        "observed_max_semantic_member_ids_per_chunk": max(
            (row["semantic_member_id_count"] for row in member_count_by_chunk),
            default=0,
        ),
        "semantic_member_id_count_by_chunk": member_count_by_chunk,
        "delivery_count_by_family": {
            family: counts.get(family, 0) for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        "all_catalog_atoms_delivered_exactly_once": True,
        "all_catalog_semantic_member_ids_delivered_exactly_once": True,
        "mixed_architecture_chunks_present": False,
        "htr_semantic_batches_are_one_per_interpretation_request": all(
            len(chunk.evidence) == 1
            for chunk in plan.chunks
            if chunk.source_family == HTR_NEURAL
        ),
        "htr_raw_chunk_atoms_delivered_to_model": False,
        "global_top_k_applied": False,
        "atoms_truncated": False,
        "arbitrary_structural_fragments_emitted": False,
        "non_grounding_numerical_summaries_delivered": False,
    }


__all__ = [
    "ARCHITECTURE_CHUNK_PLAN_SCHEMA_VERSION",
    "ARCHITECTURE_CHUNK_SCHEMA_VERSION",
    "DEFAULT_MAX_ATOMS_PER_ARCHITECTURE_CHUNK",
    "DEFAULT_MAX_BYTES_PER_ARCHITECTURE_CHUNK",
    "DEFAULT_MAX_SEMANTIC_MEMBER_IDS_PER_ARCHITECTURE_CHUNK",
    "NATIVE_FAMILY_CONCEPT_PAYLOAD_SCHEMA_VERSION",
    "NATIVE_ROLE_NEUTRAL_PAYLOAD_ADAPTER_SCHEMA_VERSION",
    "NATIVE_ROLE_NEUTRAL_UNIT_SCHEMA_VERSION",
    "NON_GROUNDING_SUMMARY_SCHEMA_VERSION",
    "ROLE_NEUTRAL_CATALOG_SCHEMA_VERSION",
    "SEMANTIC_MEMBER_BATCHING_SCHEMA_VERSION",
    "ArchitectureChunkPlan",
    "ArchitectureEvidenceChunk",
    "NonGroundingNumericalSummary",
    "RoleNeutralEvidenceCatalog",
    "Stage1EvidenceAtom",
    "audit_complete_architecture_delivery",
    "assemble_cumulative_spent_role_neutral_catalog",
    "build_complete_architecture_chunks",
    "build_role_neutral_evidence_catalog",
    "validate_role_neutral_catalog",
]
