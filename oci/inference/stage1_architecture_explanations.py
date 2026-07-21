"""Stable plain-language descriptions of the ten active Stage-1 families."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)

_EXPLANATIONS = {
    BOW_NUISANCE: (
        "Sparse bag-of-words models highlighted words and short phrases that were useful "
        "for predicting treatment assignment or the observed outcome. Interpret the visible "
        "language as possible patient measurements; importance alone does not establish a "
        "specific characteristic."
    ),
    BOW_R_LOSS: (
        "A sparse residual-effect model highlighted words and short phrases associated with "
        "variation in treatment response after its treatment and outcome predictions were "
        "accounted for. The text clues may describe patient characteristics, while their "
        "scores alone cannot name one."
    ),
    HTR_NEURAL: (
        "A trainable neural text model surfaced phrases used in its treatment, outcome, and "
        "residual-effect calculations and in matched-patient comparisons. Read each phrase for "
        "its ordinary clinical meaning and keep distinct measurements separate even when the "
        "model used them together."
    ),
    MATCHED_PAIR_UPLIFT: (
        "Comparisons of clinically similar patients with different treatments highlighted "
        "phrases and measurements associated with differences in outcomes between the paired "
        "patients. The supplied language is evidence about possible patient characteristics, "
        "not proof of an effect direction."
    ),
    EMBEDDING_WHOLE_COHORT: (
        "Whole-cohort text embeddings identified semantic contrasts among patient records. "
        "The supplied witnesses translate those contrasts back into readable words or phrases "
        "that may denote patient characteristics."
    ),
    EMBEDDING_CLUSTERED: (
        "Text embeddings were examined within groups of semantically similar patient records, "
        "then readable witnesses were selected for contrasts found inside those groups. "
        "Interpret the witnesses as local semantic clues without assuming that a cluster is a "
        "clinical category."
    ),
    TFIDF_SEMANTIC_RETRIEVAL: (
        "TF-IDF terms summarize the readable vocabulary that distinguishes records retrieved "
        "from opposing sides of semantic embedding contrasts. Use the terms to identify the "
        "patient measurement being discussed, while preserving ambiguity when the vocabulary "
        "is nonspecific."
    ),
    TFIDF_TOPICS: (
        "TF-IDF topic groups collect words and short phrases that repeatedly occur together in "
        "text patterns useful to the treatment, outcome, or residual-effect models. Review every "
        "topic member because one topic can contain several distinct patient measurements."
    ),
    TFIDF_ORPHAN_NGRAMS: (
        "Residual TF-IDF words and short phrases were retained because they carried signal but "
        "were not adequately represented by the main topic groups. Treat each visible n-gram as "
        "an independent clue rather than forcing it into a nearby topic."
    ),
    NEURAL_QUERY_MOMENTS: (
        "Learned neural queries searched patient text for recurring semantic patterns, and the "
        "supplied witnesses show the readable language associated with their aggregate response "
        "patterns. The witnesses can suggest patient characteristics; aggregate magnitudes alone "
        "cannot name them."
    ),
}

if tuple(_EXPLANATIONS) != ACTIVE_STAGE1_CONCEPT_FAMILIES:
    raise RuntimeError("Stage-1 family explanations must follow the active family contract")

PRODUCTION_STAGE1_FAMILY_EXPLANATIONS: Mapping[str, str] = MappingProxyType(_EXPLANATIONS)


def production_stage1_family_explanations() -> dict[str, str]:
    """Return a detached mutable copy in the canonical architecture order."""

    return dict(PRODUCTION_STAGE1_FAMILY_EXPLANATIONS)


__all__ = [
    "PRODUCTION_STAGE1_FAMILY_EXPLANATIONS",
    "production_stage1_family_explanations",
]
