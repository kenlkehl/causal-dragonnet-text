"""Explicit scientific fixtures for semantic-witness unit tests."""

from __future__ import annotations

from typing import Any

from oci.inference.review_spent_evidence_provider import (
    SemanticWitnessScientificConfig,
)


def semantic_witness_mapping(
    **overrides: Any,
) -> dict[str, Any]:
    retrieval_vectorizer = {
        "schema_version": "semantic_witness_tfidf_vectorizer_v1",
        "input": "content",
        "encoding": "utf-8",
        "decode_error": "strict",
        "strip_accents": "unicode",
        "lowercase": True,
        "preprocessor": None,
        "tokenizer": None,
        "analyzer": "word",
        "stop_words": None,
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range_min": 1,
        "ngram_range_max": 3,
        "max_df": 1.0,
        "min_df": 1,
        "max_features": None,
        "vocabulary": None,
        "binary": False,
        "dtype": "float64",
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True,
    }
    htr_vectorizer = {
        **retrieval_vectorizer,
        "min_df": 2,
    }
    value = {
        "schema_version": "semantic_witness_scientific_config_v1",
        "retrieval_vectorizer": retrieval_vectorizer,
        "htr_vectorizer": htr_vectorizer,
        "retrieval_min_positive_documents": 1,
        "retrieval_min_negative_documents": 1,
        "htr_min_unique_sources": 2,
        "htr_min_distinct_positive_documents": 2,
        "htr_min_positive_source_support": 2,
        "htr_attention_score_min_exclusive": 0.0,
        "htr_direction_score_min_exclusive": 0.0,
        "htr_require_strict_attention_separation": True,
        "retrieval_document_weighting_policy": "unweighted_document_mean_v1",
        "htr_source_weighting_policy": (
            "equal_source_mass_inverse_repeated_partition_count_v1"
        ),
        "htr_extreme_chunk_tie_policy": (
            "attention_then_chunk_index_then_casefolded_text_v1"
        ),
        "retrieval_ranking_policy": "absolute_score_desc_then_term_asc_v1",
        "htr_ranking_policy": (
            "score_desc_then_token_count_desc_then_term_asc_v1"
        ),
        "phrase_collision_policy": "highest_ranked_normalized_phrase_v1",
        "htr_term_overlap_policy": "retain_all_eligible_terms_v1",
        "retrieval_score_eligibility_policy": "all_finite_including_zero_v1",
        "maximum_retrieval_terms": None,
        "maximum_htr_terms": None,
        "maximum_explicit_phrases_per_attention_row": None,
        "overflow_policy": "fail_closed_without_selection_v1",
        "insufficient_source_policy": "return_empty_evidence_v1",
        "empty_vocabulary_policy": "return_empty_evidence_v1",
        "direction_numeric_dtype": "float64",
    }
    value.update(overrides)
    return value


def semantic_witness_config(
    **overrides: Any,
) -> SemanticWitnessScientificConfig:
    return SemanticWitnessScientificConfig.from_mapping(
        semantic_witness_mapping(**overrides)
    )
