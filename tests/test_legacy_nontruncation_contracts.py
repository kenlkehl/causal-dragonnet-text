from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.config import create_default_config
from oci.inference.agentic_attention_variable_forest import (
    _attention_evidence_snippet,
    _compact_token_spans,
    _tail_attention_positions,
)
from oci.inference.agentic_explicit_feature_forest import (
    AgenticFeatureProposal,
    _clinical_text_examples,
)
from oci.inference.all_evidence_fusion import (
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    LEGACY_ALL_SOURCE,
    NEURAL_QUERY_SOURCE,
    TFIDF_TOPIC_SOURCE,
    prepare_all_evidence_fusion,
)
from oci.inference.multi_model_agentic_forest import (
    _agentic_consistency_selected_proposals,
    _fallback_consistency_proposals,
    _require_complete_consistency_fallback,
)
from oci.inference.query_moment_evidence_adapter import (
    QueryMomentEvidenceAdapterConfig,
    _term_centered_excerpt,
)


def test_generated_legacy_config_does_not_prescribe_physical_gpu_ids(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "config.json"
    create_default_config(str(destination))
    generated = json.loads(destination.read_text(encoding="utf-8"))

    assert generated["device"] is None
    assert generated["gpu_ids"] is None


def _provenance() -> FoldEvidenceProvenance:
    return FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=(0, 1, 2),
        heldout_row_ids=(3,),
        artifact_id="nontruncation-test",
    )


def test_all_evidence_legacy_adapters_preserve_every_list_item_and_text_suffix():
    long_text = "prefix " + ("x" * 600) + " terminal-suffix"
    legacy = {
        "outer_fold": 1,
        "scope": "full_outer_train",
        "context": {
            "evidence_digest": {
                "confounders": {
                    "bow_blurbs": [
                        {
                            "source": "nuisance",
                            "rows": [
                                {"feature": f"bow-term-{index:02d}"} for index in range(25)
                            ],
                        }
                    ],
                    "embedding_chunks": [
                        {
                            "name": "whole-cohort",
                            "positive_aligned_chunks": [
                                {"text": f"embedding-{index:02d} {long_text}"}
                                for index in range(9)
                            ],
                        }
                    ],
                    "htr_blurbs": [
                        {
                            "stage": "nuisance",
                            "rows": [
                                {
                                    "attended_token_summary": long_text,
                                    "top_token_spans": [
                                        {"text": f"span-{index:02d}"} for index in range(15)
                                    ],
                                }
                            ],
                        }
                    ],
                },
                "effect_modifiers": {
                    "bow_blurbs": [],
                    "embedding_chunks": [],
                    "htr_blurbs": [],
                },
            }
        },
    }
    tfidf = {
        "outer_fold": 1,
        "scope": "full_outer_train",
        "discovery": {
            "topic_banks": {
                "effect": {
                    "topics": [
                        {
                            "topic_id": f"topic-{topic_index:02d}",
                            "terms": [
                                {"term": f"topic-{topic_index:02d}-term-{term_index:02d}"}
                                for term_index in range(18)
                            ],
                        }
                        for topic_index in range(14)
                    ]
                }
            }
        },
    }
    queries = {
        "outer_fold": 1,
        "scope": "outer_train",
        "query_evidence": [
            {
                "query_id": f"query-{query_index:02d}",
                "bank": "effect",
                "top_chunks": [
                    {"text": f"query-{query_index:02d}-chunk-{chunk_index:02d} {long_text}"}
                    for chunk_index in range(10)
                ],
                "top_contrastive_ngrams": [
                    {"term": f"query-{query_index:02d}-term-{term_index:02d}"}
                    for term_index in range(18)
                ],
            }
            for query_index in range(26)
        ],
    }

    request = prepare_all_evidence_fusion(
        [
            FoldEvidenceInput(LEGACY_ALL_SOURCE, legacy, _provenance()),
            FoldEvidenceInput(TFIDF_TOPIC_SOURCE, tfidf, _provenance()),
            FoldEvidenceInput(NEURAL_QUERY_SOURCE, queries, _provenance()),
        ]
    )
    contents = [block.content for block in request.evidence_blocks]

    bow = next(item for item in contents if item.get("kind") == "sparse_text_terms")
    assert len(bow["terms"]) == 25
    assert bow["terms"][-1]["term"] == "bow-term-24"
    embedding = next(item for item in contents if item.get("kind") == "embedding_contrast")
    assert len(embedding["chunks"]) == 9
    assert embedding["chunks"][-1]["text"].endswith("terminal-suffix")
    htr = next(item for item in contents if item.get("kind") == "neural_attention_summaries")
    assert "span-14" in htr["summaries"][0]
    assert "terminal-suffix" in htr["summaries"][0]
    topics = [item for item in contents if item.get("kind") == "tfidf_topic"]
    assert len(topics) == 14
    assert len(topics[-1]["terms"]) == 18
    query_blocks = [item for item in contents if item.get("kind") == "neural_query_moment"]
    assert len(query_blocks) == 26
    assert len(query_blocks[-1]["retrieved_training_excerpts"]) == 10
    assert len(query_blocks[-1]["contrastive_ngrams"]) == 18
    assert query_blocks[-1]["retrieved_training_excerpts"][-1].endswith("terminal-suffix")


def test_query_adapter_has_no_implicit_capacity_and_finite_text_guard_aborts():
    config = QueryMomentEvidenceAdapterConfig()
    config.validate()
    assert all(
        getattr(config, field) is None
        for field in (
            "max_queries",
            "max_terms_per_query",
            "max_chunks_per_query",
            "fallback_chunks_per_query",
            "max_excerpt_chars",
            "max_term_chars",
            "max_ngram_tokens",
        )
    )
    text = "marker " + ("z" * 40) + " suffix"
    assert _term_centered_excerpt(text, ("marker",), max_chars=None) == text
    with pytest.raises(ValueError, match="refusing silent note truncation"):
        _term_centered_excerpt(text, ("marker",), max_chars=12)


def test_legacy_clinical_examples_never_slice_selected_note_text():
    dataset = pd.DataFrame({"note": ["short note", "long " + ("q" * 30) + " suffix"]})

    assert _clinical_text_examples(
        dataset,
        "note",
        n_examples=None,
        max_chars=None,
    ) == dataset["note"].tolist()
    sampled = _clinical_text_examples(dataset, "note", n_examples=1, max_chars=None)
    assert len(sampled) == 1
    assert sampled[0] in dataset["note"].tolist()
    with pytest.raises(ValueError, match="refusing silent note truncation"):
        _clinical_text_examples(dataset, "note", n_examples=None, max_chars=12)


def test_attention_context_helpers_preserve_all_spans_rows_and_characters():
    long_text = "start " + ("a" * 700) + " terminal-suffix"
    spans = [
        {"text": f"span-{index:02d} " + ("b" * 150), "focus_token": f"focus-{index:02d}"}
        for index in range(7)
    ]

    compact = _compact_token_spans(spans)
    assert len(compact) == 7
    assert compact[-1]["text"].endswith("b" * 150)
    assert _attention_evidence_snippet(long_text, spans).endswith("terminal-suffix")
    selected = _tail_attention_positions(
        heldout_pos=np.asarray([0, 1, 2, 3]),
        labels=np.asarray([1.0, 1.0, 1.0, 1.0]),
        probs=np.asarray([0.1, 0.2, 0.3, 0.4]),
        max_rows=None,
    )
    assert selected.tolist() == [3, 2, 1, 0]
    with pytest.raises(ValueError, match="refusing silent attention-row omission"):
        _tail_attention_positions(
            heldout_pos=np.asarray([0, 1, 2, 3]),
            labels=np.asarray([1.0, 1.0, 1.0, 1.0]),
            probs=np.asarray([0.1, 0.2, 0.3, 0.4]),
            max_rows=3,
        )


def test_consistency_fallback_never_slices_eligible_candidates():
    first = AgenticFeatureProposal(action="add", name="first")
    second = AgenticFeatureProposal(action="add", name="second")
    summaries = [
        {
            "name": "first",
            "passes_consistency_gate": False,
            "proposed_on_full_outer_train": True,
            "inner_support_count": 0,
        },
        {
            "name": "second",
            "passes_consistency_gate": False,
            "proposed_on_full_outer_train": True,
            "inner_support_count": 0,
        },
    ]

    fallback = _fallback_consistency_proposals(
        summaries,
        {"first": first, "second": second},
    )
    assert fallback == [first, second]
    with pytest.raises(RuntimeError, match="refusing silent fallback-candidate omission"):
        _require_complete_consistency_fallback(fallback, max_selected=1)
    with pytest.raises(ValueError, match="refusing silent response truncation"):
        _agentic_consistency_selected_proposals(
            [
                {"action": "add", "name": "first"},
                {"action": "add", "name": "second"},
            ],
            candidate_summaries=summaries,
            canonical_proposals={"first": first, "second": second},
            max_selected=1,
        )
