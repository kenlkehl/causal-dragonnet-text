from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oci.inference.plain_handoff_stage2_evidence import (
    EVIDENCE_COMPILER_VERSION,
    compile_stage2_handoff_evidence,
)


def _htr_row(*, outer_fold: int, inner_fold: int, row_id: int) -> dict:
    return {
        "source": "text_models",
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "scope": "candidate_consistency_inner_train",
        "evidence": {
            "htr_evidence": {
                "effect": {
                    "attention": [
                        {
                            "row_id": row_id,
                            "chunk_index": 91,
                            "chunk_text": "Pretreatment ECOG performance status was 2.",
                            "highlighted_chunk_text": (
                                "Pretreatment **ECOG performance status** was 2."
                            ),
                            "attended_token_summary": "ECOG performance status; ECOG 2",
                            "attention": 0.8,
                            "stage": "effect_modifier",
                        }
                    ]
                }
            }
        },
    }


def test_compiler_is_fold_local_and_preserves_exact_deduplication_lineage(
    tmp_path: Path,
):
    compiled = compile_stage2_handoff_evidence(
        [
            _htr_row(outer_fold=1, inner_fold=1, row_id=3),
            _htr_row(outer_fold=1, inner_fold=2, row_id=3),
            _htr_row(outer_fold=2, inner_fold=1, row_id=3),
        ],
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=16,
        max_packet_chars=2_000,
    )

    fold_one = compiled.summary["outer_folds"]["1"]
    fold_two = compiled.summary["outer_folds"]["2"]
    assert fold_one["raw_occurrences"] == 2
    assert fold_one["exact_members"] == 1
    assert fold_one["exact_duplicate_occurrences_removed"] == 1
    assert fold_two["raw_occurrences"] == 1
    assert fold_two["exact_members"] == 1
    assert {packet["outer_fold"] for packet in compiled.packets} == {1, 2}

    member = compiled.members_by_outer_fold[1][0]
    assert member["raw_occurrence_count"] == 2
    assert {row["inner_fold"] for row in member["raw_references"]} == {1, 2}
    card = compiled.cards_by_outer_fold[1][0]
    assert card["support"]["inner_folds"] == [1, 2]
    assert card["support"]["raw_occurrence_count"] == 2
    assert compiled.lineage_by_outer_fold[1][0]["member_ids"] == [member["member_id"]]

    rendered = json.dumps(compiled.packets, sort_keys=True)
    assert '"row_id"' not in rendered
    assert '"handoff_row"' not in rendered
    assert (
        max(
            len(json.dumps(packet, separators=(",", ":"), sort_keys=True))
            for packet in compiled.packets
        )
        <= 2_000
    )


def test_compiler_reuses_fusion_tfidf_allowlist_and_drops_large_score_arrays(
    tmp_path: Path,
):
    sentinel = "SHOULD_NOT_CROSS_THE_FUSION_BOUNDARY"
    rows = [
        {
            "source": "tfidf",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {
                "discovery": {
                    "topic_banks": {
                        "treatment": {
                            "topics": [
                                {
                                    "topic_id": "topic-1",
                                    "terms": [
                                        {"term": "ECOG performance status", "loading": 0.9},
                                        {"term": "functional limitation", "loading": 0.7},
                                    ],
                                }
                            ],
                            "selected_terms": [sentinel] * 2_000,
                        },
                        "outcome": {"topics": []},
                        "effect": {"topics": []},
                    }
                }
            },
        }
    ]

    compiled = compile_stage2_handoff_evidence(
        rows,
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=16,
        max_packet_chars=2_000,
    )

    rendered = json.dumps(compiled.packets, sort_keys=True)
    assert "ECOG performance status" in rendered
    assert sentinel not in rendered
    assert compiled.packets[0]["content"]["source_families"] == ["tfidf_topics"]


def test_compiler_memory_maps_existing_stage1_embeddings_for_semantic_cards(
    tmp_path: Path,
):
    cache = tmp_path / "components" / "embedding_cache" / "cache" / "test-cache"
    cache.mkdir(parents=True)
    vectors = np.zeros((20, 4), dtype=np.float16)
    vectors[:10, 0] = 1.0
    vectors[10:, 1] = 1.0
    np.save(cache / "chunk_embeddings.npy", vectors)
    np.save(cache / "offsets.npy", np.asarray([0, 20], dtype=np.int64))
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "sentence_model_name": "cached-test-embedder",
                "normalize_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    chunks = [
        {
            "row_id": 0,
            "chunk_index": index,
            "text": f"Pretreatment clinical evidence chunk {index}",
            "similarity": 0.9 - 0.01 * index,
        }
        for index in range(20)
    ]
    rows = [
        {
            "source": "text_models",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {
                "embedding_contrast_evidence": {
                    "contrasts": [
                        {
                            "name": "treatment assignment contrast",
                            "positive_aligned_chunks": chunks,
                        }
                    ]
                }
            },
        }
    ]

    compiled = compile_stage2_handoff_evidence(
        rows,
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=16,
        max_packet_chars=2_000,
        seed=7,
    )

    assert compiled.summary["embedding_cache_model"] == "cached-test-embedder"
    assert compiled.summary["outer_folds"]["1"]["exact_members"] == 20
    # The cap is 16, but the cache contains only two distinct semantic vectors.
    assert compiled.summary["outer_folds"]["1"]["cards"] == 2
    assert all(
        card["semantic_grouping"] == "cached_embedding" for card in compiled.cards_by_outer_fold[1]
    )
    assert sum(len(lineage["member_ids"]) for lineage in compiled.lineage_by_outer_fold[1]) == 20
    assert compiled.summary["schema_version"] == EVIDENCE_COMPILER_VERSION
