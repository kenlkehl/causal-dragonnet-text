from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import AppliedInferenceConfig
from oci.models.concept_embedding_utils import chunk_text_words
from oci.inference.all_evidence_fusion import (
    ALL_SOURCE_FAMILIES,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    LEGACY_ALL_SOURCE,
    TFIDF_TOPIC_SOURCE,
    prepare_all_evidence_fusion,
)
from oci.inference.fold_honest_r_stack import FitRowProvenance
from oci.inference.review_spent_evidence_provider import (
    BoundSpentFrozenChunkEmbeddingProvider,
    ContextFitReviewSpentEvidenceProvider,
    SpentOnlyFrozenChunkEmbeddingCache,
    SpentDiscoveryEvidence,
    TfidfTopicOrphanSpentDiscoveryBackend,
    _embedding_concepts_only,
    _htr_attention_group_key,
    _htr_attention_source_key,
    _htr_concepts_only,
    _htr_phrase_has_unsafe_numeric_fragment,
    _safe_concept_phrase,
    _sanitize_digest_terms,
)
from oci.inference.multi_model_agentic_forest import _build_role_grouped_evidence_digest
import oci.inference.review_spent_evidence_provider as spent_module
import oci.inference.tfidf_upstream_gate_backend as tfidf_backend_module


def test_temporal_wording_is_not_filtered_from_spent_concepts():
    assert _safe_concept_phrase("sensor output after recalibration") == (
        "sensor output after recalibration"
    )
    assert _safe_concept_phrase("account number 12345678") == ""


def test_stage1_spent_identity_rejects_live_parallelism_mutation() -> None:
    backend = spent_module.HistoricalStage1SpentDiscoveryBackend.__new__(
        spent_module.HistoricalStage1SpentDiscoveryBackend
    )
    backend._stage1_config_snapshot = SimpleNamespace(verify_source=lambda: None)
    backend._htr_model_snapshot = SimpleNamespace(verify=lambda: None)
    backend.config = AppliedInferenceConfig()
    forest = backend.config.architecture.multi_model_forest
    forest.fold_parallelism = "1"
    forest.bow_fold_parallelism = "3"
    forest.bow_parallel_backend = "threads"
    forest.htr_fold_parallelism = "1"
    forest.cpus_total = 3
    cache_identity = {"provider": "synthetic_cache"}
    backend.embedding_cache = SimpleNamespace(identity=lambda: copy.deepcopy(cache_identity))
    backend._identity = {
        "effective_config_sha256": spent_module._effective_applied_config_sha256(
            backend.config
        ),
        "embedding_cache": cache_identity,
    }

    assert backend.identity()["effective_config_sha256"] == (
        backend._identity["effective_config_sha256"]
    )
    assert forest.htr_fold_parallelism == "1"
    forest.bow_fold_parallelism = "4"
    with pytest.raises(RuntimeError, match="effective spent Stage-1 runtime config changed"):
        backend.identity()


def _write_lazy_embedding_cache(path: Path, texts: tuple[str, ...]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [tuple(chunk_text_words(text, 3, 1, 8, "last")) for text in texts]
    offsets = np.asarray([0, *np.cumsum([len(row) for row in rows]).tolist()], dtype=np.int64)
    embeddings = np.arange(int(offsets[-1]) * 4, dtype=np.float16).reshape(int(offsets[-1]), 4)
    np.save(path / "chunk_embeddings.npy", embeddings)
    np.save(path / "offsets.npy", offsets)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "num_samples": len(texts),
                "hidden_size": 4,
                "chunk_size_words": 3,
                "chunk_overlap_words": 1,
                "max_chunks": 8,
                "chunk_selection": "last",
            }
        ),
        encoding="utf-8",
    )
    with (path / "chunk_texts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"chunks": list(row)}) + "\n")


def test_lazy_embedding_cache_decodes_only_bound_rows_and_refuses_sealed_rows(
    tmp_path: Path, monkeypatch
) -> None:
    texts = (
        "one two three four five",
        "sealed future text must stay opaque",
        "six seven eight nine ten",
    )
    _write_lazy_embedding_cache(tmp_path, texts)
    real_loads = spent_module.json.loads
    decoded_chunk_rows: list[dict] = []

    def tracking_loads(value, *args, **kwargs):
        parsed = real_loads(value, *args, **kwargs)
        if isinstance(parsed, dict) and "chunks" in parsed:
            decoded_chunk_rows.append(parsed)
        return parsed

    monkeypatch.setattr(spent_module.json, "loads", tracking_loads)
    cache = SpentOnlyFrozenChunkEmbeddingCache(tmp_path)
    assert decoded_chunk_rows == []
    bound = cache.bind_spent((0, 2), (texts[0], texts[2]))
    assert isinstance(bound, BoundSpentFrozenChunkEmbeddingProvider)
    assert len(decoded_chunk_rows) == 2
    assert bound.chunk_texts((0, 2))[0][0] == "one two three"
    assert [matrix.shape[1] for matrix in bound.chunk_matrices((0, 2))] == [4, 4]
    with pytest.raises(ValueError, match="non-spent"):
        bound.chunk_texts((1,))
    with pytest.raises(ValueError, match="non-spent"):
        bound.chunk_matrices((1,))
    assert all("sealed future" not in json.dumps(row) for row in decoded_chunk_rows)


def test_lazy_embedding_cache_rejects_mismatch_duplicate_range_and_tamper(
    tmp_path: Path,
) -> None:
    texts = ("one two three four", "five six seven eight")
    _write_lazy_embedding_cache(tmp_path, texts)
    cache = SpentOnlyFrozenChunkEmbeddingCache(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        cache.bind_spent((0,), ("changed spent text",))
    with pytest.raises(ValueError, match="unique"):
        cache.bind_spent((0, 0), (texts[0], texts[0]))
    with pytest.raises(ValueError, match="outside"):
        cache.bind_spent((9,), ("outside",))

    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tampered"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="bytes changed"):
        cache.identity()


def test_lazy_embedding_cache_rejects_path_swap_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ("one two three four", "five six seven eight")
    _write_lazy_embedding_cache(tmp_path, texts)
    target = tmp_path / "chunk_texts.jsonl"
    real_stat_signature = spent_module._stat_signature
    swapped = False

    def swapping_stat_signature(path: Path):
        nonlocal swapped
        signature = real_stat_signature(path)
        if path == target and not swapped:
            target.write_text('{"chunks":["replacement"]}\n', encoding="utf-8")
            swapped = True
        return signature

    monkeypatch.setattr(spent_module, "_stat_signature", swapping_stat_signature)
    with pytest.raises(RuntimeError, match="changed while it was being authenticated"):
        SpentOnlyFrozenChunkEmbeddingCache(tmp_path)
    assert swapped is True


@pytest.mark.parametrize(
    "filename",
    ["metadata.json", "chunk_embeddings.npy", "offsets.npy", "chunk_texts.jsonl"],
)
def test_lazy_embedding_cache_detaches_every_authenticated_representation(
    tmp_path: Path,
    filename: str,
) -> None:
    texts = ("one two three four", "five six seven eight")
    _write_lazy_embedding_cache(tmp_path, texts)
    cache = SpentOnlyFrozenChunkEmbeddingCache(tmp_path)
    identity = cache.identity()
    digest_fields = {
        "metadata.json": "metadata_sha256",
        "chunk_embeddings.npy": "embeddings_sha256",
        "offsets.npy": "offsets_sha256",
        "chunk_texts.jsonl": "chunk_texts_sha256",
    }
    assert (
        identity[digest_fields[filename]]
        == hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
    )
    assert identity["provider"] == "spent_only_frozen_chunk_embedding_cache_v2"
    assert identity["chunk_text_storage"] == "private_fd_pread_lazy_row_decode_v1"
    assert identity["embeddings_path_backed"] is False
    assert identity["private_snapshot_embedding_mmap"] is True
    assert isinstance(cache._embeddings, np.memmap)
    assert cache._embeddings.flags.writeable is False
    assert cache._offsets.flags.writeable is False
    assert not hasattr(cache, "_chunk_text_bytes")

    before = cache.bind_spent((0, 1), texts)
    before_chunks = before.chunk_texts((0, 1))
    before_matrices = before.chunk_matrices((0, 1))
    before_metadata = dict(cache.metadata)
    target = tmp_path / filename
    if filename == "metadata.json":
        target.write_text(json.dumps({"replaced": True}), encoding="utf-8")
    elif filename == "chunk_embeddings.npy":
        replacement = np.full(cache._embeddings.shape, 99.0, dtype=np.float16)
        np.save(target, replacement)
    elif filename == "offsets.npy":
        np.save(
            target,
            np.asarray([0, 1, int(cache._embeddings.shape[0])], dtype=np.int64),
        )
    else:
        target.write_text('{"chunks":["replacement"]}\n', encoding="utf-8")

    rebound = cache.bind_spent((0, 1), texts)
    assert rebound.chunk_texts((0, 1)) == before_chunks
    for actual, expected in zip(rebound.chunk_matrices((0, 1)), before_matrices):
        np.testing.assert_array_equal(actual, expected)
    assert dict(cache.metadata) == before_metadata
    with pytest.raises(RuntimeError, match="bytes changed or path changed"):
        cache.identity()


def test_lazy_embedding_cache_metadata_and_outputs_cannot_mutate_snapshot(
    tmp_path: Path,
) -> None:
    texts = ("one two three four", "five six seven eight")
    _write_lazy_embedding_cache(tmp_path, texts)
    cache = SpentOnlyFrozenChunkEmbeddingCache(tmp_path)
    metadata = cache.metadata
    metadata["chunk_size_words"] = 999
    bound = cache.bind_spent((0,), (texts[0],))
    output = bound.chunk_matrix(0)
    output[:] = -1
    assert cache.metadata["chunk_size_words"] == 3
    assert np.all(bound.chunk_matrix(0) >= 0)


def _legacy_payload(outer_fold: int, review_round: int) -> dict:
    return {
        "outer_fold": outer_fold,
        "scope": "inner_train",
        "inner_fold": review_round + 1,
        "context": {
            "evidence_digest": {
                "confounders": {
                    "bow_blurbs": [
                        {
                            "source": "nuisance.confounder_overlap",
                            "rows": [{"feature": "baseline wheel alignment", "score": 2.0}],
                        }
                    ],
                    "embedding_chunks": [
                        {
                            "name": "whole cohort treatment contrast",
                            "contrast_family": "whole cohort",
                            "concept_probe_scores": [
                                {"concept": "baseline actuator status", "score": 0.7}
                            ],
                        }
                    ],
                    "htr_blurbs": [
                        {
                            "stage": "nuisance",
                            "rows": [{"top_token_spans": [{"token": "baseline vibration burden"}]}],
                        }
                    ],
                },
                "effect_modifiers": {
                    "bow_blurbs": [
                        {
                            "source": "ensemble_r.pseudo_target_positive",
                            "rows": [{"feature": "baseline coating status", "score": 3.0}],
                        },
                        {
                            "source": "matched_pair_uplift.uplift_pair_features",
                            "rows": [{"feature": "prior load burden", "score": 1.5}],
                        },
                    ],
                    "embedding_chunks": [
                        {
                            "name": "effect residual cluster component",
                            "contrast_family": "cluster component",
                            "concept_probe_scores": [
                                {"concept": "baseline calibration pattern", "score": -0.6}
                            ],
                        }
                    ],
                    "htr_blurbs": [
                        {
                            "stage": "effect",
                            "rows": [{"top_token_spans": [{"token": "baseline material result"}]}],
                        }
                    ],
                },
            }
        },
    }


def _tfidf_payload(outer_fold: int, review_round: int) -> dict:
    topic = lambda phrase: {"terms": [{"term": phrase, "loading": 0.8}]}  # noqa: E731
    return {
        "outer_fold": outer_fold,
        "scope": "inner_train",
        "inner_fold": review_round + 1,
        "discovery": {
            "topic_banks": {
                "treatment": {"topics": [topic("baseline routing pattern")]},
                "outcome": {"topics": [topic("baseline failure pattern")]},
                "effect": {"topics": [topic("baseline sensor phrase")]},
            },
            "effect_orphan_ngram_branch": {
                "selected_cluster_ids": ["cluster_001"],
                "selected_clusters": [
                    {
                        "cluster_id": "cluster_001",
                        "terms": [{"term": "unmodeled baseline phrase", "fit_rank": 2}],
                    }
                ],
            },
        },
    }


class _Backend:
    def __init__(self, source_kind: str):
        self.source_kind = source_kind
        self.calls = 0

    def identity(self):
        return {"backend": f"test_{self.source_kind}_v1"}

    def fit_discovery(
        self,
        *,
        outer_fold,
        review_round,
        exact_spent_row_ids,
        spent_texts,
        spent_treatment,
        spent_outcome,
        work_dir,
    ):
        del spent_texts, spent_treatment, spent_outcome, work_dir
        self.calls += 1
        payload = (
            _legacy_payload(outer_fold, review_round)
            if self.source_kind == LEGACY_ALL_SOURCE
            else _tfidf_payload(outer_fold, review_round)
        )
        return SpentDiscoveryEvidence.create(
            source_kind=self.source_kind,
            payload=payload,
            fit_row_provenance=FitRowProvenance(fit_row_ids=frozenset(exact_spent_row_ids)),
        )


def _call_from_frame(provider, frame: pd.DataFrame):
    spent = frame.loc[frame["partition"] == "spent"]
    sealed = frame.loc[frame["partition"] == "sealed"]
    return provider.get_spent_evidence_inputs(
        outer_fold=2,
        review_round=0,
        exact_spent_row_ids=tuple(spent["row_id"]),
        exact_sealed_row_ids=tuple(sealed["row_id"]),
        spent_texts=tuple(spent["text"]),
        spent_treatment=spent["treatment"].to_numpy(dtype=float),
        spent_outcome=spent["outcome"].to_numpy(dtype=float),
    )


def test_provider_is_invariant_to_every_future_gate_value_and_emits_exact_provenance(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "row_id": [10, 11, 12, 20, 21],
            "partition": ["spent", "spent", "spent", "sealed", "sealed"],
            "text": ["spent a", "spent b", "spent c", "sealed a", "sealed b"],
            "treatment": [0, 1, 0, 0, 1],
            "outcome": [0, 0, 1, 1, 0],
        }
    )
    legacy = _Backend(LEGACY_ALL_SOURCE)
    tfidf = _Backend(TFIDF_TOPIC_SOURCE)
    provider = ContextFitReviewSpentEvidenceProvider(
        backends=(legacy, tfidf),
        cache_dir=tmp_path / "cache",
    )
    first = _call_from_frame(provider, frame)
    changed = frame.copy()
    sealed = changed["partition"] == "sealed"
    changed.loc[sealed, "text"] = "future gate changed completely"
    changed.loc[sealed, "treatment"] = 1 - changed.loc[sealed, "treatment"]
    changed.loc[sealed, "outcome"] = 1 - changed.loc[sealed, "outcome"]
    second = _call_from_frame(provider, changed)

    assert [item.payload for item in first] == [item.payload for item in second]
    assert legacy.calls == tfidf.calls == 1
    assert all(isinstance(item, FoldEvidenceInput) for item in first)
    for item in first:
        assert item.provenance.train_row_ids == (10, 11, 12)
        assert item.provenance.heldout_row_ids == (20, 21)
        assert item.provenance.scope == "inner_train"
        assert item.provenance.inner_fold == 1

    request = prepare_all_evidence_fusion(first)
    assert set(request.source_family_coverage["present_source_families"]) == set(
        ALL_SOURCE_FAMILIES
    ) - {"neural_query_moments", "sparse_query_moments"}
    reviewer_json = json.dumps(request.context())
    assert "future gate changed" not in reviewer_json
    assert '"row_id"' not in reviewer_json
    assert "sealed a" not in reviewer_json


class _LeakyLineageBackend(_Backend):
    def fit_discovery(self, **kwargs):
        result = super().fit_discovery(**kwargs)
        return SpentDiscoveryEvidence.create(
            source_kind=result.source_kind,
            payload=result.payload,
            fit_row_provenance=FitRowProvenance(
                fit_row_ids=frozenset((*kwargs["exact_spent_row_ids"], 999))
            ),
        )


def test_provider_rejects_future_row_in_recursive_fit_lineage(tmp_path: Path) -> None:
    provider = ContextFitReviewSpentEvidenceProvider(
        backends=(_LeakyLineageBackend(TFIDF_TOPIC_SOURCE),),
        cache_dir=tmp_path,
        required_source_families=(),
    )
    with pytest.raises(ValueError, match="exact-spent FitRowProvenance"):
        provider.get_spent_evidence_inputs(
            outer_fold=1,
            review_round=0,
            exact_spent_row_ids=(1, 2, 3),
            exact_sealed_row_ids=(4,),
            spent_texts=("a", "b", "c"),
            spent_treatment=np.asarray([0.0, 1.0, 0.0]),
            spent_outcome=np.asarray([0.0, 0.0, 1.0]),
        )


def test_payload_boundary_rejects_rows_and_excerpts() -> None:
    lineage = FitRowProvenance(fit_row_ids=frozenset({1, 2}))
    for unsafe in (
        {"row_id": 1, "term": "baseline sensor"},
        {"text": "raw source sentence"},
        {"retrieved_chunks": ["raw source sentence"]},
        {"term": "account number 12345678"},
    ):
        with pytest.raises(ValueError):
            SpentDiscoveryEvidence.create(
                source_kind=TFIDF_TOPIC_SOURCE,
                payload=unsafe,
                fit_row_provenance=lineage,
            )


def test_stage1_semantic_projection_discards_raw_embedding_and_htr_excerpts() -> None:
    embedding = _embedding_concepts_only(
        {
            "contrasts": [
                {
                    "name": "effect cluster component",
                    "role_hint": "effect_modifier",
                    "contrast_family": "cluster component",
                    "positive_aligned_chunks": [
                        {"row_id": 5, "text": "baseline coating positive pattern"}
                    ],
                    "negative_aligned_chunks": [
                        {"row_id": 8, "text": "baseline coating absent pattern"}
                    ],
                }
            ]
        }
    )
    htr = _htr_concepts_only(
        {
            "effect": {
                "attention": [
                    {
                        "row_id": 5,
                        "evidence_snippet": "this full note sentence must disappear",
                        "top_token_spans": [
                            {"text": "baseline material code"},
                        ],
                    }
                ]
            }
        }
    )
    serialized = json.dumps({"embedding": embedding, "htr": htr})
    assert "row_id" not in serialized
    assert "full note sentence" not in serialized
    assert "baseline material code" in serialized
    assert "baseline coating positive pattern" not in serialized
    assert embedding["contrasts"][0]["concept_probe_scores"]

    digest = _sanitize_digest_terms(
        _build_role_grouped_evidence_digest(
            importance={}, embedding_evidence=embedding, htr_evidence=htr
        )
    )
    projected = SpentDiscoveryEvidence.create(
        source_kind=LEGACY_ALL_SOURCE,
        payload={
            "outer_fold": 1,
            "scope": "inner_train",
            "inner_fold": 1,
            "context": {"evidence_digest": digest},
        },
        fit_row_provenance=FitRowProvenance(fit_row_ids=frozenset({1, 2})),
    )
    compacted = prepare_all_evidence_fusion(
        [
            FoldEvidenceInput(
                LEGACY_ALL_SOURCE,
                projected.payload,
                FoldEvidenceProvenance(
                    outer_fold=1,
                    train_row_ids=(1, 2),
                    heldout_row_ids=(3,),
                    scope="inner_train",
                    inner_fold=1,
                    artifact_id="stage1-projection-test",
                ),
            )
        ]
    )
    assert {"htr_neural", "embedding_clustered"} <= set(
        compacted.source_family_coverage["present_source_families"]
    )


def test_chunk_only_htr_projection_is_contrastive_deterministic_and_excerpt_free() -> None:
    high_texts = (
        "baseline hydraulic imbalance pressure elevated before system assignment",
        "baseline hydraulic imbalance with pressure elevation at initial assessment",
        "baseline hydraulic imbalance and pressure elevation during eligibility review",
    )
    low_texts = (
        "routine scheduling paperwork reviewed before facility visit",
        "routine scheduling forms prepared for facility intake",
        "routine scheduling reminder entered for office appointment",
    )
    attention = [
        row
        for row_id, high_text, low_text in zip((101, 102, 103), high_texts, low_texts)
        for row in (
            {
                "row_id": row_id,
                "chunk_index": 0,
                "chunk_text": high_text,
                "attention": 0.9,
            },
            {
                "row_id": row_id,
                "chunk_index": 1,
                "chunk_text": low_text,
                "attention": 0.1,
            },
        )
    ]

    projected = _htr_concepts_only({"effect": {"attention": attention}})
    permuted = _htr_concepts_only({"effect": {"attention": list(reversed(attention))}})

    assert projected == permuted
    concepts = projected["effect"]["attention"]
    assert concepts
    assert len(concepts) <= 6
    assert all(1 <= len(row["attended_token_summary"].split()) <= 3 for row in concepts)
    concept_tokens = [token for row in concepts for token in row["attended_token_summary"].split()]
    assert len(concept_tokens) == len(set(concept_tokens))
    serialized = json.dumps(projected, sort_keys=True)
    assert all(text not in serialized for text in (*high_texts, *low_texts))
    assert "row_id" not in serialized
    assert "chunk_text" not in serialized

    digest = _sanitize_digest_terms(
        _build_role_grouped_evidence_digest(
            importance={}, embedding_evidence={}, htr_evidence=projected
        )
    )
    evidence = SpentDiscoveryEvidence.create(
        source_kind=LEGACY_ALL_SOURCE,
        payload={
            "outer_fold": 1,
            "scope": "inner_train",
            "inner_fold": 1,
            "context": {"evidence_digest": digest},
        },
        fit_row_provenance=FitRowProvenance(fit_row_ids=frozenset({1, 2, 3})),
    )
    compacted = prepare_all_evidence_fusion(
        [
            FoldEvidenceInput(
                LEGACY_ALL_SOURCE,
                evidence.payload,
                FoldEvidenceProvenance(
                    outer_fold=1,
                    train_row_ids=(1, 2, 3),
                    heldout_row_ids=(4,),
                    scope="inner_train",
                    inner_fold=1,
                    artifact_id="chunk-only-htr-projection-test",
                ),
            )
        ]
    )
    assert "htr_neural" in compacted.source_family_coverage["present_source_families"]
    assert all(text not in json.dumps(compacted.context(), sort_keys=True) for text in high_texts)


def test_chunk_only_htr_projection_separates_model_folds_and_pair_sides() -> None:
    fold_one = {
        "outer_fold": 3,
        "fold": 1,
        "row_id": 7,
        "pair_side": "candidate",
    }
    fold_two = {**fold_one, "fold": 2}
    assert _htr_attention_source_key(fold_one) == _htr_attention_source_key(fold_two)
    assert _htr_attention_group_key(fold_one) != _htr_attention_group_key(fold_two)

    no_row_candidate = {
        "outer_fold": 3,
        "fold": 1,
        "pair_side": "candidate",
        "candidate_row_id": 11,
        "control_row_id": 12,
    }
    no_row_control = {**no_row_candidate, "pair_side": "matched_control"}
    assert _htr_attention_source_key(no_row_candidate) == ("11", "candidate")
    assert _htr_attention_source_key(no_row_control) == ("12", "matched_control")


def test_chunk_only_htr_projection_rejects_name_and_numeric_fragments() -> None:
    attention = [
        row
        for row_id, suffix in ((1, "alpha"), (2, "beta"))
        for row in (
            {
                "row_id": row_id,
                "chunk_index": 0,
                "chunk_text": (
                    f"John Smith SMITH JONES baseline hydraulic imbalance level 4 "
                    f"AX4 Q7 Z900E {suffix}"
                ),
                "attention": 0.9,
            },
            {
                "row_id": row_id,
                "chunk_index": 1,
                "chunk_text": f"routine scheduling office paperwork cohort {suffix}",
                "attention": 0.1,
            },
        )
    ]

    projected = _htr_concepts_only({"effect": {"attention": attention}})
    serialized = json.dumps(projected, sort_keys=True)
    assert projected
    assert "john" not in serialized
    assert "smith" not in serialized
    assert "jones" not in serialized
    phrases = [row["attended_token_summary"] for row in projected["effect"]["attention"]]
    assert all("4" not in phrase.split() for phrase in phrases)
    assert any(token in {"ax4", "q7", "z900e"} for phrase in phrases for token in phrase.split())
    assert _htr_phrase_has_unsafe_numeric_fragment("batch 78")
    assert _htr_phrase_has_unsafe_numeric_fragment("record abc123456")
    assert not _htr_phrase_has_unsafe_numeric_fragment("ax4 q7 z900e lm n2")

    contextual_lowercase_name = [
        row
        for row_id, suffix in ((1, "alpha"), (2, "beta"))
        for row in (
            {
                "row_id": row_id,
                "chunk_index": 0,
                "chunk_text": f"named maria santos baseline hydraulic sensor {suffix}",
                "attention": 0.9,
            },
            {
                "row_id": row_id,
                "chunk_index": 1,
                "chunk_text": f"routine scheduling paperwork {suffix}",
                "attention": 0.1,
            },
        )
    ]
    assert _htr_concepts_only({"effect": {"attention": contextual_lowercase_name}}) == {}


@pytest.mark.parametrize(
    "attention",
    [
        [],
        [
            {
                "row_id": 1,
                "chunk_index": 0,
                "chunk_text": "baseline hydraulic sensor",
                "attention": 0.9,
            },
            {"row_id": 1, "chunk_index": 1, "chunk_text": "routine scheduling", "attention": 0.1},
        ],
        [
            {"row_id": row_id, "chunk_index": index, "chunk_text": text, "attention": 0.0}
            for row_id in (1, 2)
            for index, text in enumerate(("baseline hydraulic sensor", "routine scheduling"))
        ],
        [
            {"row_id": row_id, "chunk_index": index, "chunk_text": text, "attention": np.nan}
            for row_id in (1, 2)
            for index, text in enumerate(("baseline hydraulic sensor", "routine scheduling"))
        ],
        [
            {
                "row_id": row_id,
                "chunk_index": index,
                "chunk_text": text,
                "attention": score,
            }
            for row_id in (1, 2)
            for index, (text, score) in enumerate(
                (("account number 12345678 hydraulic sensor", 0.9), ("routine scheduling", 0.1))
            )
        ],
        [
            {
                "row_id": row_id,
                "chunk_index": index,
                "chunk": text,
                "attention": score,
            }
            for row_id in (1, 2)
            for index, (text, score) in enumerate(
                (("baseline hydraulic sensor", 0.9), ("routine scheduling", 0.1))
            )
        ],
    ],
    ids=("empty", "one-row", "zero", "nonfinite", "unsafe", "unknown-alias"),
)
def test_chunk_only_htr_projection_fails_closed_without_usable_contrast(
    attention: list[dict],
) -> None:
    assert _htr_concepts_only({"effect": {"attention": attention}}) == {}


def test_tfidf_backend_refits_only_spent_rows_and_builds_safe_orphans(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "stage1.json"
    config_path.write_text("{}", encoding="utf-8")
    config = AppliedInferenceConfig()
    forest = config.architecture.multi_model_forest
    forest.bow_views = forest.bow_views[:1]
    forest.tfidf_topic.topic_count = 1
    snapshot = SimpleNamespace(
        source_path=config_path.resolve(),
        sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        applied_config=lambda: copy.deepcopy(config),
        verify_source=lambda: None,
    )
    monkeypatch.setattr(
        tfidf_backend_module,
        "_historical_stage1_config_snapshot",
        lambda _path, _snapshot=None: snapshot,
    )

    observed = {}

    def fake_fit(**kwargs):
        observed["fit"] = kwargs["fit_df"].copy()
        observed["heldout"] = kwargs["heldout_df"].copy()
        assert kwargs["enable_heldout_score_tests"] is False
        output = Path(kwargs["artifact_dir"])
        output.mkdir(parents=True, exist_ok=True)
        scores_path = output / "effect_ngram_scores.parquet"
        pd.DataFrame(
            {
                "feature": ["baseline uncommon sensor", "represented sensor"],
                "eligible": [True, True],
                "fit_signed_score": [3.0, 2.0],
                "combined_importance": [3.0, 2.0],
                "support_control": [3, 3],
                "support_treated": [3, 3],
            }
        ).to_parquet(scores_path, index=False)
        return {
            "topic_banks": {
                "treatment": {"topics": [{"terms": [{"term": "baseline routing", "loading": 0.8}]}]},
                "outcome": {"topics": [{"terms": [{"term": "baseline failure", "loading": 0.7}]}]},
                "effect": {"topics": [{"terms": [{"term": "represented sensor", "loading": 0.6}]}]},
            },
            "artifacts": {"ngram_scores": {"effect": str(scores_path)}},
        }

    monkeypatch.setattr(
        "oci.inference.review_spent_evidence_provider.fit_tfidf_topic_context",
        fake_fit,
    )
    backend = TfidfTopicOrphanSpentDiscoveryBackend(stage1_config_path=config_path)
    result = backend.fit_discovery(
        outer_fold=3,
        review_round=1,
        exact_spent_row_ids=(2, 4, 6, 8),
        spent_texts=("alpha", "beta", "gamma", "delta"),
        spent_treatment=np.asarray([0.0, 1.0, 0.0, 1.0]),
        spent_outcome=np.asarray([0.0, 0.0, 1.0, 1.0]),
        work_dir=tmp_path / "work",
    )

    assert tuple(observed["fit"]["_oci_row_id"]) == (2, 4, 6, 8)
    assert list(observed["heldout"].columns) == ["_oci_row_id", "clinical_text"]
    assert result.fit_row_provenance.recursive_fit_row_ids() == frozenset({2, 4, 6, 8})
    payload = result.payload
    assert payload["scope"] == "inner_train"
    assert payload["inner_fold"] == 2
    clusters = payload["discovery"]["effect_orphan_ngram_branch"]["selected_clusters"]
    assert clusters[0]["terms"][0]["term"] == "baseline uncommon sensor"
    assert "represented sensor" not in json.dumps(clusters)
    assert "artifacts" not in json.dumps(payload)
