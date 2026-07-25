from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from oci.inference.all_evidence_fusion import (
    FoldEvidenceProvenance,
    NEURAL_QUERY_MOMENTS,
    QUERY_MOMENTS,
    SPARSE_QUERY_MOMENTS,
    SPARSE_QUERY_SOURCE,
    prepare_all_evidence_fusion,
)
from oci.inference.query_moment_evidence_adapter import (
    QueryMomentEvidenceAdapterConfig,
    adapt_query_moment_evidence,
    derive_sparse_query_moment_evidence,
    load_query_moment_evidence_artifact,
    reseal_legacy_neural_query_moment_evidence,
)
from oci.inference.tfidf_topic_discovery import row_set_fingerprint


def _provenance() -> FoldEvidenceProvenance:
    return FoldEvidenceProvenance(
        outer_fold=2,
        train_row_ids=(10, 11, 12, 13, 14, 15),
        heldout_row_ids=(20, 21),
        artifact_id="query-moment-fold-2",
    )


def _artifact_query(*, row_id: int = 10) -> list[dict]:
    return [
        {
            "query_id": "effect_query_001",
            "bank": "effect",
            "mechanical_role": "effect_modifier",
            "statistical_gate_applied": False,
            "member_count": 4,
            "member_subfolds": [1, 2],
            "fit_standardized_score": 2.25,
            "top_chunks": [
                {
                    "evidence_id": f"effect_query_001__row_{row_id:05d}__chunk_003",
                    "_oci_row_id": row_id,
                    "chunk_index": 3,
                    "similarity": 0.82,
                    "text": "amber lattice was documented before selection",
                }
            ],
            "top_contrastive_ngrams": [{"term": "amber lattice", "tfidf_contrast": 0.31}],
        }
    ]


def _tfidf_evidence() -> dict:
    def topic(topic_id: str, *terms: str) -> dict:
        return {
            "topic_id": topic_id,
            "terms": [
                {"term": term, "loading": 1.0 / (index + 1)} for index, term in enumerate(terms)
            ],
        }

    return {
        "outer_fold": 2,
        "discovery": {
            "topic_banks": {
                "treatment": {"topics": [topic("topic-a", "amber lattice", "quiet orchard")]},
                "outcome": {"topics": [topic("topic-b", "silver compass")]},
                "effect": {"topics": [topic("topic-c", "violet bridge")]},
            },
            "topic_score_tests": {
                "effect_orphan_ngram_branch": {
                    "selected_cluster_ids": ["cluster-a"],
                    "selected_clusters": [
                        {
                            "cluster_id": "cluster-a",
                            "terms": [{"term": "copper meadow", "fit_rank": 9}],
                        }
                    ],
                }
            },
        },
    }


def _fallback_inputs() -> dict:
    return {
        "outer_train_row_ids": [10, 11, 12, 13, 14, 15],
        "outer_train_texts": [
            "amber lattice beside a silver compass",
            "quiet orchard and violet bridge",
            "silver compass in a copper meadow",
            "amber lattice near a violet bridge",
            "copper meadow beyond the quiet orchard",
            "silver compass under the violet bridge",
        ],
        "treatment": [0, 1, 0, 1, 0, 1],
        "outcome": [0.2, 1.1, -0.3, 0.7, 0.1, 1.4],
        "tfidf_topic_evidence": _tfidf_evidence(),
    }


def test_loads_bare_artifact_with_exact_row_registration_and_fusion_contract(tmp_path):
    path = tmp_path / "query_evidence.json"
    path.write_text(json.dumps(_artifact_query()), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    result = load_query_moment_evidence_artifact(
        path,
        provenance=_provenance(),
        expected_sha256=digest,
        registered_fit_row_ids=[15, 14, 13, 12, 11, 10],
        registered_heldout_row_ids=[21, 20],
    )

    assert result.audit["mode"] == "authenticated_neural_query_artifact"
    assert result.audit["source_family"] == NEURAL_QUERY_MOMENTS
    assert result.audit["retrieved_rows_are_outer_train_only"] is True
    assert result.payload["scope"] == "outer_train"
    request = prepare_all_evidence_fusion([result.as_fold_evidence_input()])
    assert request.source_family_coverage["present_source_families"] == [QUERY_MOMENTS]
    assert "_oci_row_id" not in request.render_prompt()


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (_artifact_query(row_id=20), "outer-heldout"),
        (
            [dict(_artifact_query()[0], oracle_effect=1.0)],
            "forbidden target/oracle field",
        ),
    ],
)
def test_artifact_loader_rejects_heldout_rows_and_forbidden_fields(tmp_path, body, error):
    path = tmp_path / "bad_query_evidence.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_query_moment_evidence_artifact(path, provenance=_provenance())


def test_wrapped_artifact_partition_must_match_provenance(tmp_path):
    path = tmp_path / "wrapped_query_evidence.json"
    path.write_text(
        json.dumps(
            {
                "outer_fold": 2,
                "scope": "outer_train",
                "fit_row_ids": [10, 11, 12, 13, 14, 20],
                "heldout_row_ids": [15, 21],
                "query_evidence": _artifact_query(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match fold provenance"):
        load_query_moment_evidence_artifact(path, provenance=_provenance())


def test_artifact_loader_discards_provenance_checked_empty_legacy_chunk(tmp_path):
    body = _artifact_query()
    body[0]["top_chunks"].append(
        {
            "evidence_id": "effect_query_001__row_00011__chunk_099",
            "_oci_row_id": 11,
            "chunk_index": 99,
            "similarity": 0.1,
            "text": "",
        }
    )
    path = tmp_path / "legacy_empty_chunk.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    result = load_query_moment_evidence_artifact(path, provenance=_provenance())

    assert len(result.payload["query_evidence"][0]["top_chunks"]) == 1


def test_sparse_fallback_is_deterministic_bounded_and_fold_honest():
    first = derive_sparse_query_moment_evidence(
        provenance=_provenance(),
        **_fallback_inputs(),
    )
    second = derive_sparse_query_moment_evidence(
        provenance=_provenance(),
        **_fallback_inputs(),
    )

    assert first.payload == second.payload
    assert first.audit == second.audit
    queries = first.payload["query_evidence"]
    assert [query["bank"] for query in queries] == [
        "treatment",
        "outcome",
        "effect",
        "effect",
    ]
    assert all(np.isfinite(query["fit_standardized_score"]) for query in queries)
    assert all(len(query["top_chunks"]) <= 8 for query in queries)
    cited_rows = {chunk["_oci_row_id"] for query in queries for chunk in query["top_chunks"]}
    assert cited_rows <= set(_provenance().train_row_ids)
    assert not cited_rows & set(_provenance().heldout_row_ids)
    terms = {term["term"] for query in queries for term in query["top_contrastive_ngrams"]}
    assert terms == {
        "amber lattice",
        "quiet orchard",
        "silver compass",
        "violet bridge",
        "copper meadow",
    }
    assert first.audit["moment_rows_are_outer_train_only"] is True
    assert first.audit["heldout_text_or_labels_accessed"] is False
    request = prepare_all_evidence_fusion([first.as_fold_evidence_input()])
    assert (
        request.source_family_coverage["evidence_block_count_by_source_family"][
            SPARSE_QUERY_MOMENTS
        ]
        == 4
    )
    assert QUERY_MOMENTS not in request.source_family_coverage["present_source_families"]


def test_sparse_fallback_rejects_row_input_containing_heldout_member():
    inputs = _fallback_inputs()
    inputs["outer_train_row_ids"] = [10, 11, 12, 13, 14, 20]

    with pytest.raises(ValueError, match="outer-heldout row"):
        derive_sparse_query_moment_evidence(provenance=_provenance(), **inputs)


def test_sparse_fallback_fails_closed_instead_of_truncating_query_definitions():
    inputs = _fallback_inputs()
    inputs["tfidf_topic_evidence"]["discovery"]["topic_banks"]["treatment"][
        "topics"
    ] = [
        {
            "topic_id": f"topic-{index:03d}",
            "terms": [{"term": f"configured marker {index:03d}", "loading": 1.0}],
        }
        for index in range(25)
    ]

    with pytest.raises(ValueError, match="refusing silent definition omission"):
        derive_sparse_query_moment_evidence(
            provenance=_provenance(),
            config=QueryMomentEvidenceAdapterConfig(max_queries=24),
            **inputs,
        )


def test_sparse_fallback_fails_closed_instead_of_truncating_definition_terms():
    inputs = _fallback_inputs()
    inputs["tfidf_topic_evidence"]["discovery"]["topic_banks"]["treatment"][
        "topics"
    ][0]["terms"] = [
        {"term": f"configured term {index:03d}", "loading": 1.0}
        for index in range(33)
    ]

    with pytest.raises(ValueError, match="refusing silent term omission"):
        derive_sparse_query_moment_evidence(
            provenance=_provenance(),
            config=QueryMomentEvidenceAdapterConfig(max_terms_per_query=32),
            **inputs,
        )


def test_adapter_uses_sparse_fallback_when_artifact_is_absent(tmp_path):
    result = adapt_query_moment_evidence(
        provenance=_provenance(),
        artifact_path=tmp_path / "missing.json",
        **_fallback_inputs(),
    )

    assert result.audit["mode"] == "deterministic_sparse_fallback"
    assert result.audit["source_family"] == SPARSE_QUERY_MOMENTS
    assert result.as_fold_evidence_input().source_kind == SPARSE_QUERY_SOURCE


def test_sparse_adapter_payload_cannot_be_registered_as_neural(tmp_path):
    sparse = derive_sparse_query_moment_evidence(
        provenance=_provenance(),
        **_fallback_inputs(),
    )
    path = tmp_path / "misregistered_sparse.json"
    path.write_text(json.dumps(sparse.payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different source kind"):
        load_query_moment_evidence_artifact(path, provenance=_provenance())


def test_legacy_reseal_projects_scope_and_exact_subfold_partitions(tmp_path):
    evidence_path = tmp_path / "query_evidence.json"
    evidence_path.write_text(json.dumps(_artifact_query()), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scope": {
                    "outer_fold": 2,
                    "fit_row_fingerprint": row_set_fingerprint(
                        _provenance().train_row_ids
                    ),
                    "heldout_row_fingerprint": row_set_fingerprint(
                        _provenance().heldout_row_ids
                    ),
                },
                "ignored_diagnostic": {"sentry": "not part of the scope projection"},
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "query_subfold_audit.json"
    audit_path.write_text(
        json.dumps(
            [
                {
                    "fold": 1,
                    "identity_payload": {
                        "train_row_ids": [13, 14, 15],
                        "validation_row_ids": [10, 11, 12],
                        "ignored_labels": [0, 1, 0],
                    },
                },
                {
                    "fold": 2,
                    "identity_payload": {
                        "train_row_ids": [10, 11, 12],
                        "validation_row_ids": [13, 14, 15],
                        "ignored_labels": [1, 0, 1],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    bundle = reseal_legacy_neural_query_moment_evidence(
        query_evidence_path=evidence_path,
        query_subfold_audit_path=audit_path,
        summary_path=summary_path,
        provenance=_provenance(),
    )

    assert bundle["source_kind"] == "neural_query_moments"
    assert bundle["fit_row_ids"] == list(_provenance().train_row_ids)
    assert bundle["heldout_row_ids"] == list(_provenance().heldout_row_ids)
    assert bundle["source_provenance"]["exact_subfold_partitions_verified"] is True
    wrapped_path = tmp_path / "query_evidence.fold_scoped.json"
    wrapped_path.write_text(json.dumps(bundle), encoding="utf-8")
    loaded = load_query_moment_evidence_artifact(
        wrapped_path,
        provenance=_provenance(),
    )
    assert loaded.audit["artifact_declared_full_partition"] is True
    assert loaded.audit["artifact_declared_neural_source_kind"] is True
