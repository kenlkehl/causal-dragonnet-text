from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from oci.inference.all_evidence_fusion import (
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPIC_SOURCE,
    prepare_all_evidence_fusion,
    source_text_temporal_policy_audit,
)
from oci.inference.tfidf_orphan_evidence_adapter import (
    OrphanNgramEvidenceCapacityOverflowError,
    OrphanNgramEvidenceAdapterConfig,
    adapt_full_outer_orphan_ngram_evidence,
)
from oci.inference.tfidf_topic_discovery import row_set_fingerprint


def _topic_banks() -> dict:
    return {
        "treatment": {
            "topics": [
                {
                    "topic_id": "treatment_001",
                    "terms": [{"term": "amber lattice", "loading": 0.8}],
                }
            ]
        },
        "outcome": {
            "topics": [
                {
                    "topic_id": "outcome_001",
                    "terms": [{"term": "quiet orchard", "loading": 0.7}],
                }
            ]
        },
        "effect": {
            "topics": [
                {
                    "topic_id": "effect_001",
                    "terms": [{"term": "silver compass", "loading": 0.9}],
                }
            ]
        },
    }


def _score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [
                "copper meadow",
                "copper meadow marker",
                "violet bridge",
                "violet bridge signal",
                "amber lattice",
                "patient id 12345678",
                "treatment response",
                "appointment scheduling",
                "discarded but strong",
                "weak residual",
            ],
            "signed_score": [4.5, 4.1, -3.9, -3.6, 9.0, 8.0, 7.0, 6.0, 10.0, 0.2],
            "unsigned_score": [4.5, 4.1, 3.9, 3.6, 9.0, 8.0, 7.0, 6.0, 10.0, 0.2],
            "combined_importance": [4.4, 4.0, 3.8, 3.5, 9.0, 8.0, 7.0, 6.0, 10.0, 0.2],
            "eligible": [True, True, True, True, True, True, True, True, False, True],
            "support_control": [20] * 10,
            "support_treated": [22] * 10,
        }
    )


def _row(score_path, *, digest: str | None = None) -> dict:
    fit_ids = [0, 1, 2, 3, 4, 5]
    heldout_ids = [6, 7]
    reference = {"path": str(score_path)}
    if digest is not None:
        reference["sha256"] = digest
    discovery = {
        "schema_version": "tfidf_topic_discovery_v3_safe_arrays",
        "scope_id": "outer_002_full_train",
        "fit_row_ids": fit_ids,
        "heldout_row_ids": heldout_ids,
        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "heldout_score_tests_enabled": False,
        "topic_banks": _topic_banks(),
        "topic_score_tests": {
            "status": "not_run",
            "reason": "outer labels reserved",
            "uses_heldout_treatment_and_outcome": False,
        },
        "artifacts": {
            "ngram_scores": {"effect": reference},
            "topic_score_tests": None,
        },
    }
    return {
        "schema_version": "multi_model_forest_handoff_v2",
        "outer_fold": 2,
        "inner_fold": None,
        "scope": "full_outer_train",
        "fit_row_ids": fit_ids,
        "heldout_row_ids": heldout_ids,
        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
        "discovery": discovery,
    }


def _write_scores(tmp_path, *, name="effect_ngram_scores.parquet"):
    directory = tmp_path / "outer_002_full_train"
    directory.mkdir()
    path = directory / name
    _score_frame().to_parquet(path, index=False)
    return path


def _adapter_config(**overrides) -> OrphanNgramEvidenceAdapterConfig:
    values = {
        "min_abs_fit_score": 2.0,
        "lexical_overlap_threshold": 0.5,
        "max_candidates": None,
        "max_clusters": None,
        "max_terms_per_cluster": None,
        "max_term_chars": None,
        "max_ngram_tokens": None,
    }
    values.update(overrides)
    return OrphanNgramEvidenceAdapterConfig(**values)


def test_builds_authenticated_compact_branch_and_fusion_payload(tmp_path):
    score_path = _write_scores(tmp_path)
    digest = hashlib.sha256(score_path.read_bytes()).hexdigest()

    result = adapt_full_outer_orphan_ngram_evidence(
        _row(score_path, digest=digest),
        score_path,
        config=_adapter_config(),
    )

    branch = result.branch
    assert branch["status"] == "completed"
    assert branch["selected_cluster_ids"] == [
        "effect_orphan_outer_002_001",
        "effect_orphan_outer_002_002",
        "effect_orphan_outer_002_003",
    ]
    assert [
        [term["term"] for term in cluster["terms"]] for cluster in branch["selected_clusters"]
    ] == [
        ["treatment response"],
        ["copper meadow", "copper meadow marker"],
        ["violet bridge", "violet bridge signal"],
    ]
    first = branch["selected_clusters"][0]["terms"][0]
    assert first["signed_score"] == pytest.approx(7.0)
    assert first["fit_signed_score"] == pytest.approx(7.0)
    assert first["fit_rank"] == 7
    assert result.audit["artifact"]["sha256"] == digest
    assert result.audit["artifact"]["declared_sha256_verified"] is True
    assert result.audit["represented_topic_term_exclusion_count"] == 1
    assert result.audit["identifier_noise_exclusion_count"] == 1
    assert "post_treatment_noise_exclusion_count" not in result.audit
    assert result.audit["source_text_temporal_policy"] == source_text_temporal_policy_audit()
    assert result.audit["administrative_noise_exclusion_count"] == 1
    assert result.audit["ineligible_row_count"] == 1
    assert result.audit["below_min_abs_fit_score_count"] == 1
    assert result.audit["heldout_text_or_labels_accessed"] is False
    assert "source_artifact_audit" in branch
    assert (
        "source_artifact_audit"
        not in result.fusion_payload["discovery"]["effect_orphan_ngram_branch"]
    )

    provenance = FoldEvidenceProvenance(
        outer_fold=2,
        train_row_ids=(0, 1, 2, 3, 4, 5),
        heldout_row_ids=(6, 7),
        artifact_id="tfidf-orphan-fold-2",
    )
    request = prepare_all_evidence_fusion(
        [
            FoldEvidenceInput(
                source_kind=TFIDF_TOPIC_SOURCE,
                payload=result.fusion_payload,
                provenance=provenance,
            )
        ]
    )
    assert TFIDF_ORPHAN_NGRAMS in request.source_family_coverage["present_source_families"]


def test_selection_is_deterministic_and_uses_lexical_overlap_only(tmp_path):
    score_path = _write_scores(tmp_path)
    config = OrphanNgramEvidenceAdapterConfig(
        min_abs_fit_score=2.0,
        lexical_overlap_threshold=0.5,
        max_candidates=None,
        max_clusters=3,
        max_terms_per_cluster=2,
        max_term_chars=None,
        max_ngram_tokens=None,
    )

    first = adapt_full_outer_orphan_ngram_evidence(_row(score_path), score_path, config=config)
    second = adapt_full_outer_orphan_ngram_evidence(_row(score_path), score_path, config=config)

    assert first.branch == second.branch
    assert first.audit == second.audit
    assert all(1 <= len(cluster["terms"]) <= 2 for cluster in first.branch["selected_clusters"])


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda row: row.update(scope="candidate_selection_inner_fit"), "full_outer_train"),
        (
            lambda row: row["discovery"].update(heldout_score_tests_enabled=True),
            "heldout score tests",
        ),
        (
            lambda row: row["discovery"]["topic_score_tests"].update(
                status="completed", uses_heldout_treatment_and_outcome=True
            ),
            "heldout-scored",
        ),
        (
            lambda row: row["discovery"]["artifacts"].update(
                topic_score_tests="topic_score_tests.json"
            ),
            "heldout/test score artifact",
        ),
        (lambda row: row.update(oracle_effect=0.9), "forbidden true/oracle field"),
        (
            lambda row: row["discovery"].update(fit_row_ids=[0, 1, 2, 3, 4, 7]),
            "discovery fit rows",
        ),
    ],
)
def test_rejects_non_outer_or_leaky_discovery_rows(tmp_path, mutation, error):
    score_path = _write_scores(tmp_path)
    row = _row(score_path)
    mutation(row)

    with pytest.raises(ValueError, match=error):
        adapt_full_outer_orphan_ngram_evidence(
            row,
            score_path,
            config=_adapter_config(),
        )


def test_rejects_wrong_path_or_registered_hash(tmp_path):
    score_path = _write_scores(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_path = other_dir / "effect_ngram_scores.parquet"
    _score_frame().to_parquet(other_path, index=False)

    with pytest.raises(ValueError, match="path does not match"):
        adapt_full_outer_orphan_ngram_evidence(
            _row(score_path),
            other_path,
            config=_adapter_config(),
        )

    with pytest.raises(ValueError, match="SHA-256"):
        adapt_full_outer_orphan_ngram_evidence(
            _row(score_path, digest="0" * 64),
            score_path,
            config=_adapter_config(),
        )


def test_rejects_heldout_filename_and_forbidden_score_fields(tmp_path):
    heldout_path = _write_scores(tmp_path, name="heldout_effect_ngram_scores.parquet")
    with pytest.raises(ValueError, match="heldout/test"):
        adapt_full_outer_orphan_ngram_evidence(
            _row(heldout_path),
            heldout_path,
            config=_adapter_config(),
        )

    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    score_path = safe_dir / "effect_ngram_scores.parquet"
    frame = _score_frame()
    frame["oracle_score"] = 1.0
    frame.to_parquet(score_path, index=False)
    with pytest.raises(ValueError, match="forbidden heldout/test fields"):
        adapt_full_outer_orphan_ngram_evidence(
            _row(score_path),
            score_path,
            config=_adapter_config(),
        )


def test_rejects_non_outer_fit_rows_declared_inside_score_table(tmp_path):
    directory = tmp_path / "outer_002_full_train"
    directory.mkdir()
    score_path = directory / "effect_ngram_scores.parquet"
    frame = _score_frame()
    frame["scope"] = "outer_fit"
    frame.loc[0, "scope"] = "heldout"
    frame.to_parquet(score_path, index=False)

    with pytest.raises(ValueError, match="non-outer-fit score rows"):
        adapt_full_outer_orphan_ngram_evidence(
            _row(score_path),
            score_path,
            config=_adapter_config(),
        )


def test_candidate_and_cluster_capacities_fail_closed_without_selection(tmp_path):
    score_path = _write_scores(tmp_path)

    with pytest.raises(
        OrphanNgramEvidenceCapacityOverflowError,
        match="max_candidates=2.*no candidates were silently discarded",
    ):
        adapt_full_outer_orphan_ngram_evidence(
            _row(score_path),
            score_path,
            config=_adapter_config(max_candidates=2),
        )

    with pytest.raises(
        OrphanNgramEvidenceCapacityOverflowError,
        match="max_clusters=2.*no cluster was silently discarded",
    ):
        adapt_full_outer_orphan_ngram_evidence(
            _row(score_path),
            score_path,
            config=_adapter_config(max_clusters=2),
        )


def test_term_capacities_fail_closed_without_text_or_member_omission(tmp_path):
    score_path = _write_scores(tmp_path)

    with pytest.raises(
        OrphanNgramEvidenceCapacityOverflowError,
        match="max_term_chars.*no term text was silently discarded",
    ):
        adapt_full_outer_orphan_ngram_evidence(
            _row(score_path),
            score_path,
            config=_adapter_config(max_term_chars=5),
        )

    with pytest.raises(
        OrphanNgramEvidenceCapacityOverflowError,
        match="max_ngram_tokens.*no term was silently discarded",
    ):
        adapt_full_outer_orphan_ngram_evidence(
            _row(score_path),
            score_path,
            config=_adapter_config(max_ngram_tokens=1),
        )

    with pytest.raises(
        OrphanNgramEvidenceCapacityOverflowError,
        match="max_terms_per_cluster=1.*no cluster member was silently discarded",
    ):
        adapt_full_outer_orphan_ngram_evidence(
            _row(score_path),
            score_path,
            config=_adapter_config(max_terms_per_cluster=1),
        )
