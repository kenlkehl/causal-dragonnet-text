from __future__ import annotations

from copy import deepcopy

from oci.inference.all_evidence_fusion import (
    EMBEDDING_CLUSTERED,
    EXACT_INNER_RECURRENCE_VERSION,
    FoldEvidenceInput,
    FoldEvidenceProvenance,
    LEGACY_ALL_SOURCE,
    LEGACY_COMPACTION_STRATEGY_VERSION,
    MATCHED_PAIR_UPLIFT,
    TFIDF_TOPICS,
    TFIDF_TOPIC_SOURCE,
    prepare_all_evidence_fusion,
)
from oci.inference.all_evidence_fusion_runner import (
    _build_exact_inner_recurrence,
    _tfidf_exact_inner_terms,
)
from oci.inference.multi_model_agentic_forest import _build_role_grouped_evidence_digest


def _provenance() -> FoldEvidenceProvenance:
    return FoldEvidenceProvenance(
        outer_fold=1,
        train_row_ids=(0, 1, 2, 3),
        heldout_row_ids=(4, 5),
        artifact_id="test",
    )


def _legacy_payload(*, reverse: bool = False) -> dict:
    bow = [
        {
            "source": f"linear_{index}.pseudo_target_positive",
            "view_name": f"linear_{index}",
            "evidence_type": "pseudo_target_positive",
            "meaning": "R-stage sparse-text signal",
            "rows": [{"feature": f"linear term {index}", "score": index + 1}],
        }
        for index in range(20)
    ]
    bow.extend(
        [
            {
                "source": "extratrees_1_3.pseudo_target_negative",
                "view_name": "extratrees_1_3",
                "evidence_type": "pseudo_target_negative",
                "meaning": "R-stage nonlinear sparse-text signal",
                "rows": [{"feature": "nonlinear retained term", "score": 5}],
            },
            {
                "source": "ensemble_r.linear_1_2.pseudo_target_positive",
                "view_name": "linear_1_2",
                "evidence_type": "pseudo_target_positive",
                "meaning": "ensemble R-stage sparse-text signal",
                "rows": [{"feature": "ensemble retained term", "score": 5}],
            },
            {
                "source": "matched_pair_uplift.linear_1_2.uplift_pair_features",
                "view_name": "linear_1_2",
                "evidence_type": "uplift_pair_features",
                "meaning": "matched-pair uplift sparse-text signal",
                "rows": [{"feature": "matched retained term", "score": 5}],
            },
        ]
    )
    embedding = [
        {
            "name": f"whole_effect_{index}",
            "contrast_family": "effect_residual",
            "positive_aligned_chunks": [{"text": f"whole embedding text {index}"}],
        }
        for index in range(20)
    ]
    embedding.append(
        {
            "name": "cluster_effect_component_7",
            "contrast_family": "cluster effect residual",
            "cluster_component_index": 7,
            "positive_aligned_chunks": [{"text": "cluster embedding retained text"}],
        }
    )
    htr = [
        {
            "stage": "effect",
            "rows": [{"attended_token_summary": f"HTR effect summary {index}"}],
        }
        for index in range(20)
    ]
    htr.append(
        {
            "stage": "pair_uplift",
            "rows": [{"attended_token_summary": "HTR pair retained summary"}],
        }
    )
    if reverse:
        bow.reverse()
        embedding.reverse()
        htr.reverse()
    return {
        "outer_fold": 1,
        "scope": "full_outer_train",
        "context": {
            "evidence_digest": {
                "confounders": {"bow_blurbs": [], "embedding_chunks": [], "htr_blurbs": []},
                "effect_modifiers": {
                    "bow_blurbs": bow,
                    "embedding_chunks": embedding,
                    "htr_blurbs": htr,
                },
            }
        },
    }


def test_legacy_compaction_interleaves_late_model_and_family_strata_deterministically():
    first = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(), _provenance())]
    )
    second = prepare_all_evidence_fusion(
        [FoldEvidenceInput(LEGACY_ALL_SOURCE, _legacy_payload(reverse=True), _provenance())]
    )

    assert first.context() == second.context()
    serialized = first.render_prompt()
    assert "nonlinear retained term" in serialized
    assert "ensemble retained term" in serialized
    assert "matched retained term" in serialized
    assert "cluster embedding retained text" in serialized
    assert "HTR pair retained summary" in serialized
    assert MATCHED_PAIR_UPLIFT in first.source_family_coverage["present_source_families"]
    assert EMBEDDING_CLUSTERED in first.source_family_coverage["present_source_families"]
    audit = first.source_family_coverage["legacy_compaction"]
    assert audit["schema_version"] == LEGACY_COMPACTION_STRATEGY_VERSION
    assert audit["discovered_group_count"] > audit["retained_group_count"]
    assert audit["retained_unique_value_count_by_axis"]["bow_model"] >= 4


def test_legacy_digest_carries_explicit_bow_model_even_for_opaque_view_names():
    digest = _build_role_grouped_evidence_digest(
        importance={
            "views": [
                {
                    "view_name": "opaque_view_name",
                    "view_config": {"bow_model": "xgboost"},
                    "pseudo_target_positive": [{"feature": "marker", "score": 2.0}],
                }
            ]
        },
        embedding_evidence={},
        htr_evidence={},
    )

    assert digest["effect_modifiers"]["bow_blurbs"][0]["bow_model"] == "xgboost"


def test_exact_inner_tfidf_recurrence_matches_terms_not_latent_topic_ids():
    fold_one = {
        "topic_banks": {
            "effect": {
                "topics": [
                    {"topic_id": "latent_001", "terms": [{"term": "Marker-Alpha"}]},
                    {"topic_id": "latent_shared", "terms": [{"term": "fold one only"}]},
                ]
            }
        }
    }
    fold_two = {
        "topic_banks": {
            "effect": {
                "topics": [
                    {"topic_id": "latent_999", "terms": [{"term": "marker alpha"}]},
                    {"topic_id": "latent_shared", "terms": [{"term": "fold two only"}]},
                ]
            }
        }
    }
    recurrence = _build_exact_inner_recurrence(
        {
            1: _tfidf_exact_inner_terms(fold_one),
            2: _tfidf_exact_inner_terms(fold_two),
        }
    )

    assert recurrence["schema_version"] == EXACT_INNER_RECURRENCE_VERSION
    assert recurrence["latent_topic_ids_compared_across_folds"] is False
    terms = [term for group in recurrence["groups"] for term in group["terms"]]
    assert terms == [
        {
            "term": "marker alpha",
            "inner_fold_support_count": 2,
            "occurrence_count": 2,
        }
    ]


def test_validated_exact_inner_recurrence_is_visible_as_support_evidence():
    recurrence = _build_exact_inner_recurrence(
        {
            1: {(TFIDF_TOPICS, "effect_modifier", "marker alpha")},
            2: {(TFIDF_TOPICS, "effect_modifier", "marker alpha")},
            3: {(TFIDF_TOPICS, "effect_modifier", "marker alpha")},
        }
    )
    payload = {
        "outer_fold": 1,
        "scope": "full_outer_train",
        "discovery": {
            "topic_banks": {
                "effect": {
                    "topics": [{"topic_id": "outer_only", "terms": [{"term": "outer term"}]}]
                }
            },
            "exact_inner_recurrence": deepcopy(recurrence),
        },
    }
    request = prepare_all_evidence_fusion(
        [FoldEvidenceInput(TFIDF_TOPIC_SOURCE, payload, _provenance())]
    )

    recurrence_blocks = [
        block.content
        for block in request.evidence_blocks
        if block.content.get("kind") == "exact_inner_normalized_term_recurrence"
    ]
    assert recurrence_blocks == [
        {
            "kind": "exact_inner_normalized_term_recurrence",
            "normalization_version": EXACT_INNER_RECURRENCE_VERSION,
            "inner_fold_count": 3,
            "discovered_recurrent_term_count": 1,
            "retained_term_count": 1,
            "terms": [
                {
                    "term": "marker alpha",
                    "inner_fold_support_count": 3,
                    "inner_fold_support_fraction": 1.0,
                    "occurrence_count": 3,
                }
            ],
        }
    ]
