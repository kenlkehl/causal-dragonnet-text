from __future__ import annotations

import re

import pytest

from oci.inference.all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from oci.inference.production_coordinate_preserving_upstream_schema import (
    PRODUCTION_COORDINATE_PRESERVING_REGISTRY_VERSION,
    build_production_coordinate_preserving_schema,
    production_coordinate_preserving_registry_audit,
)

_VIEWS = (
    "linear_unigram_c0p5",
    "linear_1_2",
    "linear_1_3",
    "linear_2_4_min_df3",
    "extratrees_1_3",
    "random_forest_1_2",
)


def _config():
    return build_production_coordinate_preserving_schema(
        namespace="all_evidence_upstream",
        bow_view_names=_VIEWS,
        source_config_sha256="a" * 64,
        cluster_max_components=5,
        tfidf_topic_count=100,
        max_orphan_features=32,
        neural_query_counts={"treatment": 5, "outcome": 5, "effect": 5},
    )


def test_real_registry_preserves_every_configured_slot_without_v2_rank_truncation():
    config = _config()
    audit = production_coordinate_preserving_registry_audit(config)

    assert audit["schema_version"] == PRODUCTION_COORDINATE_PRESERVING_REGISTRY_VERSION
    assert audit["calibrated_source_count"] == 7
    assert audit["required_named_raw_coordinate_count"] == 44
    assert audit["conditional_named_raw_coordinate_count"] == 28
    assert audit["stable_named_raw_slot_count"] == 72
    assert audit["volatile_family_count"] == 6
    assert audit["volatile_raw_coordinate_capacity"] == 352
    assert audit["maximum_child_raw_coordinate_count"] == 424
    assert audit["emitted_raw_column_count"] == 464

    widths = {
        (item.source_kind, item.consumer_role): item.signed_order_width
        for item in config.volatile_raw_families
    }
    assert widths == {
        ("embedding_clustered", PROPENSITY_NUISANCE_FEATURE_ROLE): 10,
        ("embedding_clustered", UNCALIBRATED_EFFECT_MODIFIER_ROLE): 10,
        ("tfidf_topics", PROPENSITY_NUISANCE_FEATURE_ROLE): 100,
        ("tfidf_topics", OUTCOME_NUISANCE_FEATURE_ROLE): 100,
        ("tfidf_topic_contrast", UNCALIBRATED_EFFECT_MODIFIER_ROLE): 100,
        ("tfidf_orphan_ngrams", UNCALIBRATED_EFFECT_MODIFIER_ROLE): 32,
    }


def test_exact_nuisance_and_query_coordinates_precede_volatile_topic_remainders():
    config = _config()
    by_key = {
        (item.child_name, item.source_kind, item.consumer_role): item
        for item in config.named_raw_coordinates
    }
    for view in _VIEWS:
        assert (
            f"stage1_raw__bow__{view}__treatment_pred__as_propensity",
            "bow_nuisance",
            PROPENSITY_NUISANCE_FEATURE_ROLE,
        ) in by_key
        assert (
            f"stage1_raw__bow__{view}__outcome_pred__as_outcome",
            "bow_nuisance",
            OUTCOME_NUISANCE_FEATURE_ROLE,
        ) in by_key
    assert (
        "tfidf_nuisance_treatment",
        "tfidf_topics",
        PROPENSITY_NUISANCE_FEATURE_ROLE,
    ) in by_key
    assert (
        "tfidf_nuisance_outcome",
        "tfidf_topics",
        OUTCOME_NUISANCE_FEATURE_ROLE,
    ) in by_key
    for bank, kind, role in (
        (
            "treatment",
            "neural_query_treatment_moments",
            PROPENSITY_NUISANCE_FEATURE_ROLE,
        ),
        ("outcome", "neural_query_outcome_moments", OUTCOME_NUISANCE_FEATURE_ROLE),
        (
            "effect",
            "neural_query_effect_moments",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        ),
    ):
        for name in (
            f"neural_query_{bank}_signed_mean",
            f"neural_query_{bank}_absolute_max",
            *(f"neural_query_{bank}_signed_order_{rank:02d}" for rank in range(1, 6)),
        ):
            assert (name, kind, role) in by_key


def test_conditional_pair_and_whole_embedding_slots_have_presence_columns():
    config = _config()
    conditional = [item for item in config.named_raw_coordinates if not item.required]
    assert len(conditional) == 28
    assert sum(item.source_kind == "matched_pair_uplift" for item in conditional) == 14
    assert sum(item.source_kind == "embedding_whole_cohort" for item in conditional) == 14

    schema_names = [name for name, _kind, _role in config.raw_output_schema()]
    assert sum(name.endswith("__presence") for name in schema_names) == 28
    assert "stage1_raw__htr__matched_pair_uplift_delta_logit" in schema_names
    assert (
        "stage1_raw__embedding__global_confounder_average__mean_cosine__as_outcome" in schema_names
    )


def test_volatile_membership_patterns_are_bounded_to_configured_indices():
    config = _config()
    by_key = {(item.source_kind, item.consumer_role): item for item in config.volatile_raw_families}
    cluster = by_key[("embedding_clustered", PROPENSITY_NUISANCE_FEATURE_ROLE)]
    assert re.fullmatch(
        cluster.child_name_pattern,
        "stage1_raw__embedding__cluster_confounder_treatment_pc5__max_cosine__as_propensity",
    )
    assert not re.fullmatch(
        cluster.child_name_pattern,
        "stage1_raw__embedding__cluster_confounder_treatment_pc6__max_cosine__as_propensity",
    )
    topic = by_key[("tfidf_topic_contrast", UNCALIBRATED_EFFECT_MODIFIER_ROLE)]
    assert re.fullmatch(topic.child_name_pattern, "tfidf_effect_topic_100")
    assert not re.fullmatch(topic.child_name_pattern, "tfidf_effect_topic_101")
    orphan = by_key[("tfidf_orphan_ngrams", UNCALIBRATED_EFFECT_MODIFIER_ROLE)]
    assert re.fullmatch(orphan.child_name_pattern, "tfidf_orphan_032_0123456789ab")
    assert not re.fullmatch(orphan.child_name_pattern, "tfidf_orphan_033_0123456789ab")


@pytest.mark.parametrize(
    "overrides",
    [
        {"bow_view_names": _VIEWS[:5]},
        {"cluster_max_components": 0},
        {"tfidf_topic_count": 0},
        {"max_orphan_features": 0},
        {"neural_query_counts": {"treatment": 5, "outcome": 5}},
    ],
)
def test_registry_rejects_incomplete_configuration(overrides):
    arguments = {
        "namespace": "all_evidence_upstream",
        "bow_view_names": _VIEWS,
        "source_config_sha256": "a" * 64,
        "cluster_max_components": 5,
        "tfidf_topic_count": 100,
        "max_orphan_features": 32,
        "neural_query_counts": {"treatment": 5, "outcome": 5, "effect": 5},
    }
    arguments.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        build_production_coordinate_preserving_schema(**arguments)
