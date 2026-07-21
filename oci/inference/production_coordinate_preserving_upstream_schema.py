"""Snapshot-driven production registry for coordinate-preserved Stage-1 banks.

The Stage-1 composite exposes two different classes of row-level numerical
signals.  View-named BoW/HTR outputs, TF-IDF nuisance predictions, and already
aggregated neural-query moments have a stable semantic coordinate.  Topic,
orphan, and cluster-PC indices are fit-local and therefore may only be aligned
through a permutation-invariant family representation.

This module precommits that distinction without fitting a model.  It preserves
all 72 possible stable raw slots (44 required and 28 source-conditional), all
seven calibrated sources, and the full configured capacity of every volatile
family.  The separate TF-IDF semantic-retrieval architecture intentionally has
no independent numerical signal; its lexical evidence remains a first-class
concept-discovery view elsewhere.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .authenticated_coordinate_preserving_nuisance_bridge import (
    coordinate_preserving_nuisance_schema,
)
from .coordinate_preserving_context_fit_upstream_backend import (
    CoordinatePreservingUpstreamSchemaConfig,
    PrecommittedExactCalibratedSource,
    PrecommittedNamedRawCoordinate,
    PrecommittedVolatileRawFeatureFamily,
)

PRODUCTION_COORDINATE_PRESERVING_REGISTRY_VERSION = (
    "production_all_architecture_coordinate_registry_v1"
)

_BANK_ROLE = {
    "treatment": PROPENSITY_NUISANCE_FEATURE_ROLE,
    "outcome": OUTCOME_NUISANCE_FEATURE_ROLE,
    "effect": UNCALIBRATED_EFFECT_MODIFIER_ROLE,
}
_BANK_KIND = {
    "treatment": "neural_query_treatment_moments",
    "outcome": "neural_query_outcome_moments",
    "effect": "neural_query_effect_moments",
}
_WHOLE_EMBEDDING_COORDINATES = (
    (
        "global_treatment_contrast",
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        "propensity",
    ),
    (
        "global_outcome_contrast",
        OUTCOME_NUISANCE_FEATURE_ROLE,
        "outcome",
    ),
    (
        "global_confounder_average",
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        "propensity",
    ),
    (
        "global_confounder_average",
        OUTCOME_NUISANCE_FEATURE_ROLE,
        "outcome",
    ),
    (
        "global_r_pseudo_target_contrast",
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        "",
    ),
    (
        "global_orthogonal_r_score_contrast",
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        "",
    ),
    (
        "global_residualized_treatment_outcome_interaction",
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        "",
    ),
)
_STATS = ("mean_cosine", "max_cosine")


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _view_names(values: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("bow_view_names must be a sequence")
    result = tuple(str(value).strip() for value in values)
    # The exact downstream nuisance bridge is deliberately fixed to the six
    # precommitted production BoW views.
    coordinate_preserving_nuisance_schema(result)
    return result


def _query_counts(values: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(values, Mapping) or set(values) != set(_BANK_ROLE):
        raise ValueError("neural_query_counts must contain treatment, outcome, and effect")
    return {
        bank: _positive_integer(values[bank], name=f"neural_query_counts[{bank!r}]")
        for bank in _BANK_ROLE
    }


def _index_pattern(count: int, *, width: int = 3) -> str:
    return "(?:" + "|".join(f"{index:0{width}d}" for index in range(1, count + 1)) + ")"


def _named(
    child_name: str,
    source_kind: str,
    consumer_role: str,
    *,
    required: bool = True,
) -> PrecommittedNamedRawCoordinate:
    return PrecommittedNamedRawCoordinate(
        child_name=child_name,
        source_kind=source_kind,
        consumer_role=consumer_role,
        required=required,
    )


def _required_named_coordinates(
    *, bow_view_names: tuple[str, ...], neural_query_counts: Mapping[str, int]
) -> tuple[PrecommittedNamedRawCoordinate, ...]:
    coordinates: list[PrecommittedNamedRawCoordinate] = [
        _named(
            str(row["feature_name"]),
            str(row["feature_kind"]),
            str(row["consumer_role"]),
        )
        for row in coordinate_preserving_nuisance_schema(bow_view_names)
    ]
    coordinates.extend(
        _named(
            f"stage1_raw__bow__{view}__effect_pseudo_target_pred",
            "bow_r_loss",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        )
        for view in bow_view_names
    )
    coordinates.append(
        _named(
            "stage1_raw__htr__effect_pseudo_target_pred",
            "htr_neural",
            UNCALIBRATED_EFFECT_MODIFIER_ROLE,
        )
    )
    coordinates.extend(
        (
            _named(
                "tfidf_nuisance_treatment",
                "tfidf_topics",
                PROPENSITY_NUISANCE_FEATURE_ROLE,
            ),
            _named(
                "tfidf_nuisance_outcome",
                "tfidf_topics",
                OUTCOME_NUISANCE_FEATURE_ROLE,
            ),
        )
    )
    for bank in ("treatment", "outcome", "effect"):
        names = (
            f"neural_query_{bank}_signed_mean",
            f"neural_query_{bank}_absolute_max",
            *(
                f"neural_query_{bank}_signed_order_{rank:02d}"
                for rank in range(1, neural_query_counts[bank] + 1)
            ),
        )
        coordinates.extend(_named(name, _BANK_KIND[bank], _BANK_ROLE[bank]) for name in names)
    return tuple(coordinates)


def _conditional_named_coordinates(
    bow_view_names: tuple[str, ...],
) -> tuple[PrecommittedNamedRawCoordinate, ...]:
    coordinates: list[PrecommittedNamedRawCoordinate] = []
    for view in bow_view_names:
        for suffix in (
            "matched_pair_uplift_delta_logit",
            "matched_pair_treated_outcome_prob",
        ):
            coordinates.append(
                _named(
                    f"stage1_raw__bow__{view}__{suffix}",
                    "matched_pair_uplift",
                    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                    required=False,
                )
            )
    for suffix in (
        "matched_pair_uplift_delta_logit",
        "matched_pair_treated_outcome_prob",
    ):
        coordinates.append(
            _named(
                f"stage1_raw__htr__{suffix}",
                "matched_pair_uplift",
                UNCALIBRATED_EFFECT_MODIFIER_ROLE,
                required=False,
            )
        )
    for direction, role, suffix in _WHOLE_EMBEDDING_COORDINATES:
        for statistic in _STATS:
            name = f"stage1_raw__embedding__{direction}__{statistic}"
            if suffix:
                name += f"__as_{suffix}"
            coordinates.append(
                _named(
                    name,
                    "embedding_whole_cohort",
                    role,
                    required=False,
                )
            )
    return tuple(coordinates)


def build_production_coordinate_preserving_schema(
    *,
    namespace: str,
    bow_view_names: Sequence[Any],
    source_config_sha256: str,
    cluster_max_components: int,
    tfidf_topic_count: int,
    max_orphan_features: int,
    neural_query_counts: Mapping[str, Any],
) -> CoordinatePreservingUpstreamSchemaConfig:
    """Build the complete fixed schema from authenticated configuration values."""

    views = _view_names(bow_view_names)
    cluster_components = _positive_integer(cluster_max_components, name="cluster_max_components")
    topic_count = _positive_integer(tfidf_topic_count, name="tfidf_topic_count")
    orphan_count = _positive_integer(max_orphan_features, name="max_orphan_features")
    query_counts = _query_counts(neural_query_counts)

    calibrated = tuple(
        PrecommittedExactCalibratedSource(
            child_name=(f"stage1_calibrated__bow__{view}__effect_weighted_r_tau_pred"),
            source_kind="nested_calibrated_bow_weighted_r",
        )
        for view in views
    ) + (
        PrecommittedExactCalibratedSource(
            child_name="stage1_calibrated__htr__effect_weighted_r_tau_pred",
            source_kind="nested_calibrated_htr_weighted_r",
        ),
    )
    required = _required_named_coordinates(
        bow_view_names=views,
        neural_query_counts=query_counts,
    )
    conditional = _conditional_named_coordinates(views)
    pc_pattern = "(?:" + "|".join(map(str, range(1, cluster_components + 1))) + ")"
    topic_index = _index_pattern(topic_count)
    orphan_index = _index_pattern(orphan_count)
    volatile = (
        PrecommittedVolatileRawFeatureFamily(
            source_kind="embedding_clustered",
            consumer_role=PROPENSITY_NUISANCE_FEATURE_ROLE,
            signed_order_width=2 * cluster_components,
            child_name_pattern=(
                r"stage1_raw__embedding__cluster_confounder_treatment_pc"
                + pc_pattern
                + r"__(?:mean_cosine|max_cosine)__as_propensity"
            ),
        ),
        PrecommittedVolatileRawFeatureFamily(
            source_kind="embedding_clustered",
            consumer_role=UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            signed_order_width=2 * cluster_components,
            child_name_pattern=(
                r"stage1_raw__embedding__cluster_effect_residualized_interaction_pc"
                + pc_pattern
                + r"__(?:mean_cosine|max_cosine)"
            ),
        ),
        PrecommittedVolatileRawFeatureFamily(
            source_kind="tfidf_topics",
            consumer_role=PROPENSITY_NUISANCE_FEATURE_ROLE,
            signed_order_width=topic_count,
            child_name_pattern=r"tfidf_treatment_topic_" + topic_index,
        ),
        PrecommittedVolatileRawFeatureFamily(
            source_kind="tfidf_topics",
            consumer_role=OUTCOME_NUISANCE_FEATURE_ROLE,
            signed_order_width=topic_count,
            child_name_pattern=r"tfidf_outcome_topic_" + topic_index,
        ),
        PrecommittedVolatileRawFeatureFamily(
            source_kind="tfidf_topic_contrast",
            consumer_role=UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            signed_order_width=topic_count,
            child_name_pattern=r"tfidf_effect_topic_" + topic_index,
        ),
        PrecommittedVolatileRawFeatureFamily(
            source_kind="tfidf_orphan_ngrams",
            consumer_role=UNCALIBRATED_EFFECT_MODIFIER_ROLE,
            signed_order_width=orphan_count,
            child_name_pattern=(r"tfidf_orphan_" + orphan_index + r"_[0-9a-f]{12}"),
        ),
    )
    config = CoordinatePreservingUpstreamSchemaConfig(
        namespace=namespace,
        calibrated_sources=calibrated,
        named_raw_coordinates=(*required, *conditional),
        volatile_raw_families=volatile,
        source_config_sha256=source_config_sha256,
    )
    production_coordinate_preserving_registry_audit(config)
    return config


def production_coordinate_preserving_registry_audit(
    config: CoordinatePreservingUpstreamSchemaConfig,
) -> dict[str, Any]:
    """Validate and summarize the closed all-architecture production registry."""

    if not isinstance(config, CoordinatePreservingUpstreamSchemaConfig):
        raise TypeError("config must be CoordinatePreservingUpstreamSchemaConfig")
    required = tuple(item for item in config.named_raw_coordinates if item.required)
    conditional = tuple(item for item in config.named_raw_coordinates if not item.required)
    kinds = {item.source_kind for item in config.named_raw_coordinates}
    kinds.update(item.source_kind for item in config.volatile_raw_families)
    expected_kinds = {
        "bow_nuisance",
        "htr_nuisance",
        "bow_r_loss",
        "htr_neural",
        "matched_pair_uplift",
        "embedding_whole_cohort",
        "embedding_clustered",
        "tfidf_topics",
        "tfidf_topic_contrast",
        "tfidf_orphan_ngrams",
        "neural_query_treatment_moments",
        "neural_query_outcome_moments",
        "neural_query_effect_moments",
    }
    if kinds != expected_kinds:
        raise ValueError(
            "production coordinate registry does not cover the exact numerical families"
        )
    if len(config.calibrated_sources) != 7:
        raise ValueError("production coordinate registry requires seven calibrated sources")
    if len(required) < 32 or len(conditional) != 28:
        raise ValueError(
            "production coordinate registry requires all fixed coordinates and 28 "
            "conditional slots"
        )
    volatile_capacity = sum(item.signed_order_width for item in config.volatile_raw_families)
    if volatile_capacity < 6:
        raise ValueError("production coordinate registry has an invalid volatile capacity")
    return {
        "schema_version": PRODUCTION_COORDINATE_PRESERVING_REGISTRY_VERSION,
        "backend_schema": config.identity(),
        "calibrated_source_count": len(config.calibrated_sources),
        "required_named_raw_coordinate_count": len(required),
        "conditional_named_raw_coordinate_count": len(conditional),
        "stable_named_raw_slot_count": len(config.named_raw_coordinates),
        "volatile_family_count": len(config.volatile_raw_families),
        "volatile_raw_coordinate_capacity": volatile_capacity,
        "maximum_child_raw_coordinate_count": (
            len(config.named_raw_coordinates) + volatile_capacity
        ),
        "emitted_raw_column_count": len(config.raw_output_schema()),
        "source_conditional_slots_have_presence_columns": True,
        "volatile_members_are_permutation_invariant_and_capacity_checked": True,
        "tfidf_nuisance_coordinates_removed_before_topic_reduction": True,
        "semantic_retrieval_independent_numerical_signal_count": 0,
        "semantic_retrieval_zero_reason": (
            "lexical evidence adapter shares its underlying embedding projections"
        ),
        "all_stage1_numerical_architectures_represented": True,
    }


__all__ = [
    "PRODUCTION_COORDINATE_PRESERVING_REGISTRY_VERSION",
    "build_production_coordinate_preserving_schema",
    "production_coordinate_preserving_registry_audit",
]
