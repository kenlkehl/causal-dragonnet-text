"""Deployment binding for the six role-neutral Stage 1 producers.

The execution coordinator deliberately knows nothing about cohort columns,
model locators, or producer hyperparameters.  This module is the concrete
single-node binding from an authenticated ``_PreparedBuild`` to that
coordinator interface.

Every scientific producer setting must be repeated in the corresponding
``ScientificWorkflowSpec.architecture_profiles`` entry and must agree with
the already validated Stage 1 or neural-query profile.  Dataclass defaults
are never used to fill a missing value.  Text-only held-out capabilities are
constructed separately from fit-row labels, and the matched-pair producer can
obtain nuisance probabilities only by freshly authenticating its already
completed sibling BoW artifact.
"""

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ..config import (
    BoWViewConfig,
    ClusterLocalEmbeddingScientificConfig,
    TfidfTopicDiscoveryConfig,
)
from ..models.hierarchical_transformer_extractor import (
    HierarchicalTransformerExtractor,
)
from .neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from .neural_query_context_backend import ContextFitNeuralQueryService
from .neural_numerical_replay import validate_neural_replay_settings
from .production_stage1_bundle import _PreparedBuild
from .production_stage1_cluster_preflight_artifact import (
    load_production_stage1_cluster_preflight_artifact,
)
from .production_stage1_role_neutral_execution import (
    BoundRoleNeutralComponentProducer,
    RoleNeutralComponentInvocation,
    RoleNeutralOperationalComponentReport,
    RoleNeutralProducerFactories,
)
from .review_spent_evidence_provider import (
    SemanticWitnessScientificConfig,
    SemanticWitnessTfidfVectorizerConfig,
)
from .role_neutral_all_ten_binding import (
    authenticate_role_neutral_bow_component,
    authenticate_role_neutral_embedding_component,
    authenticate_role_neutral_htr_component,
    authenticate_role_neutral_matched_pair_component,
    authenticate_role_neutral_neural_query_component,
    authenticate_role_neutral_tfidf_component,
)
from .role_neutral_bow_group_execution import (
    AuthenticatedRoleNeutralBoWNuisanceBank,
    RoleNeutralBoWPhysicalGroupRequest,
    execute_role_neutral_bow_physical_group,
    load_authenticated_role_neutral_bow_nuisance_bank,
)
from .role_neutral_embedding_group_execution import (
    EmbeddingContrastSpec,
    ExactHeldoutEmbeddingBatch,
    RoleNeutralEmbeddingPhysicalGroupRequest,
    RoleNeutralEmbeddingScientificConfig,
    execute_role_neutral_embedding_physical_group,
)
from .role_neutral_htr_group_execution import (
    RoleNeutralHTRConfig,
    RoleNeutralHTRPhysicalGroupRequest,
    execute_role_neutral_htr_physical_group,
)
from .role_neutral_matched_pair_group_execution import (
    RoleNeutralMatchedPairConfig,
    RoleNeutralMatchedPairPhysicalGroupRequest,
    execute_role_neutral_matched_pair_from_bow_nuisance_bank,
)
from .role_neutral_neural_query_group_execution import (
    COMPLETE_EMBEDDING_TEXT_POLICY,
    FAIL_CLOSED_EVIDENCE_CAPACITY_POLICY,
    REGISTERED_HELDOUT_TRANSFORM_POLICY,
    RoleNeutralNeuralQueryPhysicalGroupRequest,
    execute_role_neutral_neural_query_physical_group,
)
from .role_neutral_tfidf_group_execution import (
    RoleNeutralTfidfPhysicalGroupRequest,
    execute_role_neutral_tfidf_physical_group,
)


WORD_TREATMENT_OUTCOME = "word_treatment_outcome"
WORD_RESIDUAL_EFFECT = "word_residual_effect"
HIERARCHICAL_TRANSFORMER = "hierarchical_transformer"
MATCHED_PATIENT_UPLIFT = "matched_patient_uplift"
WHOLE_COHORT_EMBEDDINGS = "whole_cohort_embeddings"
CLUSTER_LOCAL_EMBEDDINGS = "cluster_local_embeddings"
LEXICAL_SEMANTIC_RETRIEVAL = "lexical_semantic_retrieval"
TFIDF_TOPICS_PROFILE = "tfidf_topics"
RESIDUAL_TFIDF_NGRAMS = "residual_tfidf_ngrams"
LEARNED_NEURAL_QUERIES = "learned_neural_queries"

_PROFILE_ORDER = (
    WORD_TREATMENT_OUTCOME,
    WORD_RESIDUAL_EFFECT,
    HIERARCHICAL_TRANSFORMER,
    MATCHED_PATIENT_UPLIFT,
    WHOLE_COHORT_EMBEDDINGS,
    CLUSTER_LOCAL_EMBEDDINGS,
    LEXICAL_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS_PROFILE,
    RESIDUAL_TFIDF_NGRAMS,
    LEARNED_NEURAL_QUERIES,
)

_BOW_CONFIGURATION_KEYS = frozenset(
    {"view_configs", "nuisance_folds", "effect_folds", "e_clip"}
)
_BOW_VIEW_KEYS = frozenset(field.name for field in fields(BoWViewConfig))
_HTR_CONFIGURATION_KEYS = frozenset(
    field.name
    for field in fields(RoleNeutralHTRConfig)
    if field.name != "model_tree_sha256"
)
_MATCHED_CONFIGURATION_KEYS = frozenset(
    field.name for field in fields(RoleNeutralMatchedPairConfig)
)
_MATCHED_EXTRACTOR_KEYS = frozenset(
    {
        "sentence_encoder_model",
        "freeze_sentence_encoder",
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "max_chunk_length",
        "num_transformer_layers",
        "num_attention_heads",
        "transformer_dim",
        "transformer_dropout",
        "projection_dim",
        "hash_embedding_dim",
        "sentence_encoder_batch_size",
        "sentence_encoder_backend",
        "sentence_pooling",
        "normalize_sentence_embeddings",
        "trainable_sentence_encoder_layers",
        "role_attention",
        "w_attention_heads",
        "x_attention_heads",
        "transformer_feedforward_dim",
        "transformer_activation",
        "transformer_norm_style",
        "transformer_layer_norm_eps",
        "transformer_layer_norm_elementwise_affine",
        "transformer_layer_norm_bias",
        "transformer_attention_dropout",
        "transformer_residual_dropout",
        "transformer_feedforward_dropout",
        "transformer_attention_bias",
        "transformer_feedforward_bias",
        "output_projection_depth",
        "output_projection_hidden_dim",
        "output_projection_activation",
        "output_projection_dropout",
        "output_projection_hidden_layer_norm",
        "output_projection_final_layer_norm",
        "output_projection_bias",
        "pool_token_init_std",
        "positional_encoding_base",
        "environment_override_policy",
    }
)
_EMBEDDING_CONFIGURATION_KEYS = frozenset(
    field.name for field in fields(RoleNeutralEmbeddingScientificConfig)
)
_EMBEDDING_CONTRAST_KEYS = frozenset(
    {field.name for field in fields(EmbeddingContrastSpec)}
    | {"target_source", "sample_weight_target_source"}
)
_TFIDF_CONFIGURATION_KEYS = frozenset(
    {
        "outcome_type",
        "bow_views",
        "nuisance_folds",
        "tfidf_nested_calibration_folds",
        "tfidf_topic",
    }
)
_TFIDF_TOPIC_KEYS = frozenset(
    field.name for field in fields(TfidfTopicDiscoveryConfig)
)
_QUERY_CONFIGURATION_KEYS = frozenset(
    {
        "query_config",
        "nuisance_folds",
        "outcome_type",
        "evidence_capacity_policy",
        "embedding_text_coverage_policy",
        "heldout_transform_policy",
        "replay_comparison_policy",
        "replay_relative_tolerance",
        "replay_absolute_tolerance",
    }
)
_QUERY_CONFIG_KEYS = frozenset(
    field.name for field in fields(NeuralQueryAgenticForestConfig)
)
_SEMANTIC_WITNESS_CONFIGURATION_KEYS = frozenset(
    field.name for field in fields(SemanticWitnessScientificConfig)
)
_SEMANTIC_WITNESS_VECTORIZER_KEYS = frozenset(
    field.name for field in fields(SemanticWitnessTfidfVectorizerConfig)
)
_CLUSTER_LOCAL_EMBEDDING_CONFIGURATION_KEYS = frozenset(
    field.name for field in fields(ClusterLocalEmbeddingScientificConfig)
)

_TARGET_SOURCE_REGISTRY = frozenset(
    {
        "fit_treatment",
        "fit_outcome",
        "fit_treatment_outcome_interaction",
        "fit_treatment_outcome_cell_code",
        "fit_r_pseudo_from_authenticated_bow_nuisance",
        "fit_orthogonal_r_score_from_authenticated_bow_nuisance",
        "fit_treatment_residual_squared_from_authenticated_bow_nuisance",
    }
)


class RoleNeutralScientificContractError(ValueError):
    """The portable scientific profile cannot fully configure a producer."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _profile(
    profiles: Mapping[str, Mapping[str, Any]],
    name: str,
) -> Mapping[str, Any]:
    value = profiles.get(name)
    if not isinstance(value, Mapping):
        raise RoleNeutralScientificContractError(
            f"architecture_profiles.{name} must be one configured object"
        )
    if value.get("enabled") is not True:
        raise RoleNeutralScientificContractError(
            f"architecture_profiles.{name}.enabled must be explicitly true"
        )
    implementation = value.get("implementation")
    if not isinstance(implementation, str) or not implementation.strip():
        raise RoleNeutralScientificContractError(
            f"architecture_profiles.{name}.implementation must be nonempty"
        )
    return value


def _require_exact_mapping(
    value: Any,
    *,
    expected_keys: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RoleNeutralScientificContractError(
            f"{label} must be an explicitly configured object"
        )
    observed = set(value)
    missing = sorted(expected_keys - observed)
    extra = sorted(observed - expected_keys)
    if missing or extra:
        raise RoleNeutralScientificContractError(
            f"{label} must be closed and explicit; "
            f"missing={missing}, extra={extra}"
        )
    return copy.deepcopy(dict(value))


def _require_exact_scientific_tree(
    value: Any,
    *,
    template: Any,
    label: str,
) -> None:
    """Recursively reject missing/extra leaves in a typed scientific tree."""

    if isinstance(template, Mapping):
        if not isinstance(value, Mapping):
            raise RoleNeutralScientificContractError(
                f"{label} must be an explicitly configured object"
            )
        observed = set(value)
        expected = set(template)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise RoleNeutralScientificContractError(
                f"{label} must be closed and explicit; "
                f"missing={missing}, extra={extra}"
            )
        for key in sorted(expected):
            _require_exact_scientific_tree(
                value[key],
                template=template[key],
                label=f"{label}.{key}",
            )


def _shared_profile(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
    primary: str,
) -> None:
    value = _profile(profiles, name)
    if value.get("shared_physical_producer") != primary:
        raise RoleNeutralScientificContractError(
            f"architecture_profiles.{name}.shared_physical_producer "
            f"must explicitly equal {primary!r}"
        )


def missing_role_neutral_architecture_profile_fields(
    architecture_profiles: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """List missing leaf contracts without instantiating any producer.

    This audit intentionally expands a missing configuration object into its
    scientific leaves so an operator sees the exact settings that otherwise
    would have fallen back to source-code or dataclass defaults.
    """

    if not isinstance(architecture_profiles, Mapping):
        return ("architecture_profiles",)
    missing: list[str] = []

    def require_profile(name: str) -> Mapping[str, Any]:
        value = architecture_profiles.get(name)
        if not isinstance(value, Mapping):
            missing.append(f"architecture_profiles.{name}")
            return {}
        for key in ("enabled", "implementation"):
            if key not in value:
                missing.append(f"architecture_profiles.{name}.{key}")
        return value

    def require_configuration(
        name: str,
        keys: Sequence[str],
        *,
        nested_lists: Mapping[str, Sequence[str]] | None = None,
        nested_mappings: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        profile = require_profile(name)
        configuration = profile.get("producer_configuration")
        if not isinstance(configuration, Mapping):
            for key in keys:
                missing.append(
                    "architecture_profiles."
                    f"{name}.producer_configuration.{key}"
                )
            for child_name, child_keys in (nested_lists or {}).items():
                for key in child_keys:
                    missing.append(
                        "architecture_profiles."
                        f"{name}.producer_configuration.{child_name}[].{key}"
                    )
            for child_name, child_keys in (
                nested_mappings or {}
            ).items():
                for key in child_keys:
                    missing.append(
                        "architecture_profiles."
                        f"{name}.producer_configuration.{child_name}.{key}"
                    )
            return
        for key in keys:
            if key not in configuration:
                missing.append(
                    "architecture_profiles."
                    f"{name}.producer_configuration.{key}"
                )
        for child_name, child_keys in (nested_lists or {}).items():
            children = configuration.get(child_name)
            if not isinstance(children, list) or not children:
                for key in child_keys:
                    missing.append(
                        "architecture_profiles."
                        f"{name}.producer_configuration.{child_name}[].{key}"
                    )
                continue
            for index, child in enumerate(children):
                if not isinstance(child, Mapping):
                    missing.append(
                        "architecture_profiles."
                        f"{name}.producer_configuration.{child_name}[{index}]"
                    )
                    continue
                for key in child_keys:
                    if key not in child:
                        missing.append(
                            "architecture_profiles."
                            f"{name}.producer_configuration."
                            f"{child_name}[{index}].{key}"
                        )
        for child_name, child_keys in (
            nested_mappings or {}
        ).items():
            child = configuration.get(child_name)
            if not isinstance(child, Mapping):
                for key in child_keys:
                    missing.append(
                        "architecture_profiles."
                        f"{name}.producer_configuration.{child_name}.{key}"
                    )
                continue
            for key in child_keys:
                if key not in child:
                    missing.append(
                        "architecture_profiles."
                        f"{name}.producer_configuration.{child_name}.{key}"
                    )

    def require_tree(value: Any, template: Any, path: str) -> None:
        if not isinstance(template, Mapping):
            return
        if not isinstance(value, Mapping):
            for key in template:
                missing.append(f"{path}.{key}")
            return
        for key, child_template in template.items():
            if key not in value:
                missing.append(f"{path}.{key}")
                continue
            require_tree(value[key], child_template, f"{path}.{key}")

    require_configuration(
        WORD_TREATMENT_OUTCOME,
        sorted(_BOW_CONFIGURATION_KEYS),
        nested_lists={"view_configs": sorted(_BOW_VIEW_KEYS)},
    )
    require_configuration(
        HIERARCHICAL_TRANSFORMER,
        sorted(_HTR_CONFIGURATION_KEYS),
    )
    require_configuration(
        MATCHED_PATIENT_UPLIFT,
        sorted(_MATCHED_CONFIGURATION_KEYS),
        nested_mappings={
            "htr_extractor": sorted(_MATCHED_EXTRACTOR_KEYS)
        },
    )
    require_configuration(
        WHOLE_COHORT_EMBEDDINGS,
        sorted(_EMBEDDING_CONFIGURATION_KEYS),
        nested_lists={
            "contrasts": sorted(_EMBEDDING_CONTRAST_KEYS)
        },
    )
    require_configuration(
        CLUSTER_LOCAL_EMBEDDINGS,
        sorted(_CLUSTER_LOCAL_EMBEDDING_CONFIGURATION_KEYS),
    )
    require_configuration(
        LEXICAL_SEMANTIC_RETRIEVAL,
        sorted(_SEMANTIC_WITNESS_CONFIGURATION_KEYS),
        nested_mappings={
            "retrieval_vectorizer": sorted(
                _SEMANTIC_WITNESS_VECTORIZER_KEYS
            ),
            "htr_vectorizer": sorted(
                _SEMANTIC_WITNESS_VECTORIZER_KEYS
            ),
        },
    )
    require_configuration(
        TFIDF_TOPICS_PROFILE,
        sorted(_TFIDF_CONFIGURATION_KEYS),
        nested_lists={"bow_views": sorted(_BOW_VIEW_KEYS)},
        nested_mappings={
            "tfidf_topic": sorted(_TFIDF_TOPIC_KEYS)
        },
    )
    require_configuration(
        LEARNED_NEURAL_QUERIES,
        sorted(_QUERY_CONFIGURATION_KEYS),
        nested_mappings={
            "query_config": sorted(_QUERY_CONFIG_KEYS)
        },
    )
    bow_profile = architecture_profiles.get(WORD_TREATMENT_OUTCOME)
    if isinstance(bow_profile, Mapping):
        bow_configuration = bow_profile.get("producer_configuration")
        if isinstance(bow_configuration, Mapping):
            raw_views = bow_configuration.get("view_configs")
            if isinstance(raw_views, list):
                if not raw_views:
                    missing.append(
                        "architecture_profiles.word_treatment_outcome."
                        "producer_configuration.view_configs[]"
                    )
                view_template = asdict(BoWViewConfig())
                for index, raw_view in enumerate(raw_views):
                    require_tree(
                        raw_view,
                        view_template,
                        "architecture_profiles.word_treatment_outcome."
                        f"producer_configuration.view_configs[{index}]",
                    )
    tfidf_profile = architecture_profiles.get(TFIDF_TOPICS_PROFILE)
    if isinstance(tfidf_profile, Mapping):
        tfidf_configuration = tfidf_profile.get("producer_configuration")
        if isinstance(tfidf_configuration, Mapping):
            raw_views = tfidf_configuration.get("bow_views")
            if isinstance(raw_views, list):
                if not raw_views:
                    missing.append(
                        "architecture_profiles.tfidf_topics."
                        "producer_configuration.bow_views[]"
                    )
                view_template = asdict(BoWViewConfig())
                for index, raw_view in enumerate(raw_views):
                    require_tree(
                        raw_view,
                        view_template,
                        "architecture_profiles.tfidf_topics."
                        f"producer_configuration.bow_views[{index}]",
                    )
            require_tree(
                tfidf_configuration.get("tfidf_topic"),
                asdict(TfidfTopicDiscoveryConfig()),
                "architecture_profiles.tfidf_topics."
                "producer_configuration.tfidf_topic",
            )
    lexical_profile = architecture_profiles.get(
        LEXICAL_SEMANTIC_RETRIEVAL
    )
    if isinstance(lexical_profile, Mapping):
        lexical_configuration = lexical_profile.get(
            "producer_configuration"
        )
        if isinstance(lexical_configuration, Mapping):
            lexical_path = (
                "architecture_profiles.lexical_semantic_retrieval."
                "producer_configuration"
            )
            for key in sorted(
                set(lexical_configuration)
                - _SEMANTIC_WITNESS_CONFIGURATION_KEYS
            ):
                missing.append(f"{lexical_path}.{key}")
            for vectorizer_name in (
                "retrieval_vectorizer",
                "htr_vectorizer",
            ):
                vectorizer = lexical_configuration.get(vectorizer_name)
                if not isinstance(vectorizer, Mapping):
                    continue
                for key in sorted(
                    set(vectorizer)
                    - _SEMANTIC_WITNESS_VECTORIZER_KEYS
                ):
                    missing.append(
                        f"{lexical_path}.{vectorizer_name}.{key}"
                    )
    cluster_profile = architecture_profiles.get(
        CLUSTER_LOCAL_EMBEDDINGS
    )
    if isinstance(cluster_profile, Mapping):
        cluster_configuration = cluster_profile.get(
            "producer_configuration"
        )
        if isinstance(cluster_configuration, Mapping):
            cluster_path = (
                "architecture_profiles.cluster_local_embeddings."
                "producer_configuration"
            )
            for key in sorted(
                set(cluster_configuration)
                - _CLUSTER_LOCAL_EMBEDDING_CONFIGURATION_KEYS
            ):
                missing.append(f"{cluster_path}.{key}")
    for name, primary in (
        (WORD_RESIDUAL_EFFECT, WORD_TREATMENT_OUTCOME),
        (CLUSTER_LOCAL_EMBEDDINGS, WHOLE_COHORT_EMBEDDINGS),
        (LEXICAL_SEMANTIC_RETRIEVAL, WHOLE_COHORT_EMBEDDINGS),
        (RESIDUAL_TFIDF_NGRAMS, TFIDF_TOPICS_PROFILE),
    ):
        profile = require_profile(name)
        if "shared_physical_producer" not in profile:
            missing.append(
                f"architecture_profiles.{name}.shared_physical_producer"
            )
        elif profile.get("shared_physical_producer") != primary:
            missing.append(
                f"architecture_profiles.{name}.shared_physical_producer"
            )
    return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class _ScientificBindings:
    bow_views: tuple[BoWViewConfig, ...]
    bow_nuisance_folds: int
    bow_effect_folds: int
    bow_e_clip: float
    htr: RoleNeutralHTRConfig
    matched_pair: RoleNeutralMatchedPairConfig
    embedding: RoleNeutralEmbeddingScientificConfig
    embedding_target_sources: Mapping[str, str]
    semantic_witness: SemanticWitnessScientificConfig
    neural_query_configuration: Mapping[str, Any]


def _bow_configuration(
    profile: Mapping[str, Any],
) -> tuple[tuple[BoWViewConfig, ...], int, int, float]:
    configuration = _require_exact_mapping(
        profile.get("producer_configuration"),
        expected_keys=_BOW_CONFIGURATION_KEYS,
        label=(
            "architecture_profiles.word_treatment_outcome."
            "producer_configuration"
        ),
    )
    raw_views = configuration["view_configs"]
    if not isinstance(raw_views, list) or not raw_views:
        raise RoleNeutralScientificContractError(
            "word_treatment_outcome producer_configuration.view_configs "
            "must be a nonempty configured list"
        )
    views: list[BoWViewConfig] = []
    for index, raw in enumerate(raw_views):
        values = _require_exact_mapping(
            raw,
            expected_keys=_BOW_VIEW_KEYS,
            label=(
                "architecture_profiles.word_treatment_outcome."
                f"producer_configuration.view_configs[{index}]"
            ),
        )
        _require_exact_scientific_tree(
            values,
            template=asdict(BoWViewConfig()),
            label=(
                "architecture_profiles.word_treatment_outcome."
                f"producer_configuration.view_configs[{index}]"
            ),
        )
        views.append(BoWViewConfig(**values))
    nuisance = configuration["nuisance_folds"]
    effect = configuration["effect_folds"]
    clip = configuration["e_clip"]
    if (
        isinstance(nuisance, bool)
        or not isinstance(nuisance, int)
        or nuisance < 2
        or isinstance(effect, bool)
        or not isinstance(effect, int)
        or effect < 2
        or isinstance(clip, bool)
        or not isinstance(clip, (int, float))
        or not 0.0 < float(clip) < 0.5
    ):
        raise RoleNeutralScientificContractError(
            "word-treatment/outcome fold or clipping configuration is invalid"
        )
    return tuple(views), int(nuisance), int(effect), float(clip)


def _htr_configuration(
    *,
    profile: Mapping[str, Any],
    model_tree_sha256: str,
) -> RoleNeutralHTRConfig:
    values = _require_exact_mapping(
        profile.get("producer_configuration"),
        expected_keys=_HTR_CONFIGURATION_KEYS,
        label=(
            "architecture_profiles.hierarchical_transformer."
            "producer_configuration"
        ),
    )
    values["model_tree_sha256"] = str(model_tree_sha256)
    return RoleNeutralHTRConfig.from_mapping(values)


def _matched_configuration(
    profile: Mapping[str, Any],
) -> RoleNeutralMatchedPairConfig:
    values = _require_exact_mapping(
        profile.get("producer_configuration"),
        expected_keys=_MATCHED_CONFIGURATION_KEYS,
        label=(
            "architecture_profiles.matched_patient_uplift."
            "producer_configuration"
        ),
    )
    values["htr_extractor"] = _require_exact_mapping(
        values["htr_extractor"],
        expected_keys=_MATCHED_EXTRACTOR_KEYS,
        label=(
            "architecture_profiles.matched_patient_uplift."
            "producer_configuration.htr_extractor"
        ),
    )
    return RoleNeutralMatchedPairConfig(**values)


def _embedding_configuration(
    profile: Mapping[str, Any],
) -> tuple[RoleNeutralEmbeddingScientificConfig, Mapping[str, str]]:
    values = _require_exact_mapping(
        profile.get("producer_configuration"),
        expected_keys=_EMBEDDING_CONFIGURATION_KEYS,
        label=(
            "architecture_profiles.whole_cohort_embeddings."
            "producer_configuration"
        ),
    )
    raw_contrasts = values["contrasts"]
    if not isinstance(raw_contrasts, list) or not raw_contrasts:
        raise RoleNeutralScientificContractError(
            "whole_cohort_embeddings producer_configuration.contrasts "
            "must be a nonempty configured list"
        )
    contrasts: list[EmbeddingContrastSpec] = []
    target_sources: dict[str, str] = {}
    for index, raw in enumerate(raw_contrasts):
        row = _require_exact_mapping(
            raw,
            expected_keys=_EMBEDDING_CONTRAST_KEYS,
            label=(
                "architecture_profiles.whole_cohort_embeddings."
                f"producer_configuration.contrasts[{index}]"
            ),
        )
        target_source = str(row.pop("target_source"))
        if target_source not in _TARGET_SOURCE_REGISTRY:
            raise RoleNeutralScientificContractError(
                "embedding target_source is unsupported; configured value "
                f"{target_source!r}, supported={sorted(_TARGET_SOURCE_REGISTRY)}"
            )
        sample_weight_target_source = row.pop(
            "sample_weight_target_source"
        )
        sample_weight_target_name = row.get(
            "sample_weight_target_name"
        )
        if sample_weight_target_name is None:
            if sample_weight_target_source is not None:
                raise RoleNeutralScientificContractError(
                    "embedding contrast without a sample-weight target cannot "
                    "configure sample_weight_target_source"
                )
        else:
            if (
                not isinstance(sample_weight_target_source, str)
                or sample_weight_target_source
                not in _TARGET_SOURCE_REGISTRY
            ):
                raise RoleNeutralScientificContractError(
                    "embedding sample_weight_target_source is unsupported; "
                    f"configured value {sample_weight_target_source!r}, "
                    f"supported={sorted(_TARGET_SOURCE_REGISTRY)}"
                )
        contrast = EmbeddingContrastSpec(**row)
        if (
            contrast.target_name in target_sources
            and target_sources[contrast.target_name] != target_source
        ):
            raise RoleNeutralScientificContractError(
                "one embedding target name maps to conflicting target sources"
            )
        target_sources[contrast.target_name] = target_source
        if contrast.sample_weight_target_name is not None:
            if (
                contrast.sample_weight_target_name in target_sources
                and target_sources[contrast.sample_weight_target_name]
                != sample_weight_target_source
            ):
                raise RoleNeutralScientificContractError(
                    "one embedding sample-weight target maps to conflicting "
                    "target sources"
                )
            target_sources[contrast.sample_weight_target_name] = str(
                sample_weight_target_source
            )
        contrasts.append(contrast)
    values["contrasts"] = tuple(contrasts)
    if (
        values["semantic_stop_words"] is not None
        and not isinstance(values["semantic_stop_words"], str)
    ):
        if not isinstance(values["semantic_stop_words"], list):
            raise RoleNeutralScientificContractError(
                "embedding semantic_stop_words must be null, the configured "
                "'english' vocabulary, or an explicit list"
            )
        values["semantic_stop_words"] = tuple(values["semantic_stop_words"])
    return (
        RoleNeutralEmbeddingScientificConfig(**values),
        copy.deepcopy(target_sources),
    )


def _cluster_local_embedding_configuration(
    *,
    profile: Mapping[str, Any],
    prepared: _PreparedBuild,
) -> ClusterLocalEmbeddingScientificConfig:
    values = _require_exact_mapping(
        profile.get("producer_configuration"),
        expected_keys=_CLUSTER_LOCAL_EMBEDDING_CONFIGURATION_KEYS,
        label=(
            "architecture_profiles.cluster_local_embeddings."
            "producer_configuration"
        ),
    )
    try:
        configured = ClusterLocalEmbeddingScientificConfig.from_mapping(
            values
        )
    except (TypeError, ValueError) as exc:
        raise RoleNeutralScientificContractError(
            "cluster_local_embeddings producer_configuration is invalid"
        ) from exc
    expected = (
        prepared.config.architecture.multi_model_forest.embedding_contrast
        .cluster_local_scientific
    )
    if (
        type(expected) is not ClusterLocalEmbeddingScientificConfig
        or configured.as_dict() != expected.as_dict()
    ):
        raise RoleNeutralScientificContractError(
            "cluster-local embedding producer configuration differs from "
            "the authenticated explicit Stage 1 profile"
        )
    return configured


def _semantic_witness_configuration(
    *,
    profile: Mapping[str, Any],
    prepared: _PreparedBuild,
) -> SemanticWitnessScientificConfig:
    label = (
        "architecture_profiles.lexical_semantic_retrieval."
        "producer_configuration"
    )
    try:
        configured = SemanticWitnessScientificConfig.from_mapping(
            profile.get("producer_configuration"),
            label=label,
        )
    except (TypeError, ValueError) as exc:
        raise RoleNeutralScientificContractError(str(exc)) from exc
    expected = prepared.semantic_witness_scientific_config
    if type(expected) is not SemanticWitnessScientificConfig:
        raise RoleNeutralScientificContractError(
            "prepared Stage 1 request omits its authenticated closed "
            "semantic-witness scientific configuration"
        )
    if _canonical(configured.as_dict()) != _canonical(expected.as_dict()):
        raise RoleNeutralScientificContractError(
            "lexical-semantic-retrieval producer configuration differs from "
            "the authenticated portable Stage 1 request"
        )
    return configured


def _tfidf_configuration(
    *,
    profile: Mapping[str, Any],
    prepared: _PreparedBuild,
) -> None:
    configured = _require_exact_mapping(
        profile.get("producer_configuration"),
        expected_keys=_TFIDF_CONFIGURATION_KEYS,
        label=(
            "architecture_profiles.tfidf_topics.producer_configuration"
        ),
    )
    forest = prepared.config.architecture.multi_model_forest
    expected = {
        "outcome_type": str(prepared.config.outcome_type),
        "bow_views": [asdict(view) for view in forest.bow_views],
        "nuisance_folds": int(forest.nuisance_folds),
        "tfidf_nested_calibration_folds": int(
            forest.tfidf_nested_calibration_folds
        ),
        "tfidf_topic": asdict(forest.tfidf_topic),
    }
    if _canonical(configured) != _canonical(expected):
        raise RoleNeutralScientificContractError(
            "architecture_profiles.tfidf_topics.producer_configuration "
            "differs from the authenticated explicit Stage 1 profile"
        )


def _neural_query_configuration(
    *,
    profile: Mapping[str, Any],
    prepared: _PreparedBuild,
) -> Mapping[str, Any]:
    configured = _require_exact_mapping(
        profile.get("producer_configuration"),
        expected_keys=_QUERY_CONFIGURATION_KEYS,
        label=(
            "architecture_profiles.learned_neural_queries."
            "producer_configuration"
        ),
    )
    configured["query_config"] = _require_exact_mapping(
        configured["query_config"],
        expected_keys=_QUERY_CONFIG_KEYS,
        label=(
            "architecture_profiles.learned_neural_queries."
            "producer_configuration.query_config"
        ),
    )
    validate_neural_replay_settings(
        policy=configured["replay_comparison_policy"],
        relative_tolerance=configured["replay_relative_tolerance"],
        absolute_tolerance=configured["replay_absolute_tolerance"],
    )
    expected = {
        "query_config": asdict(prepared.query_config),
        "nuisance_folds": int(prepared.options.query_nuisance_folds),
        "outcome_type": str(prepared.config.outcome_type),
        "evidence_capacity_policy": FAIL_CLOSED_EVIDENCE_CAPACITY_POLICY,
        "embedding_text_coverage_policy": COMPLETE_EMBEDDING_TEXT_POLICY,
        "heldout_transform_policy": REGISTERED_HELDOUT_TRANSFORM_POLICY,
    }
    observed_core = {
        key: value
        for key, value in configured.items()
        if not key.startswith("replay_")
    }
    if _canonical(observed_core) != _canonical(expected):
        raise RoleNeutralScientificContractError(
            "learned-neural-query producer configuration differs from the "
            "authenticated explicit query/Stage 1 profiles"
        )
    return copy.deepcopy(configured)


def _assert_bow_profile_matches_prepared(
    *,
    prepared: _PreparedBuild,
    views: Sequence[BoWViewConfig],
    nuisance_folds: int,
    effect_folds: int,
    e_clip: float,
) -> None:
    forest = prepared.config.architecture.multi_model_forest
    configured = {
        "view_configs": [asdict(value) for value in views],
        "nuisance_folds": int(nuisance_folds),
        "effect_folds": int(effect_folds),
        "e_clip": float(e_clip),
    }
    expected = {
        "view_configs": [asdict(value) for value in forest.bow_views],
        "nuisance_folds": int(forest.nuisance_folds),
        "effect_folds": int(forest.effect_folds),
        "e_clip": float(forest.e_clip),
    }
    if _canonical(configured) != _canonical(expected):
        raise RoleNeutralScientificContractError(
            "word-model producer configuration differs from the authenticated "
            "explicit Stage 1 profile"
        )


def _expected_htr_profile(prepared: _PreparedBuild) -> Mapping[str, Any]:
    architecture = prepared.config.architecture
    discovery = architecture.agentic_attention_variable_forest
    training = prepared.config.training
    if discovery.nuisance_epochs is None or discovery.effect_epochs is None:
        raise RoleNeutralScientificContractError(
            "legacy Stage 1 profile lacks finite "
            "agentic_attention_variable_forest nuisance/effect epochs required "
            "by the role-neutral HTR producer"
        )
    if training.effect_batch_size is None:
        raise RoleNeutralScientificContractError(
            "legacy Stage 1 profile lacks training.effect_batch_size required "
            "as the explicit HTR prediction batch size"
        )
    return {
        "sentence_encoder_model_kind": "authenticated_local_tree",
        "freeze_sentence_encoder": bool(
            architecture.htr_freeze_sentence_encoder
        ),
        "chunk_size_words": int(architecture.htr_chunk_size_words),
        "chunk_overlap_words": int(architecture.htr_chunk_overlap_words),
        "max_chunks": int(architecture.htr_max_chunks),
        "max_chunk_length": int(architecture.htr_max_chunk_length),
        "num_transformer_layers": int(architecture.htr_num_layers),
        "num_attention_heads": int(architecture.htr_num_heads),
        "transformer_dim": int(architecture.htr_transformer_dim),
        "transformer_dropout": float(architecture.htr_dropout),
        "projection_dim": int(architecture.htr_projection_dim),
        "hash_embedding_dim": int(architecture.htr_hash_embedding_dim),
        "sentence_encoder_batch_size": int(
            architecture.htr_sentence_encoder_batch_size
        ),
        "sentence_encoder_backend": str(
            architecture.htr_sentence_encoder_backend
        ),
        "sentence_pooling": str(architecture.htr_sentence_pooling),
        "normalize_sentence_embeddings": bool(
            architecture.htr_normalize_sentence_embeddings
        ),
        "trainable_sentence_encoder_layers": int(
            architecture.htr_trainable_sentence_encoder_layers
        ),
        "role_attention": bool(architecture.htr_role_attention),
        "w_attention_heads": int(architecture.htr_w_attention_heads),
        "x_attention_heads": int(architecture.htr_x_attention_heads),
        "transformer_feedforward_dim": int(
            architecture.htr_transformer_feedforward_dim
        ),
        "transformer_activation": str(
            architecture.htr_transformer_activation
        ),
        "transformer_norm_style": str(
            architecture.htr_transformer_norm_style
        ),
        "transformer_layer_norm_eps": float(
            architecture.htr_transformer_layer_norm_eps
        ),
        "transformer_layer_norm_elementwise_affine": bool(
            architecture.htr_transformer_layer_norm_elementwise_affine
        ),
        "transformer_layer_norm_bias": bool(
            architecture.htr_transformer_layer_norm_bias
        ),
        "transformer_attention_dropout": float(
            architecture.htr_transformer_attention_dropout
        ),
        "transformer_residual_dropout": float(
            architecture.htr_transformer_residual_dropout
        ),
        "transformer_feedforward_dropout": float(
            architecture.htr_transformer_feedforward_dropout
        ),
        "transformer_attention_bias": bool(
            architecture.htr_transformer_attention_bias
        ),
        "transformer_feedforward_bias": bool(
            architecture.htr_transformer_feedforward_bias
        ),
        "output_projection_depth": int(
            architecture.htr_output_projection_depth
        ),
        "output_projection_hidden_dim": int(
            architecture.htr_output_projection_hidden_dim
        ),
        "output_projection_activation": str(
            architecture.htr_output_projection_activation
        ),
        "output_projection_dropout": float(
            architecture.htr_output_projection_dropout
        ),
        "output_projection_hidden_layer_norm": bool(
            architecture.htr_output_projection_hidden_layer_norm
        ),
        "output_projection_final_layer_norm": bool(
            architecture.htr_output_projection_final_layer_norm
        ),
        "output_projection_bias": bool(
            architecture.htr_output_projection_bias
        ),
        "pool_token_init_std": float(
            architecture.htr_pool_token_init_std
        ),
        "positional_encoding_base": float(
            architecture.htr_positional_encoding_base
        ),
        "environment_override_policy": str(
            architecture.htr_environment_override_policy
        ),
        "require_live_unfrozen_encoder_attestation": bool(
            architecture.htr_require_live_unfrozen_encoder_attestation
        ),
        "hidden_dim": int(architecture.causal_head_hidden_outcome_dim),
        "nuisance_head_depth": int(architecture.htr_nuisance_head_depth),
        "nuisance_head_activation": str(
            architecture.htr_nuisance_head_activation
        ),
        "nuisance_head_dropout": float(
            architecture.htr_nuisance_head_dropout
        ),
        "nuisance_head_layer_norm": bool(
            architecture.htr_nuisance_head_layer_norm
        ),
        "nuisance_head_bias": bool(
            architecture.htr_nuisance_head_bias
        ),
        "effect_head_depth": int(architecture.htr_effect_head_depth),
        "effect_head_activation": str(
            architecture.htr_effect_head_activation
        ),
        "effect_head_dropout": float(
            architecture.htr_effect_head_dropout
        ),
        "effect_head_layer_norm": bool(
            architecture.htr_effect_head_layer_norm
        ),
        "effect_head_bias": bool(architecture.htr_effect_head_bias),
        "nuisance_folds": int(discovery.nuisance_folds),
        "effect_folds": int(discovery.effect_folds),
        "nuisance_epochs": int(discovery.nuisance_epochs),
        "effect_epochs": int(discovery.effect_epochs),
        "batch_size": int(training.batch_size),
        "prediction_batch_size": int(training.effect_batch_size),
        "optimizer_name": str(training.optimizer),
        "learning_rate": float(training.learning_rate),
        "weight_decay": float(training.weight_decay),
        "adamw_beta1": float(training.adamw_beta1),
        "adamw_beta2": float(training.adamw_beta2),
        "adamw_eps": float(training.adamw_eps),
        "adamw_amsgrad": bool(training.adamw_amsgrad),
        "adamw_maximize": bool(training.adamw_maximize),
        "adamw_foreach": bool(training.adamw_foreach),
        "adamw_capturable": bool(training.adamw_capturable),
        "adamw_differentiable": bool(training.adamw_differentiable),
        "adamw_fused": bool(training.adamw_fused),
        "optimizer_zero_grad_set_to_none": bool(
            training.optimizer_zero_grad_set_to_none
        ),
        "alpha_propensity": float(training.alpha_propensity),
        "nuisance_label_smoothing": float(
            discovery.nuisance_label_smoothing
        ),
        "nuisance_calibration": str(discovery.nuisance_calibration),
        "e_clip": float(discovery.e_clip),
        "r_stage_min_propensity": float(
            discovery.r_stage_min_propensity
        ),
        "r_stage_max_propensity": float(
            discovery.r_stage_max_propensity
        ),
        "gradient_clip_norm": float(training.gradient_clip_norm),
        "gradient_clip_norm_type": float(
            training.gradient_clip_norm_type
        ),
        "gradient_clip_error_if_nonfinite": bool(
            training.gradient_clip_error_if_nonfinite
        ),
        "gradient_clip_foreach": bool(training.gradient_clip_foreach),
        "effect_objectives": [str(discovery.effect_objective)],
        "outcome_type": str(prepared.config.outcome_type),
    }


def _expected_matched_profile(prepared: _PreparedBuild) -> Mapping[str, Any]:
    architecture = prepared.config.architecture
    forest = architecture.multi_model_forest
    discovery = architecture.agentic_attention_variable_forest
    training = prepared.config.training
    if discovery.effect_epochs is None:
        raise RoleNeutralScientificContractError(
            "legacy Stage 1 profile lacks "
            "agentic_attention_variable_forest.effect_epochs required by "
            "the matched-pair HTR subproducer"
        )
    if training.effect_batch_size is None:
        raise RoleNeutralScientificContractError(
            "legacy Stage 1 profile lacks training.effect_batch_size required "
            "by the matched-pair HTR subproducer"
        )
    extractor = {
        key: value
        for key, value in _expected_htr_profile(prepared).items()
        if key in (_MATCHED_EXTRACTOR_KEYS - {"sentence_encoder_model"})
    }
    extractor["sentence_encoder_model"] = "authenticated_local_tree"
    return {
        "effect_folds": int(forest.effect_folds),
        "propensity_caliper": float(
            forest.matched_pair_propensity_caliper
        ),
        "outcome_caliper": float(forest.matched_pair_outcome_caliper),
        "max_controls_per_candidate": int(
            forest.matched_pair_max_controls_per_candidate
        ),
        "nearest_fallback_controls": int(
            forest.matched_pair_nearest_fallback_controls
        ),
        "bow_l2_alpha": float(forest.matched_pair_bow_l2_alpha),
        "bow_max_iter": int(forest.matched_pair_bow_max_iter),
        "bow_optimizer_method": str(
            forest.matched_pair_bow_optimizer_method
        ),
        "bow_optimizer_ftol": float(
            forest.matched_pair_bow_optimizer_ftol
        ),
        "bow_optimizer_gtol": float(
            forest.matched_pair_bow_optimizer_gtol
        ),
        "bow_optimizer_maxls": int(
            forest.matched_pair_bow_optimizer_maxls
        ),
        "bow_optimizer_maxcor": int(
            forest.matched_pair_bow_optimizer_maxcor
        ),
        "bow_optimizer_maxfun": int(
            forest.matched_pair_bow_optimizer_maxfun
        ),
        "bow_optimizer_tol": (
            None
            if forest.matched_pair_bow_optimizer_tol is None
            else float(forest.matched_pair_bow_optimizer_tol)
        ),
        "bow_optimizer_initialization": str(
            forest.matched_pair_bow_optimizer_initialization
        ),
        "bow_require_optimizer_success": bool(
            forest.matched_pair_bow_require_optimizer_success
        ),
        "htr_epochs": int(discovery.effect_epochs),
        "htr_batch_size": int(training.effect_batch_size),
        "htr_learning_rate": float(training.learning_rate),
        "htr_weight_decay": float(training.weight_decay),
        "htr_optimizer_name": str(
            forest.matched_pair_htr_optimizer_name
        ),
        "htr_adamw_beta1": float(
            forest.matched_pair_htr_adamw_beta1
        ),
        "htr_adamw_beta2": float(
            forest.matched_pair_htr_adamw_beta2
        ),
        "htr_adamw_eps": float(forest.matched_pair_htr_adamw_eps),
        "htr_adamw_amsgrad": bool(
            forest.matched_pair_htr_adamw_amsgrad
        ),
        "htr_adamw_maximize": bool(
            forest.matched_pair_htr_adamw_maximize
        ),
        "htr_adamw_foreach": bool(
            forest.matched_pair_htr_adamw_foreach
        ),
        "htr_adamw_capturable": bool(
            forest.matched_pair_htr_adamw_capturable
        ),
        "htr_adamw_differentiable": bool(
            forest.matched_pair_htr_adamw_differentiable
        ),
        "htr_adamw_fused": bool(
            forest.matched_pair_htr_adamw_fused
        ),
        "htr_optimizer_zero_grad_set_to_none": bool(
            forest.matched_pair_htr_optimizer_zero_grad_set_to_none
        ),
        "htr_gradient_clip_norm": float(
            forest.matched_pair_htr_gradient_clip_norm
        ),
        "htr_gradient_clip_norm_type": float(
            forest.matched_pair_htr_gradient_clip_norm_type
        ),
        "htr_gradient_clip_error_if_nonfinite": bool(
            forest.matched_pair_htr_gradient_clip_error_if_nonfinite
        ),
        "htr_gradient_clip_foreach": bool(
            forest.matched_pair_htr_gradient_clip_foreach
        ),
        "htr_hidden_dim": int(architecture.causal_head_hidden_outcome_dim),
        "htr_dropout": float(architecture.htr_dropout),
        "htr_head_depth": int(forest.matched_pair_htr_head_depth),
        "htr_head_activation": str(
            forest.matched_pair_htr_head_activation
        ),
        "htr_head_layer_norm": bool(
            forest.matched_pair_htr_head_layer_norm
        ),
        "htr_head_bias": bool(forest.matched_pair_htr_head_bias),
        "htr_extractor": extractor,
    }


def _assert_embedding_profile_matches_prepared(
    *,
    prepared: _PreparedBuild,
    profile: Mapping[str, Any],
    config: RoleNeutralEmbeddingScientificConfig,
) -> None:
    legacy = (
        prepared.config.architecture.multi_model_forest.embedding_contrast
    )
    query = prepared.query_config
    semantic_expected = {
        "semantic_ngram_min": int(query.evidence_ngram_range_min),
        "semantic_ngram_max": int(query.evidence_ngram_range_max),
        "semantic_token_pattern": query.evidence_ngram_token_pattern,
        "semantic_lowercase": bool(query.evidence_ngram_lowercase),
        "semantic_strip_accents": query.evidence_ngram_strip_accents,
        "semantic_min_df": query.evidence_ngram_min_df,
        "semantic_max_df": query.evidence_ngram_max_df,
        "semantic_sublinear_tf": bool(query.evidence_ngram_sublinear_tf),
        "semantic_norm": query.evidence_ngram_norm,
        "semantic_use_idf": bool(query.evidence_ngram_use_idf),
        "semantic_smooth_idf": bool(query.evidence_ngram_smooth_idf),
        "semantic_stop_words": query.evidence_ngram_stop_words,
        "maximum_semantic_terms": (
            query.evidence_ngram_vocabulary_max_features
        ),
    }
    semantic_observed = {
        key: getattr(config, key) for key in semantic_expected
    }
    if _canonical(semantic_observed) != _canonical(semantic_expected):
        raise RoleNeutralScientificContractError(
            "whole-cohort embedding semantic configuration differs from "
            "the authenticated explicit neural-query profile"
        )
    if (
        bool(config.normalize_patient_embeddings)
        != bool(legacy.normalize_embeddings)
        or float(config.pseudo_target_quantile)
        != float(legacy.pseudo_target_quantile)
        or bool(config.pseudo_target_weighted)
        != bool(legacy.pseudo_target_weighted)
    ):
        raise RoleNeutralScientificContractError(
            "whole-cohort embedding normalization or pseudo-target settings "
            "differ from the authenticated explicit Stage 1 profile"
        )
    raw_rows = profile["producer_configuration"]["contrasts"]
    expected_rows = {
        (
            "marginal",
            "binary_zero_one",
            "fit_treatment",
            None,
        ),
        (
            "marginal",
            "binary_zero_one",
            "fit_outcome",
            None,
        ),
        (
            "r_pseudo_target",
            "configured_quantile_tails",
            "fit_r_pseudo_from_authenticated_bow_nuisance",
            (
                "fit_treatment_residual_squared_from_authenticated_bow_nuisance"
                if bool(legacy.pseudo_target_weighted)
                else None
            ),
        ),
    }
    if bool(legacy.include_cell_contrasts):
        expected_rows.update(
            {
                (
                    "within_treatment_arm_outcome",
                    "treated_arm_outcome_cell_difference",
                    "fit_treatment_outcome_cell_code",
                    None,
                ),
                (
                    "within_treatment_arm_outcome",
                    "untreated_arm_outcome_cell_difference",
                    "fit_treatment_outcome_cell_code",
                    None,
                ),
                (
                    "treatment_outcome_cell_interaction",
                    "treatment_outcome_cell_difference_in_differences",
                    "fit_treatment_outcome_cell_code",
                    None,
                ),
            }
        )
    if bool(legacy.include_confounder_vector_contrast):
        expected_rows.add(
            (
                "marginal_confounder_average",
                "average_normalized_treatment_outcome_marginals",
                "fit_treatment_outcome_cell_code",
                None,
            )
        )
    if bool(legacy.include_residualized_interaction_contrast):
        expected_rows.add(
            (
                "residualized_treatment_outcome_cell_interaction",
                (
                    "cell_difference_in_differences_"
                    "residualized_from_marginals"
                ),
                "fit_treatment_outcome_cell_code",
                None,
            )
        )
    if bool(legacy.include_orthogonal_r_score_contrasts):
        expected_rows.add(
            (
                "orthogonal_r_score",
                "configured_quantile_tails",
                "fit_orthogonal_r_score_from_authenticated_bow_nuisance",
                None,
            )
        )
    observed_rows = {
        (
            str(row["contrast_family"]),
            str(row["split_rule"]),
            str(row["target_source"]),
            (
                None
                if row["sample_weight_target_source"] is None
                else str(row["sample_weight_target_source"])
            ),
        )
        for row in raw_rows
    }
    if observed_rows != expected_rows or len(raw_rows) != len(expected_rows):
        raise RoleNeutralScientificContractError(
            "whole-cohort embedding contrasts differ from the explicitly "
            "enabled Stage 1 evidence families"
        )


def _scientific_bindings(
    *,
    prepared: _PreparedBuild,
    profiles: Mapping[str, Mapping[str, Any]],
) -> _ScientificBindings:
    missing = missing_role_neutral_architecture_profile_fields(profiles)
    if missing:
        raise RoleNeutralScientificContractError(
            "ScientificWorkflowSpec architecture_profiles omit role-neutral "
            "producer settings; missing scientific contract: "
            + ", ".join(missing)
        )
    if set(profiles) != set(_PROFILE_ORDER):
        raise RoleNeutralScientificContractError(
            "role-neutral factory requires exactly the ten architecture profiles"
        )
    for name in _PROFILE_ORDER:
        _profile(profiles, name)
    _shared_profile(
        profiles,
        name=WORD_RESIDUAL_EFFECT,
        primary=WORD_TREATMENT_OUTCOME,
    )
    _shared_profile(
        profiles,
        name=CLUSTER_LOCAL_EMBEDDINGS,
        primary=WHOLE_COHORT_EMBEDDINGS,
    )
    _shared_profile(
        profiles,
        name=LEXICAL_SEMANTIC_RETRIEVAL,
        primary=WHOLE_COHORT_EMBEDDINGS,
    )
    _shared_profile(
        profiles,
        name=RESIDUAL_TFIDF_NGRAMS,
        primary=TFIDF_TOPICS_PROFILE,
    )

    views, nuisance_folds, effect_folds, clip = _bow_configuration(
        _profile(profiles, WORD_TREATMENT_OUTCOME)
    )
    _assert_bow_profile_matches_prepared(
        prepared=prepared,
        views=views,
        nuisance_folds=nuisance_folds,
        effect_folds=effect_folds,
        e_clip=clip,
    )
    htr = _htr_configuration(
        profile=_profile(profiles, HIERARCHICAL_TRANSFORMER),
        model_tree_sha256=prepared.htr_model_sha256,
    )
    expected_htr = _expected_htr_profile(prepared)
    observed_htr = {
        key: value
        for key, value in htr.as_dict().items()
        if key
        not in {
            "schema_version",
            "model_tree_sha256",
            "text_truncation_applied",
            "replay_comparison_policy",
            "replay_relative_tolerance",
            "replay_absolute_tolerance",
        }
    }
    if _canonical(observed_htr) != _canonical(expected_htr):
        raise RoleNeutralScientificContractError(
            "hierarchical-transformer producer configuration differs from "
            "the authenticated explicit Stage 1 profile"
        )
    matched = _matched_configuration(
        _profile(profiles, MATCHED_PATIENT_UPLIFT)
    )
    observed_matched = {
        key: value
        for key, value in matched.as_dict().items()
        if key
        not in {
            "schema_version",
            "replay_comparison_policy",
            "replay_relative_tolerance",
            "replay_absolute_tolerance",
        }
    }
    if _canonical(observed_matched) != _canonical(
        {
            **_expected_matched_profile(prepared),
            "text_truncation_policy": (
                "forbidden_capacity_must_not_bind_v1"
            ),
            "matched_pair_subproducers": ["bow", "htr"],
        }
    ):
        raise RoleNeutralScientificContractError(
            "matched-pair producer configuration differs from the "
            "authenticated explicit Stage 1 profile"
        )
    embedding, target_sources = _embedding_configuration(
        _profile(profiles, WHOLE_COHORT_EMBEDDINGS)
    )
    _assert_embedding_profile_matches_prepared(
        prepared=prepared,
        profile=_profile(profiles, WHOLE_COHORT_EMBEDDINGS),
        config=embedding,
    )
    _cluster_local_embedding_configuration(
        profile=_profile(profiles, CLUSTER_LOCAL_EMBEDDINGS),
        prepared=prepared,
    )
    semantic_witness = _semantic_witness_configuration(
        profile=_profile(profiles, LEXICAL_SEMANTIC_RETRIEVAL),
        prepared=prepared,
    )
    _tfidf_configuration(
        profile=_profile(profiles, TFIDF_TOPICS_PROFILE),
        prepared=prepared,
    )
    neural = _neural_query_configuration(
        profile=_profile(profiles, LEARNED_NEURAL_QUERIES),
        prepared=prepared,
    )
    return _ScientificBindings(
        bow_views=views,
        bow_nuisance_folds=nuisance_folds,
        bow_effect_folds=effect_folds,
        bow_e_clip=clip,
        htr=htr,
        matched_pair=matched,
        embedding=embedding,
        embedding_target_sources=target_sources,
        semantic_witness=semantic_witness,
        neural_query_configuration=neural,
    )


@dataclass(frozen=True)
class _GroupInputs:
    fit_texts: tuple[str, ...]
    fit_treatment: tuple[float, ...]
    fit_outcome: tuple[float, ...]
    heldout_texts: tuple[str, ...]


def _group_inputs(
    prepared: _PreparedBuild,
    invocation: RoleNeutralComponentInvocation,
) -> _GroupInputs:
    if invocation.plan is not prepared.stage1_scope_plan:
        if (
            invocation.plan.scientific_content_sha256
            != prepared.stage1_scope_plan.scientific_content_sha256
            or invocation.physical_owner
            != prepared.stage1_scope_plan.scope(
                invocation.physical_owner.scope_id
            )
        ):
            raise ValueError("producer invocation belongs to another prepared plan")
    owner = invocation.physical_owner
    frame = prepared.modeling_data
    fit_rows = list(owner.fit_row_ids)
    heldout_rows = list(owner.heldout_row_ids)
    fit_projection = frame.iloc[fit_rows]
    heldout_text_projection = frame.iloc[heldout_rows][
        [prepared.config.text_column]
    ]
    fit_texts = tuple(
        fit_projection[prepared.config.text_column].tolist()
    )
    heldout_texts = tuple(
        heldout_text_projection[prepared.config.text_column].tolist()
    )
    if (
        len(fit_texts) != len(fit_rows)
        or len(heldout_texts) != len(heldout_rows)
        or any(not isinstance(value, str) for value in fit_texts)
        or any(not isinstance(value, str) for value in heldout_texts)
    ):
        raise ValueError("prepared cohort text differs from canonical scope rows")
    treatment = tuple(
        map(
            float,
            fit_projection[
                prepared.config.treatment_column
            ].to_numpy(dtype=float),
        )
    )
    outcome = tuple(
        map(
            float,
            fit_projection[
                prepared.config.outcome_column
            ].to_numpy(dtype=float),
        )
    )
    if (
        not np.isfinite(treatment).all()
        or not np.isfinite(outcome).all()
        or not set(treatment).issubset({0.0, 1.0})
        or not set(outcome).issubset({0.0, 1.0})
    ):
        raise ValueError("prepared physical-owner fit labels must be finite binary")
    # No held-out treatment/outcome projection is created or retained.
    return _GroupInputs(
        fit_texts=fit_texts,
        fit_treatment=treatment,
        fit_outcome=outcome,
        heldout_texts=heldout_texts,
    )


def _text_loader(
    *,
    owner_rows: tuple[int, ...],
    texts: tuple[str, ...],
) -> Callable[[tuple[int, ...]], tuple[str, ...]]:
    expected = tuple(owner_rows)

    def load(row_ids: tuple[int, ...]) -> tuple[str, ...]:
        if tuple(map(int, row_ids)) != expected:
            raise ValueError("producer requested another held-out row order")
        return texts

    return load


def _embedding_targets(
    *,
    inputs: _GroupInputs,
    target_sources: Mapping[str, str],
    nuisance_bank: AuthenticatedRoleNeutralBoWNuisanceBank,
    fit_row_ids: tuple[int, ...],
) -> Mapping[str, np.ndarray]:
    if type(nuisance_bank) is not AuthenticatedRoleNeutralBoWNuisanceBank:
        raise TypeError(
            "embedding targets require the authenticated sibling BoW "
            "nuisance bank"
        )
    if tuple(map(int, fit_row_ids)) != nuisance_bank.fit_row_ids:
        raise ValueError(
            "embedding target rows differ from the authenticated BoW "
            "nuisance bank"
        )
    treatment = np.asarray(inputs.fit_treatment, dtype=np.float64)
    outcome = np.asarray(inputs.fit_outcome, dtype=np.float64)
    propensity = np.asarray(
        nuisance_bank.fit_propensity_probability,
        dtype=np.float64,
    )
    outcome_nuisance = np.asarray(
        nuisance_bank.fit_outcome_nuisance_probability,
        dtype=np.float64,
    )
    if (
        propensity.shape != treatment.shape
        or outcome_nuisance.shape != outcome.shape
        or not np.isfinite(propensity).all()
        or not np.isfinite(outcome_nuisance).all()
        or np.any(propensity <= 0.0)
        or np.any(propensity >= 1.0)
    ):
        raise ValueError(
            "authenticated BoW nuisance probabilities cannot construct "
            "finite role-neutral embedding targets"
        )
    treatment_residual = treatment - propensity
    outcome_residual = outcome - outcome_nuisance
    if np.any(treatment_residual == 0.0):
        raise ValueError(
            "authenticated clipped propensity produced a zero treatment "
            "residual"
        )
    r_pseudo = outcome_residual / treatment_residual
    orthogonal = outcome_residual * treatment_residual
    residual_squared = np.square(treatment_residual)
    sources = {
        "fit_treatment": treatment,
        "fit_outcome": outcome,
        "fit_treatment_outcome_interaction": (
            (2.0 * treatment - 1.0) * (2.0 * outcome - 1.0)
        ),
        "fit_treatment_outcome_cell_code": 2.0 * treatment + outcome,
        "fit_r_pseudo_from_authenticated_bow_nuisance": r_pseudo,
        "fit_orthogonal_r_score_from_authenticated_bow_nuisance": orthogonal,
        "fit_treatment_residual_squared_from_authenticated_bow_nuisance": (
            residual_squared
        ),
    }
    result = {
        target_name: np.array(sources[source], copy=True)
        for target_name, source in target_sources.items()
    }
    if any(
        value.shape != treatment.shape or not np.isfinite(value).all()
        for value in result.values()
    ):
        raise ValueError("embedding target construction produced invalid values")
    return result


def _htr_extractor_factory(
    *,
    config: RoleNeutralMatchedPairConfig,
    model_path: Path,
) -> Callable[[torch.device], HierarchicalTransformerExtractor]:
    constructor = copy.deepcopy(dict(config.as_dict()["htr_extractor"]))
    if constructor["sentence_encoder_model"] == "authenticated_local_tree":
        constructor["sentence_encoder_model"] = str(model_path)

    def create(device: torch.device) -> HierarchicalTransformerExtractor:
        return HierarchicalTransformerExtractor(
            **copy.deepcopy(constructor),
            device=device,
        )

    return create


@dataclass(frozen=True)
class PreparedBuildRoleNeutralProducerFactoriesBuilder:
    """Callable deployment binding accepted by the portable workflow hook."""

    architecture_profiles: Mapping[str, Mapping[str, Any]]
    runtime_compatibility_class: str

    def __post_init__(self) -> None:
        profiles = copy.deepcopy(dict(self.architecture_profiles))
        if set(profiles) != set(_PROFILE_ORDER):
            missing = sorted(set(_PROFILE_ORDER) - set(profiles))
            extra = sorted(set(profiles) - set(_PROFILE_ORDER))
            raise RoleNeutralScientificContractError(
                "producer factory architecture profiles differ from all ten; "
                f"missing={missing}, extra={extra}"
            )
        runtime = str(self.runtime_compatibility_class).strip()
        if not runtime:
            raise ValueError("runtime_compatibility_class must be nonempty")
        object.__setattr__(self, "architecture_profiles", profiles)
        object.__setattr__(self, "runtime_compatibility_class", runtime)

    def __call__(
        self,
        prepared: _PreparedBuild,
    ) -> RoleNeutralProducerFactories:
        if not isinstance(prepared, _PreparedBuild):
            raise TypeError(
                "role-neutral producer binding requires ProductionStage1BundleBuilder.prepare()"
            )
        if (
            prepared.cluster_preflight_manifest_path is None
            or prepared.cluster_preflight_state_bundle is None
        ):
            raise RuntimeError(
                "role-neutral producer binding requires the authenticated "
                "clustered preflight and complete no-refit state bundle"
            )
        bindings = _scientific_bindings(
            prepared=prepared,
            profiles=self.architecture_profiles,
        )
        preflight = getattr(
            prepared,
            "cluster_preflight_artifact_handle",
            None,
        )
        if preflight is None:
            # Historical generic builders do not retain a typed payload
            # handle.  The portable workflow always does, so clustered owner
            # concepts remain lazy and are never reconstructed as one
            # aggregate audit mapping.
            preflight = load_production_stage1_cluster_preflight_artifact(
                manifest_path=prepared.cluster_preflight_manifest_path,
                config=prepared.config,
                registry=prepared.registry,
                registry_content_sha256=prepared.registry_content_sha256,
                embedding_cache_identity=prepared.embedding_cache_identity,
            )
        state_bundle = prepared.cluster_preflight_state_bundle
        physical_ids = {
            scope.scope_id for scope in prepared.stage1_scope_plan.physical_scopes
        }
        if (
            set(state_bundle.states) != physical_ids
            or not callable(
                getattr(state_bundle, "manifest_path_for_owner", None)
            )
        ):
            raise RuntimeError(
                "clustered preflight state bundle omits a physical owner"
            )

        def bow_factory(
            invocation: RoleNeutralComponentInvocation,
        ) -> BoundRoleNeutralComponentProducer:
            forest = prepared.config.architecture.multi_model_forest
            configured_fold_parallelism = forest.bow_fold_parallelism
            if configured_fold_parallelism is None:
                configured_fold_parallelism = forest.fold_parallelism
            owner_cpu_budget = invocation.owner_cpu_budget
            if owner_cpu_budget is None:
                owner_cpu_budget = prepared.options.num_workers
            request = RoleNeutralBoWPhysicalGroupRequest.from_plan(
                plan=invocation.plan,
                physical_owner_scope_id=invocation.physical_owner.scope_id,
            )
            inputs = _group_inputs(prepared, invocation)
            heldout_loader = _text_loader(
                owner_rows=request.physical_owner.heldout_row_ids,
                texts=inputs.heldout_texts,
            )
            return BoundRoleNeutralComponentProducer(
                execute=lambda: execute_role_neutral_bow_physical_group(
                    request=request,
                    output_root=invocation.output_root,
                    fit_texts=inputs.fit_texts,
                    fit_treatment=inputs.fit_treatment,
                    fit_outcome=inputs.fit_outcome,
                    view_configs=bindings.bow_views,
                    nuisance_folds=bindings.bow_nuisance_folds,
                    effect_folds=bindings.bow_effect_folds,
                    e_clip=bindings.bow_e_clip,
                    bow_fold_parallelism=configured_fold_parallelism,
                    bow_parallel_backend=forest.bow_parallel_backend,
                    owner_cpu_budget=owner_cpu_budget,
                    exact_heldout_text_loader=heldout_loader,
                ),
                authenticate=lambda: authenticate_role_neutral_bow_component(
                    root=invocation.output_root,
                    plan=invocation.plan,
                    physical_owner_scope_id=(
                        invocation.physical_owner.scope_id
                    ),
                ),
            )

        def htr_factory(
            invocation: RoleNeutralComponentInvocation,
        ) -> BoundRoleNeutralComponentProducer:
            request = RoleNeutralHTRPhysicalGroupRequest.from_plan(
                plan=invocation.plan,
                physical_owner_scope_id=invocation.physical_owner.scope_id,
            )
            inputs = _group_inputs(prepared, invocation)
            heldout_loader = _text_loader(
                owner_rows=request.physical_owner.heldout_row_ids,
                texts=inputs.heldout_texts,
            )

            def execute_htr() -> Any:
                controls = invocation.htr_operational_controls
                if controls is None:
                    return execute_role_neutral_htr_physical_group(
                        request=request,
                        output_root=invocation.output_root,
                        fit_texts=inputs.fit_texts,
                        fit_treatment=inputs.fit_treatment,
                        fit_outcome=inputs.fit_outcome,
                        config=bindings.htr,
                        runtime_compatibility_class=(
                            self.runtime_compatibility_class
                        ),
                        exact_heldout_text_loader=heldout_loader,
                        htr_model_path=prepared.htr_model_path,
                        device=invocation.resource,
                    )
                captured: list[Mapping[str, Any]] = []
                execute_role_neutral_htr_physical_group(
                    request=request,
                    output_root=invocation.output_root,
                    fit_texts=inputs.fit_texts,
                    fit_treatment=inputs.fit_treatment,
                    fit_outcome=inputs.fit_outcome,
                    config=bindings.htr,
                    runtime_compatibility_class=(
                        self.runtime_compatibility_class
                    ),
                    exact_heldout_text_loader=heldout_loader,
                    htr_model_path=prepared.htr_model_path,
                    device=invocation.resource,
                    operational_controls=controls,
                    operational_attestation_sink=captured.append,
                )
                if len(captured) != 1:
                    raise RuntimeError(
                        "HTR operational execution omitted its one attestation"
                    )
                return RoleNeutralOperationalComponentReport(
                    component="htr",
                    attestation=captured[0],
                )

            return BoundRoleNeutralComponentProducer(
                execute=execute_htr,
                authenticate=lambda: authenticate_role_neutral_htr_component(
                    root=invocation.output_root,
                    plan=invocation.plan,
                    physical_owner_scope_id=(
                        invocation.physical_owner.scope_id
                    ),
                    htr_model_path=prepared.htr_model_path,
                    device=invocation.resource,
                ),
            )

        def matched_factory(
            invocation: RoleNeutralComponentInvocation,
        ) -> BoundRoleNeutralComponentProducer:
            bow_request = RoleNeutralBoWPhysicalGroupRequest.from_plan(
                plan=invocation.plan,
                physical_owner_scope_id=invocation.physical_owner.scope_id,
            )
            nuisance_bank = (
                load_authenticated_role_neutral_bow_nuisance_bank(
                    root=invocation.output_root.parent / "bow",
                    request=bow_request,
                )
            )
            if type(nuisance_bank) is not AuthenticatedRoleNeutralBoWNuisanceBank:
                raise TypeError(
                    "matched-pair factory did not receive the authenticated "
                    "sibling BoW nuisance bank"
                )
            request = RoleNeutralMatchedPairPhysicalGroupRequest.from_plan(
                plan=invocation.plan,
                physical_owner_scope_id=invocation.physical_owner.scope_id,
                htr_model_identity_sha256=prepared.htr_model_sha256,
                nuisance_artifact_identity_sha256=(
                    nuisance_bank.content_sha256
                ),
                runtime_compatibility_class=(
                    self.runtime_compatibility_class
                ),
            )
            inputs = _group_inputs(prepared, invocation)
            heldout_loader = _text_loader(
                owner_rows=request.physical_owner.heldout_row_ids,
                texts=inputs.heldout_texts,
            )
            extractor_factory = _htr_extractor_factory(
                config=bindings.matched_pair,
                model_path=prepared.htr_model_path,
            )
            return BoundRoleNeutralComponentProducer(
                execute=lambda: (
                    execute_role_neutral_matched_pair_from_bow_nuisance_bank(
                        request=request,
                        output_root=invocation.output_root,
                        fit_texts=inputs.fit_texts,
                        fit_treatment=inputs.fit_treatment,
                        fit_outcome=inputs.fit_outcome,
                        nuisance_bank=nuisance_bank,
                        view_configs=bindings.bow_views,
                        config=bindings.matched_pair,
                        htr_extractor_factory=extractor_factory,
                        exact_heldout_text_loader=heldout_loader,
                        device=invocation.resource,
                        htr_model_path=prepared.htr_model_path,
                    )
                ),
                authenticate=lambda: (
                    authenticate_role_neutral_matched_pair_component(
                        root=invocation.output_root,
                        plan=invocation.plan,
                        physical_owner_scope_id=(
                            invocation.physical_owner.scope_id
                        ),
                        htr_model_identity_sha256=(
                            prepared.htr_model_sha256
                        ),
                        nuisance_artifact_identity_sha256=(
                            nuisance_bank.content_sha256
                        ),
                        runtime_compatibility_class=(
                            self.runtime_compatibility_class
                        ),
                    )
                ),
            )

        def embedding_factory(
            invocation: RoleNeutralComponentInvocation,
        ) -> BoundRoleNeutralComponentProducer:
            request = RoleNeutralEmbeddingPhysicalGroupRequest.from_plan(
                plan=invocation.plan,
                physical_owner_scope_id=invocation.physical_owner.scope_id,
            )
            bow_request = RoleNeutralBoWPhysicalGroupRequest.from_plan(
                plan=invocation.plan,
                physical_owner_scope_id=invocation.physical_owner.scope_id,
            )
            nuisance_bank = (
                load_authenticated_role_neutral_bow_nuisance_bank(
                    root=invocation.output_root.parent / "bow",
                    request=bow_request,
                )
            )
            if type(nuisance_bank) is not AuthenticatedRoleNeutralBoWNuisanceBank:
                raise TypeError(
                    "embedding factory did not receive the authenticated "
                    "sibling BoW nuisance bank"
                )
            inputs = _group_inputs(prepared, invocation)
            fit_provider = prepared.embedding_cache.bind_spent(
                request.physical_owner.fit_row_ids,
                inputs.fit_texts,
            )
            fit_targets = _embedding_targets(
                inputs=inputs,
                target_sources=bindings.embedding_target_sources,
                nuisance_bank=nuisance_bank,
                fit_row_ids=request.physical_owner.fit_row_ids,
            )
            exact_batch: dict[str, ExactHeldoutEmbeddingBatch] = {}

            def heldout_loader(
                row_ids: tuple[int, ...],
            ) -> ExactHeldoutEmbeddingBatch:
                expected = request.physical_owner.heldout_row_ids
                if tuple(map(int, row_ids)) != expected:
                    raise ValueError(
                        "embedding producer requested another held-out row order"
                    )
                batch = ExactHeldoutEmbeddingBatch(
                    row_ids=expected,
                    texts=inputs.heldout_texts,
                    embedding_provider=prepared.embedding_cache.bind_spent(
                        expected,
                        inputs.heldout_texts,
                    ),
                )
                exact_batch["value"] = batch
                return batch

            state_manifest = state_bundle.manifest_path_for_owner(
                request.physical_owner.scope_id
            )
            return BoundRoleNeutralComponentProducer(
                execute=lambda: execute_role_neutral_embedding_physical_group(
                    request=request,
                    output_root=invocation.output_root,
                    fit_texts=inputs.fit_texts,
                    fit_targets=fit_targets,
                    fit_embedding_provider=fit_provider,
                    scientific_config=bindings.embedding,
                    clustered_preflight=preflight,
                    clustered_preflight_state_manifest=state_manifest,
                    exact_heldout_loader=heldout_loader,
                ),
                authenticate=lambda: authenticate_role_neutral_embedding_component(
                    root=invocation.output_root,
                    plan=invocation.plan,
                    request=request,
                    clustered_preflight=preflight,
                    clustered_preflight_state_manifest=state_manifest,
                    expected_scientific_config=bindings.embedding,
                    expected_fit_texts=inputs.fit_texts,
                    expected_fit_targets=fit_targets,
                    expected_exact_batch=exact_batch.get("value"),
                ),
            )

        def tfidf_factory(
            invocation: RoleNeutralComponentInvocation,
        ) -> BoundRoleNeutralComponentProducer:
            request = RoleNeutralTfidfPhysicalGroupRequest.from_plan(
                plan=invocation.plan,
                physical_owner_scope_id=invocation.physical_owner.scope_id,
            )
            inputs = _group_inputs(prepared, invocation)
            heldout_loader = _text_loader(
                owner_rows=request.physical_owner.heldout_row_ids,
                texts=inputs.heldout_texts,
            )
            return BoundRoleNeutralComponentProducer(
                execute=lambda: execute_role_neutral_tfidf_physical_group(
                    request=request,
                    output_root=invocation.output_root,
                    fit_texts=inputs.fit_texts,
                    fit_treatment=inputs.fit_treatment,
                    fit_outcome=inputs.fit_outcome,
                    config=prepared.config,
                    exact_heldout_text_loader=heldout_loader,
                ),
                authenticate=lambda: authenticate_role_neutral_tfidf_component(
                    root=invocation.output_root,
                    plan=invocation.plan,
                    physical_owner_scope_id=(
                        invocation.physical_owner.scope_id
                    ),
                ),
            )

        def neural_factory(
            invocation: RoleNeutralComponentInvocation,
        ) -> BoundRoleNeutralComponentProducer:
            inputs = _group_inputs(prepared, invocation)
            scratch = tempfile.TemporaryDirectory(
                prefix=".role-neutral-neural-query-",
                dir=invocation.output_root.parent,
            )
            try:
                service = ContextFitNeuralQueryService(
                    cache_dir=Path(scratch.name) / "executable_cache",
                    dataset_path=prepared.options.dataset_path,
                    text_column=prepared.config.text_column,
                    embedding_cache=prepared.embedding_cache,
                    stage1_config_path=prepared.options.config_path,
                    query_config=prepared.query_config,
                    nuisance_folds=int(
                        bindings.neural_query_configuration[
                            "nuisance_folds"
                        ]
                    ),
                    devices=(
                        invocation.neural_query_execution_topology.devices
                    ),
                    seed=int(invocation.physical_owner.scope_seed),
                    outcome_type=str(prepared.config.outcome_type),
                )
                request = (
                    RoleNeutralNeuralQueryPhysicalGroupRequest.from_plan(
                        plan=invocation.plan,
                        physical_owner_scope_id=(
                            invocation.physical_owner.scope_id
                        ),
                        query_config=bindings.neural_query_configuration[
                            "query_config"
                        ],
                        nuisance_folds=int(
                            bindings.neural_query_configuration[
                                "nuisance_folds"
                            ]
                        ),
                        seed=int(invocation.physical_owner.scope_seed),
                        outcome_type=str(prepared.config.outcome_type),
                        service_scientific_identity=service.identity(),
                        evidence_capacity_policy=(
                            bindings.neural_query_configuration[
                                "evidence_capacity_policy"
                            ]
                        ),
                        embedding_text_coverage_policy=(
                            bindings.neural_query_configuration[
                                "embedding_text_coverage_policy"
                            ]
                        ),
                        replay_comparison_policy=(
                            bindings.neural_query_configuration[
                                "replay_comparison_policy"
                            ]
                        ),
                        replay_relative_tolerance=(
                            bindings.neural_query_configuration[
                                "replay_relative_tolerance"
                            ]
                        ),
                        replay_absolute_tolerance=(
                            bindings.neural_query_configuration[
                                "replay_absolute_tolerance"
                            ]
                        ),
                        heldout_transform_policy=(
                            bindings.neural_query_configuration[
                                "heldout_transform_policy"
                            ]
                        ),
                    )
                )
            except BaseException:
                scratch.cleanup()
                raise
            heldout_loader = _text_loader(
                owner_rows=request.physical_owner.heldout_row_ids,
                texts=inputs.heldout_texts,
            )

            def execute_neural() -> Mapping[str, Any]:
                try:
                    return execute_role_neutral_neural_query_physical_group(
                        request=request,
                        output_root=invocation.output_root,
                        service=service,
                        fit_texts=inputs.fit_texts,
                        fit_treatment=inputs.fit_treatment,
                        fit_outcome=inputs.fit_outcome,
                        execution_topology=(
                            invocation.neural_query_execution_topology
                        ),
                        heldout_text_loader=heldout_loader,
                    )
                finally:
                    scratch.cleanup()

            return BoundRoleNeutralComponentProducer(
                execute=execute_neural,
                authenticate=lambda: (
                    authenticate_role_neutral_neural_query_component(
                        root=invocation.output_root,
                        plan=invocation.plan,
                        request=request,
                    )
                ),
            )

        return RoleNeutralProducerFactories(
            bow=bow_factory,
            htr=htr_factory,
            matched_pair=matched_factory,
            embeddings=embedding_factory,
            tfidf=tfidf_factory,
            neural_query=neural_factory,
        )


__all__ = [
    "PreparedBuildRoleNeutralProducerFactoriesBuilder",
    "RoleNeutralScientificContractError",
    "missing_role_neutral_architecture_profile_fields",
]
