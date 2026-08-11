"""Lazily loaded models used by Stage 1 and explicit-feature workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CausalForestHead": ("oci.models.causal_forest_head", "CausalForestHead"),
    "ECONML_AVAILABLE": ("oci.models.causal_forest_head", "ECONML_AVAILABLE"),
    "ConceptEmbeddingCache": (
        "oci.models.concept_embedding_cache",
        "ConceptEmbeddingCache",
    ),
    "HierarchicalTransformerExtractor": (
        "oci.models.hierarchical_transformer_extractor",
        "HierarchicalTransformerExtractor",
    ),
    "split_text_into_word_chunks": (
        "oci.models.hierarchical_transformer_extractor",
        "split_text_into_word_chunks",
    ),
    "GatedAttentionPooling": (
        "oci.models.gated_attention_pooling",
        "GatedAttentionPooling",
    ),
    "ExplicitFeatureFeaturizer": (
        "oci.models.explicit_feature_featurizer",
        "ExplicitFeatureFeaturizer",
    ),
    # Historical public names remain aliases for the role-aware featurizer.
    "ExplicitConfounderFeaturizer": (
        "oci.models.explicit_feature_featurizer",
        "ExplicitFeatureFeaturizer",
    ),
    "filter_specs_by_role": (
        "oci.models.explicit_feature_featurizer",
        "filter_specs_by_role",
    ),
    "get_raw_explicit_features": (
        "oci.models.explicit_feature_featurizer",
        "get_raw_explicit_features",
    ),
    "get_raw_confounder_features": (
        "oci.models.explicit_feature_featurizer",
        "get_raw_explicit_features",
    ),
    "get_raw_explicit_feature_matrices": (
        "oci.models.explicit_feature_featurizer",
        "get_raw_explicit_feature_matrices",
    ),
    "StructuredInteractionHead": (
        "oci.models.structured_interaction_head",
        "StructuredInteractionHead",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
