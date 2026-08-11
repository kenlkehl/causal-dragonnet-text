"""Lazily loaded model components for causal inference from text."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "DragonNet": ("oci.models.dragonnet", "DragonNet"),
    "RLearnerNet": ("oci.models.rlearner", "RLearnerNet"),
    "RoleGatedSlotRLearner": ("oci.models.rlearner", "RoleGatedSlotRLearner"),
    "FrozenLLMPoolerExtractor": (
        "oci.models.frozen_llm_pooler_extractor",
        "FrozenLLMPoolerExtractor",
    ),
    "HierarchicalLLMExtractor": (
        "oci.models.hierarchical_llm_extractor",
        "HierarchicalLLMExtractor",
    ),
    "HierarchicalTransformerExtractor": (
        "oci.models.hierarchical_transformer_extractor",
        "HierarchicalTransformerExtractor",
    ),
    "split_text_into_word_chunks": (
        "oci.models.hierarchical_transformer_extractor",
        "split_text_into_word_chunks",
    ),
    "HierarchicalCNNExtractor": (
        "oci.models.hierarchical_cnn_extractor",
        "HierarchicalCNNExtractor",
    ),
    "HierarchicalGRUExtractor": (
        "oci.models.hierarchical_gru_extractor",
        "HierarchicalGRUExtractor",
    ),
    "SimpleCNNExtractor": ("oci.models.simple_cnn_extractor", "SimpleCNNExtractor"),
    "ConceptEmbeddingCNNExtractor": (
        "oci.models.concept_embedding_cnn_extractor",
        "ConceptEmbeddingCNNExtractor",
    ),
    "ConceptTokenCNNExtractor": (
        "oci.models.concept_token_cnn_extractor",
        "ConceptTokenCNNExtractor",
    ),
    "SlotValueDiscoveryExtractor": (
        "oci.models.slot_value_discovery_extractor",
        "SlotValueDiscoveryExtractor",
    ),
    "ConceptEmbeddingCache": (
        "oci.models.concept_embedding_cache",
        "ConceptEmbeddingCache",
    ),
    "GatedAttentionPooling": (
        "oci.models.gated_attention_pooling",
        "GatedAttentionPooling",
    ),
    "LearnedTokenizer": ("oci.models.learned_tokenizer", "LearnedTokenizer"),
    "chunk_token_ids": ("oci.models.text_chunking", "chunk_token_ids"),
    "pad_and_batch_chunks": ("oci.models.text_chunking", "pad_and_batch_chunks"),
    "ExplicitFeatureFeaturizer": (
        "oci.models.explicit_feature_featurizer",
        "ExplicitFeatureFeaturizer",
    ),
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
    "HiddenStateCache": ("oci.models.hidden_state_cache", "HiddenStateCache"),
    "GPUHiddenStateStore": (
        "oci.models.gpu_hidden_state_store",
        "GPUHiddenStateStore",
    ),
    "CausalText": ("oci.models.causal_text", "CausalText"),
    "PropensityOnlyModel": (
        "oci.models.propensity_model",
        "PropensityOnlyModel",
    ),
    "PropensityNet": ("oci.models.propensity_model", "PropensityNet"),
    "create_propensity_model_from_config": (
        "oci.models.propensity_model",
        "create_propensity_model_from_config",
    ),
    "create_feature_extractor": (
        "oci.models.extractor_factory",
        "create_feature_extractor",
    ),
    "create_feature_extractor_from_config": (
        "oci.models.extractor_factory",
        "create_feature_extractor_from_config",
    ),
    "CausalForestHead": ("oci.models.causal_forest_head", "CausalForestHead"),
    "ECONML_AVAILABLE": ("oci.models.causal_forest_head", "ECONML_AVAILABLE"),
    "CausalTextForest": ("oci.models.causal_text_forest", "CausalTextForest"),
    "ContrastiveCausalTextForest": (
        "oci.models.contrastive_causal_text_forest",
        "ContrastiveCausalTextForest",
    ),
    "MatchedContrastiveEffectHead": (
        "oci.models.contrastive_causal_text_forest",
        "MatchedContrastiveEffectHead",
    ),
    "grad_reverse": ("oci.models.contrastive_causal_text_forest", "grad_reverse"),
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
