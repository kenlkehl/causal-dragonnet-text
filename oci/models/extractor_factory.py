# oci/models/extractor_factory.py
"""Factory function for creating feature extractors.

This module centralizes feature extractor instantiation logic that was previously
duplicated across CausalText, CausalTextForest, and PropensityOnlyModel.
"""

import logging
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn

from ..config import normalize_feature_extractor_type


logger = logging.getLogger(__name__)


def create_feature_extractor(
    extractor_type: str,
    device: torch.device,
    # Frozen LLM Pooler args
    flp_model_name: str = "Qwen/Qwen3-0.6B-Base",
    flp_max_length: int = 8192,
    flp_freeze_llm: bool = True,
    flp_gated_attention_dim: int = 128,
    flp_projection_dim: int = 128,
    flp_dropout: float = 0.1,
    flp_gradient_checkpointing: bool = True,
    flp_downprojection_dim: Optional[int] = None,
    flp_skip_llm: bool = False,
    flp_cached_hidden_size: int = 0,
    flp_chat_template_prompt: Optional[str] = None,
    # Hierarchical LLM args
    hlm_model_name: str = "Qwen/Qwen3-0.6B-Base",
    hlm_chunk_size: int = 2048,
    hlm_chunk_overlap: int = 256,
    hlm_max_chunks: int = 16,
    hlm_freeze_llm: bool = True,
    hlm_gated_attention_dim: int = 128,
    hlm_projection_dim: int = 128,
    hlm_dropout: float = 0.1,
    hlm_gradient_checkpointing: bool = True,
    hlm_downprojection_dim: Optional[int] = None,
    hlm_skip_llm: bool = False,
    hlm_cached_hidden_size: int = 0,
    hlm_chat_template_prompt: Optional[str] = None,
    # Hierarchical Transformer args
    htr_sentence_model: str = "prajjwal1/bert-tiny",
    htr_freeze_sentence_encoder: bool = False,
    htr_chunk_size_words: int = 96,
    htr_chunk_overlap_words: int = 24,
    htr_max_chunks: int = 128,
    htr_max_chunk_length: int = 128,
    htr_num_layers: int = 2,
    htr_num_heads: int = 4,
    htr_transformer_dim: int = 256,
    htr_dropout: float = 0.1,
    htr_projection_dim: int = 128,
    htr_hash_embedding_dim: int = 256,
    htr_sentence_encoder_batch_size: int = 128,
    htr_sentence_encoder_backend: str = "auto",
    htr_sentence_pooling: str = "auto",
    htr_normalize_sentence_embeddings: bool = True,
    htr_trainable_sentence_encoder_layers: int = 0,
    # Hierarchical CNN args
    hcnn_embedding_dim: int = 256,
    hcnn_conv_dim: int = 256,
    hcnn_kernel_size: int = 5,
    hcnn_num_conv_blocks: int = 4,
    hcnn_chunk_size: int = 512,
    hcnn_chunk_overlap: int = 64,
    hcnn_max_chunks: int = 32,
    hcnn_vocab_size: int = 50000,
    hcnn_gated_attention_dim: int = 128,
    hcnn_projection_dim: int = 128,
    hcnn_dropout: float = 0.1,
    # Hierarchical GRU args
    hgru_embedding_dim: int = 256,
    hgru_gru_hidden_dim: int = 256,
    hgru_num_gru_layers: int = 2,
    hgru_chunk_size: int = 512,
    hgru_chunk_overlap: int = 64,
    hgru_max_chunks: int = 32,
    hgru_vocab_size: int = 50000,
    hgru_gated_attention_dim: int = 128,
    hgru_projection_dim: int = 128,
    hgru_dropout: float = 0.1,
    # Simple CNN args
    scnn_embedding_dim: int = 256,
    scnn_conv_dim: int = 256,
    scnn_kernel_size: int = 5,
    scnn_num_conv_blocks: int = 4,
    scnn_max_length: int = 10000,
    scnn_vocab_size: int = 50000,
    scnn_gated_attention_dim: int = 128,
    scnn_projection_dim: int = 128,
    scnn_dropout: float = 0.1,
    # Concept embedding CNN args
    cecnn_sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cecnn_chunk_size_words: int = 64,
    cecnn_chunk_overlap_words: int = 16,
    cecnn_max_chunks: int = 128,
    cecnn_confounder_concepts: Optional[List[str]] = None,
    cecnn_effect_modifier_concepts: Optional[List[str]] = None,
    cecnn_random_features: int = 0,
    cecnn_random_confounder_features: Optional[int] = None,
    cecnn_random_modifier_features: Optional[int] = None,
    cecnn_kernel_role: str = "combined",
    cecnn_projection_dim: int = 128,
    cecnn_dropout: float = 0.1,
    cecnn_anchor_weight: float = 0.01,
    cecnn_cached_embedding_dim: int = 0,
    cecnn_normalize_embeddings: bool = True,
    cecnn_random_state: int = 42,
    # Concept token CNN args
    ctcnn_model_name: str = "Qwen/Qwen3-0.6B-Base",
    ctcnn_chunk_size: int = 2048,
    ctcnn_chunk_overlap: int = 256,
    ctcnn_max_chunks: int = 16,
    ctcnn_confounder_concepts: Optional[List[str]] = None,
    ctcnn_effect_modifier_concepts: Optional[List[str]] = None,
    ctcnn_random_features: int = 0,
    ctcnn_random_confounder_features: Optional[int] = None,
    ctcnn_random_modifier_features: Optional[int] = None,
    ctcnn_kernel_role: str = "combined",
    ctcnn_projection_dim: int = 128,
    ctcnn_dropout: float = 0.1,
    ctcnn_anchor_weight: float = 0.01,
    ctcnn_cached_hidden_size: int = 0,
    ctcnn_downprojection_dim: Optional[int] = None,
    ctcnn_normalize_embeddings: bool = True,
    ctcnn_random_state: int = 42,
    # Slot-value discovery args
    svx_sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    svx_chunk_size_words: int = 64,
    svx_chunk_overlap_words: int = 16,
    svx_max_chunks: int = 128,
    svx_confounder_concepts: Optional[List[str]] = None,
    svx_effect_modifier_concepts: Optional[List[str]] = None,
    svx_num_free_slots: int = 16,
    svx_slot_dim: int = 128,
    svx_num_value_prototypes: int = 4,
    svx_dropout: float = 0.1,
    svx_anchor_weight: float = 0.01,
    svx_cached_embedding_dim: int = 0,
    svx_normalize_embeddings: bool = True,
    svx_attention_temperature: float = 0.1,
    svx_attention_entropy_weight: float = 0.0,
    svx_query_diversity_weight: float = 0.0,
    svx_random_state: int = 42,
    # Model type
    model_type: str = "dragonnet",
) -> nn.Module:
    """
    Create a feature extractor based on the specified type.

    Args:
        extractor_type: Type of feature extractor
        device: PyTorch device to use
        model_type: Model type ("dragonnet", "rlearner", etc.)

    Returns:
        nn.Module: The instantiated feature extractor
    """
    normalized_type = normalize_feature_extractor_type(extractor_type)

    if normalized_type == "frozen_llm_pooler":
        from .frozen_llm_pooler_extractor import FrozenLLMPoolerExtractor
        extractor = FrozenLLMPoolerExtractor(
            model_name=flp_model_name,
            max_length=flp_max_length,
            freeze_llm=flp_freeze_llm,
            gated_attention_dim=flp_gated_attention_dim,
            projection_dim=flp_projection_dim,
            dropout=flp_dropout,
            gradient_checkpointing=flp_gradient_checkpointing,
            downprojection_dim=flp_downprojection_dim,
            device=device,
            skip_llm=flp_skip_llm,
            cached_hidden_size=flp_cached_hidden_size,
            chat_template_prompt=flp_chat_template_prompt,
        )
        mode = "cached" if flp_skip_llm else ("frozen" if flp_freeze_llm else "trainable")
        logger.info(f"Created Frozen LLM Pooler extractor: {flp_model_name} "
                    f"({mode}), max_length={flp_max_length}, "
                    f"projection_dim={flp_projection_dim}")
        return extractor

    elif normalized_type == "hierarchical_llm":
        from .hierarchical_llm_extractor import HierarchicalLLMExtractor
        extractor = HierarchicalLLMExtractor(
            model_name=hlm_model_name,
            chunk_size=hlm_chunk_size,
            chunk_overlap=hlm_chunk_overlap,
            max_chunks=hlm_max_chunks,
            freeze_llm=hlm_freeze_llm,
            gated_attention_dim=hlm_gated_attention_dim,
            projection_dim=hlm_projection_dim,
            dropout=hlm_dropout,
            gradient_checkpointing=hlm_gradient_checkpointing,
            downprojection_dim=hlm_downprojection_dim,
            device=device,
            skip_llm=hlm_skip_llm,
            cached_hidden_size=hlm_cached_hidden_size,
            chat_template_prompt=hlm_chat_template_prompt,
        )
        mode = "cached" if hlm_skip_llm else ("frozen" if hlm_freeze_llm else "trainable")
        logger.info(f"Created Hierarchical LLM extractor: {hlm_model_name} "
                    f"({mode}), chunk_size={hlm_chunk_size}, max_chunks={hlm_max_chunks}, "
                    f"projection_dim={hlm_projection_dim}")
        return extractor

    elif normalized_type == "hierarchical_transformer":
        from .hierarchical_transformer_extractor import HierarchicalTransformerExtractor
        extractor = HierarchicalTransformerExtractor(
            sentence_encoder_model=htr_sentence_model,
            freeze_sentence_encoder=htr_freeze_sentence_encoder,
            chunk_size_words=htr_chunk_size_words,
            chunk_overlap_words=htr_chunk_overlap_words,
            max_chunks=htr_max_chunks,
            max_chunk_length=htr_max_chunk_length,
            num_transformer_layers=htr_num_layers,
            num_attention_heads=htr_num_heads,
            transformer_dim=htr_transformer_dim,
            transformer_dropout=htr_dropout,
            projection_dim=htr_projection_dim,
            hash_embedding_dim=htr_hash_embedding_dim,
            sentence_encoder_batch_size=htr_sentence_encoder_batch_size,
            sentence_encoder_backend=htr_sentence_encoder_backend,
            sentence_pooling=htr_sentence_pooling,
            normalize_sentence_embeddings=htr_normalize_sentence_embeddings,
            trainable_sentence_encoder_layers=htr_trainable_sentence_encoder_layers,
            device=device,
        )
        logger.info(
            "Created Hierarchical Transformer extractor: model=%s, chunks=%d/%d/%d, "
            "projection_dim=%d, sentence_encoder_batch_size=%d, backend=%s",
            htr_sentence_model,
            htr_chunk_size_words,
            htr_chunk_overlap_words,
            htr_max_chunks,
            htr_projection_dim,
            htr_sentence_encoder_batch_size,
            htr_sentence_encoder_backend,
        )
        return extractor

    elif normalized_type == "hierarchical_cnn":
        from .hierarchical_cnn_extractor import HierarchicalCNNExtractor
        extractor = HierarchicalCNNExtractor(
            embedding_dim=hcnn_embedding_dim,
            conv_dim=hcnn_conv_dim,
            kernel_size=hcnn_kernel_size,
            num_conv_blocks=hcnn_num_conv_blocks,
            chunk_size=hcnn_chunk_size,
            chunk_overlap=hcnn_chunk_overlap,
            max_chunks=hcnn_max_chunks,
            vocab_size=hcnn_vocab_size,
            gated_attention_dim=hcnn_gated_attention_dim,
            projection_dim=hcnn_projection_dim,
            dropout=hcnn_dropout,
            device=device,
        )
        logger.info(f"Created Hierarchical CNN extractor: "
                    f"conv_dim={hcnn_conv_dim}, num_blocks={hcnn_num_conv_blocks}, "
                    f"chunk_size={hcnn_chunk_size}, max_chunks={hcnn_max_chunks}, "
                    f"projection_dim={hcnn_projection_dim}")
        return extractor

    elif normalized_type == "hierarchical_gru":
        from .hierarchical_gru_extractor import HierarchicalGRUExtractor
        extractor = HierarchicalGRUExtractor(
            embedding_dim=hgru_embedding_dim,
            gru_hidden_dim=hgru_gru_hidden_dim,
            num_gru_layers=hgru_num_gru_layers,
            chunk_size=hgru_chunk_size,
            chunk_overlap=hgru_chunk_overlap,
            max_chunks=hgru_max_chunks,
            vocab_size=hgru_vocab_size,
            gated_attention_dim=hgru_gated_attention_dim,
            projection_dim=hgru_projection_dim,
            dropout=hgru_dropout,
            device=device,
        )
        logger.info(f"Created Hierarchical GRU extractor: "
                    f"gru_hidden_dim={hgru_gru_hidden_dim}, num_layers={hgru_num_gru_layers}, "
                    f"chunk_size={hgru_chunk_size}, max_chunks={hgru_max_chunks}, "
                    f"projection_dim={hgru_projection_dim}")
        return extractor

    elif normalized_type == "simple_cnn":
        from .simple_cnn_extractor import SimpleCNNExtractor
        extractor = SimpleCNNExtractor(
            embedding_dim=scnn_embedding_dim,
            conv_dim=scnn_conv_dim,
            kernel_size=scnn_kernel_size,
            num_conv_blocks=scnn_num_conv_blocks,
            max_length=scnn_max_length,
            vocab_size=scnn_vocab_size,
            gated_attention_dim=scnn_gated_attention_dim,
            projection_dim=scnn_projection_dim,
            dropout=scnn_dropout,
            device=device,
        )
        logger.info(f"Created Simple CNN extractor: "
                    f"conv_dim={scnn_conv_dim}, num_blocks={scnn_num_conv_blocks}, "
                    f"max_length={scnn_max_length}, projection_dim={scnn_projection_dim}")
        return extractor

    elif normalized_type == "concept_embedding_cnn":
        from .concept_embedding_cnn_extractor import ConceptEmbeddingCNNExtractor
        extractor = ConceptEmbeddingCNNExtractor(
            sentence_model_name=cecnn_sentence_model_name,
            chunk_size_words=cecnn_chunk_size_words,
            chunk_overlap_words=cecnn_chunk_overlap_words,
            max_chunks=cecnn_max_chunks,
            confounder_concepts=cecnn_confounder_concepts or [],
            effect_modifier_concepts=cecnn_effect_modifier_concepts or [],
            random_features=cecnn_random_features,
            random_confounder_features=cecnn_random_confounder_features,
            random_modifier_features=cecnn_random_modifier_features,
            kernel_role=cecnn_kernel_role,
            projection_dim=cecnn_projection_dim,
            dropout=cecnn_dropout,
            anchor_weight=cecnn_anchor_weight,
            cached_embedding_dim=cecnn_cached_embedding_dim,
            normalize_embeddings=cecnn_normalize_embeddings,
            random_state=cecnn_random_state,
            device=device,
        )
        logger.info(
            "Created Concept Embedding CNN extractor: role=%s, chunks=%d/%d/%d, "
            "projection_dim=%d",
            cecnn_kernel_role,
            cecnn_chunk_size_words,
            cecnn_chunk_overlap_words,
            cecnn_max_chunks,
            cecnn_projection_dim,
        )
        return extractor

    elif normalized_type == "concept_token_cnn":
        from .concept_token_cnn_extractor import ConceptTokenCNNExtractor
        extractor = ConceptTokenCNNExtractor(
            model_name=ctcnn_model_name,
            chunk_size=ctcnn_chunk_size,
            chunk_overlap=ctcnn_chunk_overlap,
            max_chunks=ctcnn_max_chunks,
            confounder_concepts=ctcnn_confounder_concepts or [],
            effect_modifier_concepts=ctcnn_effect_modifier_concepts or [],
            random_features=ctcnn_random_features,
            random_confounder_features=ctcnn_random_confounder_features,
            random_modifier_features=ctcnn_random_modifier_features,
            kernel_role=ctcnn_kernel_role,
            projection_dim=ctcnn_projection_dim,
            dropout=ctcnn_dropout,
            anchor_weight=ctcnn_anchor_weight,
            cached_hidden_size=ctcnn_cached_hidden_size,
            downprojection_dim=ctcnn_downprojection_dim,
            normalize_embeddings=ctcnn_normalize_embeddings,
            random_state=ctcnn_random_state,
            device=device,
        )
        logger.info(
            "Created Concept Token CNN extractor: role=%s, chunks=%d/%d/%d, "
            "projection_dim=%d",
            ctcnn_kernel_role,
            ctcnn_chunk_size,
            ctcnn_chunk_overlap,
            ctcnn_max_chunks,
            ctcnn_projection_dim,
        )
        return extractor

    elif normalized_type == "slot_value_discovery":
        from .slot_value_discovery_extractor import SlotValueDiscoveryExtractor
        extractor = SlotValueDiscoveryExtractor(
            sentence_model_name=svx_sentence_model_name,
            chunk_size_words=svx_chunk_size_words,
            chunk_overlap_words=svx_chunk_overlap_words,
            max_chunks=svx_max_chunks,
            confounder_concepts=svx_confounder_concepts or [],
            effect_modifier_concepts=svx_effect_modifier_concepts or [],
            num_free_slots=svx_num_free_slots,
            slot_dim=svx_slot_dim,
            num_value_prototypes=svx_num_value_prototypes,
            dropout=svx_dropout,
            anchor_weight=svx_anchor_weight,
            cached_embedding_dim=svx_cached_embedding_dim,
            normalize_embeddings=svx_normalize_embeddings,
            attention_temperature=svx_attention_temperature,
            attention_entropy_weight=svx_attention_entropy_weight,
            query_diversity_weight=svx_query_diversity_weight,
            random_state=svx_random_state,
            device=device,
        )
        logger.info(
            "Created Slot-Value Discovery extractor: chunks=%d/%d/%d, "
            "seed_slots=%d, free_slots=%d, slot_dim=%d",
            svx_chunk_size_words,
            svx_chunk_overlap_words,
            svx_max_chunks,
            len((svx_confounder_concepts or []) + (svx_effect_modifier_concepts or [])),
            svx_num_free_slots,
            svx_slot_dim,
        )
        return extractor

    else:
        from ..config import VALID_EXTRACTOR_TYPES
        raise ValueError(
            f"Unsupported feature extractor type: '{extractor_type}'. "
            f"Supported types: {sorted(VALID_EXTRACTOR_TYPES)}"
        )


def create_feature_extractor_from_config(
    config: Dict[str, Any],
    device: torch.device,
    model_type: str = "dragonnet"
) -> nn.Module:
    """
    Create a feature extractor from a configuration dictionary.

    Args:
        config: Configuration dictionary (typically from CausalText.config)
        device: PyTorch device
        model_type: Model type for task-specific extractors

    Returns:
        nn.Module: The instantiated feature extractor
    """
    return create_feature_extractor(
        extractor_type=config.get('feature_extractor_type', 'frozen_llm_pooler'),
        device=device,
        model_type=model_type,
        # Frozen LLM Pooler args
        flp_model_name=config.get('flp_model_name', 'Qwen/Qwen3-0.6B-Base'),
        flp_max_length=config.get('flp_max_length', 8192),
        flp_freeze_llm=config.get('flp_freeze_llm', True),
        flp_gated_attention_dim=config.get('flp_gated_attention_dim', 128),
        flp_projection_dim=config.get('flp_projection_dim', 128),
        flp_dropout=config.get('flp_dropout', 0.1),
        flp_gradient_checkpointing=config.get('flp_gradient_checkpointing', True),
        flp_downprojection_dim=config.get('flp_downprojection_dim', None),
        flp_skip_llm=config.get('flp_skip_llm', False),
        flp_cached_hidden_size=config.get('flp_cached_hidden_size', 0),
        flp_chat_template_prompt=config.get('flp_chat_template_prompt', None),
        # Hierarchical LLM args
        hlm_model_name=config.get('hlm_model_name', 'Qwen/Qwen3-0.6B-Base'),
        hlm_chunk_size=config.get('hlm_chunk_size', 2048),
        hlm_chunk_overlap=config.get('hlm_chunk_overlap', 256),
        hlm_max_chunks=config.get('hlm_max_chunks', 16),
        hlm_freeze_llm=config.get('hlm_freeze_llm', True),
        hlm_gated_attention_dim=config.get('hlm_gated_attention_dim', 128),
        hlm_projection_dim=config.get('hlm_projection_dim', 128),
        hlm_dropout=config.get('hlm_dropout', 0.1),
        hlm_gradient_checkpointing=config.get('hlm_gradient_checkpointing', True),
        hlm_downprojection_dim=config.get('hlm_downprojection_dim', None),
        hlm_skip_llm=config.get('hlm_skip_llm', False),
        hlm_cached_hidden_size=config.get('hlm_cached_hidden_size', 0),
        hlm_chat_template_prompt=config.get('hlm_chat_template_prompt', None),
        # Hierarchical Transformer args
        htr_sentence_model=config.get('htr_sentence_model', 'prajjwal1/bert-tiny'),
        htr_freeze_sentence_encoder=config.get('htr_freeze_sentence_encoder', False),
        htr_chunk_size_words=config.get('htr_chunk_size_words', 96),
        htr_chunk_overlap_words=config.get('htr_chunk_overlap_words', 24),
        htr_max_chunks=config.get('htr_max_chunks', 128),
        htr_max_chunk_length=config.get('htr_max_chunk_length', 128),
        htr_num_layers=config.get('htr_num_layers', 2),
        htr_num_heads=config.get('htr_num_heads', 4),
        htr_transformer_dim=config.get('htr_transformer_dim', 256),
        htr_dropout=config.get('htr_dropout', 0.1),
        htr_projection_dim=config.get('htr_projection_dim', 128),
        htr_hash_embedding_dim=config.get('htr_hash_embedding_dim', 256),
        htr_sentence_encoder_batch_size=config.get('htr_sentence_encoder_batch_size', 128),
        htr_sentence_encoder_backend=config.get('htr_sentence_encoder_backend', 'auto'),
        htr_sentence_pooling=config.get('htr_sentence_pooling', 'auto'),
        htr_normalize_sentence_embeddings=config.get('htr_normalize_sentence_embeddings', True),
        htr_trainable_sentence_encoder_layers=config.get(
            'htr_trainable_sentence_encoder_layers',
            0,
        ),
        # Hierarchical CNN args
        hcnn_embedding_dim=config.get('hcnn_embedding_dim', 256),
        hcnn_conv_dim=config.get('hcnn_conv_dim', 256),
        hcnn_kernel_size=config.get('hcnn_kernel_size', 5),
        hcnn_num_conv_blocks=config.get('hcnn_num_conv_blocks', 4),
        hcnn_chunk_size=config.get('hcnn_chunk_size', 512),
        hcnn_chunk_overlap=config.get('hcnn_chunk_overlap', 64),
        hcnn_max_chunks=config.get('hcnn_max_chunks', 32),
        hcnn_vocab_size=config.get('hcnn_vocab_size', 50000),
        hcnn_gated_attention_dim=config.get('hcnn_gated_attention_dim', 128),
        hcnn_projection_dim=config.get('hcnn_projection_dim', 128),
        hcnn_dropout=config.get('hcnn_dropout', 0.1),
        # Hierarchical GRU args
        hgru_embedding_dim=config.get('hgru_embedding_dim', 256),
        hgru_gru_hidden_dim=config.get('hgru_gru_hidden_dim', 256),
        hgru_num_gru_layers=config.get('hgru_num_gru_layers', 2),
        hgru_chunk_size=config.get('hgru_chunk_size', 512),
        hgru_chunk_overlap=config.get('hgru_chunk_overlap', 64),
        hgru_max_chunks=config.get('hgru_max_chunks', 32),
        hgru_vocab_size=config.get('hgru_vocab_size', 50000),
        hgru_gated_attention_dim=config.get('hgru_gated_attention_dim', 128),
        hgru_projection_dim=config.get('hgru_projection_dim', 128),
        hgru_dropout=config.get('hgru_dropout', 0.1),
        # Simple CNN args
        scnn_embedding_dim=config.get('scnn_embedding_dim', 256),
        scnn_conv_dim=config.get('scnn_conv_dim', 256),
        scnn_kernel_size=config.get('scnn_kernel_size', 5),
        scnn_num_conv_blocks=config.get('scnn_num_conv_blocks', 4),
        scnn_max_length=config.get('scnn_max_length', 10000),
        scnn_vocab_size=config.get('scnn_vocab_size', 50000),
        scnn_gated_attention_dim=config.get('scnn_gated_attention_dim', 128),
        scnn_projection_dim=config.get('scnn_projection_dim', 128),
        scnn_dropout=config.get('scnn_dropout', 0.1),
        # Concept embedding CNN args
        cecnn_sentence_model_name=config.get(
            'cecnn_sentence_model_name',
            'sentence-transformers/all-MiniLM-L6-v2',
        ),
        cecnn_chunk_size_words=config.get('cecnn_chunk_size_words', 64),
        cecnn_chunk_overlap_words=config.get('cecnn_chunk_overlap_words', 16),
        cecnn_max_chunks=config.get('cecnn_max_chunks', 128),
        cecnn_confounder_concepts=config.get('cecnn_confounder_concepts', []),
        cecnn_effect_modifier_concepts=config.get('cecnn_effect_modifier_concepts', []),
        cecnn_random_features=config.get('cecnn_random_features', 0),
        cecnn_random_confounder_features=config.get(
            'cecnn_random_confounder_features', None
        ),
        cecnn_random_modifier_features=config.get(
            'cecnn_random_modifier_features', None
        ),
        cecnn_kernel_role=config.get('cecnn_kernel_role', 'combined'),
        cecnn_projection_dim=config.get('cecnn_projection_dim', 128),
        cecnn_dropout=config.get('cecnn_dropout', 0.1),
        cecnn_anchor_weight=config.get('cecnn_anchor_weight', 0.01),
        cecnn_cached_embedding_dim=config.get('cecnn_cached_embedding_dim', 0),
        cecnn_normalize_embeddings=config.get('cecnn_normalize_embeddings', True),
        cecnn_random_state=config.get('cecnn_random_state', 42),
        # Concept token CNN args
        ctcnn_model_name=config.get('ctcnn_model_name', 'Qwen/Qwen3-0.6B-Base'),
        ctcnn_chunk_size=config.get('ctcnn_chunk_size', 2048),
        ctcnn_chunk_overlap=config.get('ctcnn_chunk_overlap', 256),
        ctcnn_max_chunks=config.get('ctcnn_max_chunks', 16),
        ctcnn_confounder_concepts=config.get('ctcnn_confounder_concepts', []),
        ctcnn_effect_modifier_concepts=config.get('ctcnn_effect_modifier_concepts', []),
        ctcnn_random_features=config.get('ctcnn_random_features', 0),
        ctcnn_random_confounder_features=config.get(
            'ctcnn_random_confounder_features', None
        ),
        ctcnn_random_modifier_features=config.get(
            'ctcnn_random_modifier_features', None
        ),
        ctcnn_kernel_role=config.get('ctcnn_kernel_role', 'combined'),
        ctcnn_projection_dim=config.get('ctcnn_projection_dim', 128),
        ctcnn_dropout=config.get('ctcnn_dropout', 0.1),
        ctcnn_anchor_weight=config.get('ctcnn_anchor_weight', 0.01),
        ctcnn_cached_hidden_size=config.get('ctcnn_cached_hidden_size', 0),
        ctcnn_downprojection_dim=config.get('ctcnn_downprojection_dim', None),
        ctcnn_normalize_embeddings=config.get('ctcnn_normalize_embeddings', True),
        ctcnn_random_state=config.get('ctcnn_random_state', 42),
        # Slot-value discovery args
        svx_sentence_model_name=config.get(
            'svx_sentence_model_name',
            'sentence-transformers/all-MiniLM-L6-v2',
        ),
        svx_chunk_size_words=config.get('svx_chunk_size_words', 64),
        svx_chunk_overlap_words=config.get('svx_chunk_overlap_words', 16),
        svx_max_chunks=config.get('svx_max_chunks', 128),
        svx_confounder_concepts=config.get('svx_confounder_concepts', []),
        svx_effect_modifier_concepts=config.get('svx_effect_modifier_concepts', []),
        svx_num_free_slots=config.get('svx_num_free_slots', 16),
        svx_slot_dim=config.get('svx_slot_dim', 128),
        svx_num_value_prototypes=config.get('svx_num_value_prototypes', 4),
        svx_dropout=config.get('svx_dropout', 0.1),
        svx_anchor_weight=config.get('svx_anchor_weight', 0.01),
        svx_cached_embedding_dim=config.get('svx_cached_embedding_dim', 0),
        svx_normalize_embeddings=config.get('svx_normalize_embeddings', True),
        svx_attention_temperature=config.get('svx_attention_temperature', 0.1),
        svx_attention_entropy_weight=config.get('svx_attention_entropy_weight', 0.0),
        svx_query_diversity_weight=config.get('svx_query_diversity_weight', 0.0),
        svx_random_state=config.get('svx_random_state', 42),
    )
