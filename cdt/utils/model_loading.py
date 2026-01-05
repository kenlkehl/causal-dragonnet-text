# cdt/utils/model_loading.py
"""Utilities for loading pretrained models."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch


logger = logging.getLogger(__name__)


def extract_feature_extractor_config(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract feature extractor configuration from checkpoint.

    Args:
        checkpoint: Model checkpoint dictionary

    Returns:
        Dictionary with feature extractor configuration
    """
    config = checkpoint.get('config', {})

    # Extract key dimensions
    fe_config = {
        'embedding_dim': None,
        'num_latent_confounders': config.get('num_latent_confounders', 0),
        'num_explicit_confounders': 0,
        'value_dim': config.get('value_dim', 128),
        'num_attention_heads': config.get('num_attention_heads', 4),
        'projection_dim': config.get('dragonnet_representation_dim', 128),
    }

    # Try to infer from state dict if config not available
    if 'feature_extractor' in checkpoint:
        fe_state = checkpoint['feature_extractor']

        # Get embedding dimension from latent confounders
        if 'latent_confounders' in fe_state:
            fe_config['embedding_dim'] = fe_state['latent_confounders'].shape[1]
            fe_config['num_latent_confounders'] = fe_state['latent_confounders'].shape[0]

        # Get explicit confounders
        if 'explicit_confounders' in fe_state:
            fe_config['num_explicit_confounders'] = fe_state['explicit_confounders'].shape[0]

    # Calculate total confounders
    fe_config['num_total_confounders'] = (
        fe_config['num_latent_confounders'] +
        fe_config['num_explicit_confounders']
    )

    return fe_config


def load_pretrained_with_dimension_matching(
    current_model,
    pretrained_checkpoint: Dict[str, Any],
    strict: bool = False,
    auto_adjust: bool = True
) -> Dict[str, Any]:
    """
    Load pretrained weights with automatic dimension matching.

    Note: With the cross-attention architecture, dimension matching is simplified.
    The output dimension is now projection_dim (= dragonnet_representation_dim),
    not num_confounders * features_per_confounder.

    Args:
        current_model: Current model to load weights into (CausalDragonnetText)
        pretrained_checkpoint: Pretrained checkpoint dictionary
        strict: Whether to require exact dimension match
        auto_adjust: Whether to automatically adjust current model for dimension mismatch

    Returns:
        Dictionary with loading information and results
    """
    result = {
        'loaded': False,
        'adjustments_made': [],
        'missing_keys': [],
        'unexpected_keys': []
    }

    # Extract configurations
    pretrained_config = extract_feature_extractor_config(pretrained_checkpoint)
    current_fe = current_model.feature_extractor

    # Check basic compatibility
    if pretrained_config['embedding_dim'] is not None:
        if pretrained_config['embedding_dim'] != current_fe.embedding_dim:
            logger.error(f"Embedding dimension mismatch: pretrained={pretrained_config['embedding_dim']}, "
                        f"current={current_fe.embedding_dim}")
            return result

    # Try to load feature extractor weights
    if 'feature_extractor' in pretrained_checkpoint:
        try:
            # Load with strict=False to handle missing/extra keys
            missing, unexpected = current_fe.load_state_dict(
                pretrained_checkpoint['feature_extractor'],
                strict=False
            )
            result['missing_keys'] = list(missing) if missing else []
            result['unexpected_keys'] = list(unexpected) if unexpected else []

            if missing:
                logger.warning(f"Missing keys when loading feature_extractor: {missing}")
            if unexpected:
                logger.info(f"Unexpected keys in checkpoint (ignored): {unexpected}")

            result['loaded'] = True
            logger.info("Loaded feature_extractor weights (non-strict)")

        except Exception as e:
            logger.warning(f"Failed to load feature_extractor weights: {e}")

    # Try to load DragonNet representation layers
    if hasattr(current_model, 'net') and 'dragonnet' in pretrained_checkpoint:
        try:
            dragonnet_loaded = current_model.net.load_pretrained_representation(
                pretrained_checkpoint
            )
            if dragonnet_loaded:
                logger.info("Loaded DragonNet representation layers")
                result['adjustments_made'].append('dragonnet_representation_loaded')
        except Exception as e:
            logger.warning(f"Failed to load DragonNet representation: {e}")

    return result


def create_compatible_model_from_checkpoint(
    checkpoint_path: Path,
    model_class,
    override_config: Optional[Dict[str, Any]] = None,
    device: str = 'cuda:0'
):
    """
    Create a model that's compatible with a pretrained checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model_class: Model class (CausalDragonnetText)
        override_config: Optional config parameters to override
        device: Device string

    Returns:
        Model instance with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    # Apply overrides
    if override_config:
        config.update(override_config)

    # Set device
    config['device'] = device

    # Create model from config
    model = model_class(**config)

    # Load weights
    load_result = load_pretrained_with_dimension_matching(
        model, checkpoint, strict=False, auto_adjust=True
    )

    if not load_result['loaded']:
        logger.warning("Failed to load weights - model has random initialization")

    return model
