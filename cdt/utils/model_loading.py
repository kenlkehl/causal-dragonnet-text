# cdt/utils/model_loading.py
"""Utilities for loading pretrained models with dimension matching."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn


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
        'features_per_confounder': config.get('features_per_confounder', 1),
        'aggregator_mode': config.get('aggregator_mode', 'attn'),
    }
    
    # Try to infer from state dict if config not available
    if 'feature_extractor' in checkpoint:
        fe_state = checkpoint['feature_extractor']
        
        # Get embedding dimension
        if 'latent_confounders' in fe_state:
            fe_config['embedding_dim'] = fe_state['latent_confounders'].shape[1]
            fe_config['num_latent_confounders'] = fe_state['latent_confounders'].shape[0]
        
        # Get explicit confounders
        if 'explicit_confounders' in fe_state:
            fe_config['num_explicit_confounders'] = fe_state['explicit_confounders'].shape[0]
        
        # Get features per confounder from projection shape if present
        if 'confounder_projection' in fe_state:
            projection_shape = fe_state['confounder_projection'].shape
            if len(projection_shape) == 3:
                fe_config['features_per_confounder'] = projection_shape[1]
    
    # Calculate total confounders and output dimension
    fe_config['num_total_confounders'] = (
        fe_config['num_latent_confounders'] + 
        fe_config['num_explicit_confounders']
    )
    
    # Get aggregator features per confounder
    if 'aggregator' in checkpoint.get('feature_extractor', {}):
        agg_state = checkpoint['feature_extractor']['aggregator']
        if hasattr(agg_state, 'features_per_conf'):
            fe_config['aggregator_features_per_conf'] = agg_state.features_per_conf
        else:
            fe_config['aggregator_features_per_conf'] = 1
    else:
        # Infer from mode
        fe_config['aggregator_features_per_conf'] = 2 if fe_config['aggregator_mode'] == 'stats' else 1
    
    fe_config['total_output_dim'] = (
        fe_config['num_total_confounders'] * 
        fe_config['features_per_confounder'] * 
        fe_config['aggregator_features_per_conf']
    )
    
    return fe_config


def compute_dimension_mismatch(
    pretrained_config: Dict[str, Any],
    current_config: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compute dimension mismatch between pretrained and current model.
    
    Args:
        pretrained_config: Config from pretrained model
        current_config: Config from current model
    
    Returns:
        Tuple of (has_mismatch, mismatch_info)
    """
    mismatch_info = {
        'embedding_dim_match': True,
        'total_confounders_match': True,
        'output_dim_match': True,
        'pretrained_output_dim': pretrained_config['total_output_dim'],
        'current_output_dim': current_config['total_output_dim'],
        'pretrained_confounders': pretrained_config['num_total_confounders'],
        'current_confounders': current_config['num_total_confounders'],
        'required_phantom_confounders': 0,
        'can_load': True,
        'load_strategy': 'direct'
    }
    
    # Check embedding dimension (must match)
    if pretrained_config['embedding_dim'] != current_config['embedding_dim']:
        mismatch_info['embedding_dim_match'] = False
        mismatch_info['can_load'] = False
        mismatch_info['load_strategy'] = 'incompatible'
        return True, mismatch_info
    
    # Check total confounders
    if pretrained_config['num_total_confounders'] != current_config['num_total_confounders']:
        mismatch_info['total_confounders_match'] = False
    
    # Check output dimension
    if pretrained_config['total_output_dim'] != current_config['total_output_dim']:
        mismatch_info['output_dim_match'] = False
    
    # Determine loading strategy
    if pretrained_config['total_output_dim'] > current_config['total_output_dim']:
        # Pretrained model is larger - use phantom confounders
        dim_diff = pretrained_config['total_output_dim'] - current_config['total_output_dim']
        out_per_conf = current_config['features_per_confounder'] * current_config['aggregator_features_per_conf']
        
        if dim_diff % out_per_conf == 0:
            mismatch_info['required_phantom_confounders'] = dim_diff // out_per_conf
            mismatch_info['load_strategy'] = 'phantom_padding'
            mismatch_info['can_load'] = True
        else:
            mismatch_info['can_load'] = False
            mismatch_info['load_strategy'] = 'incompatible_dimensions'
    
    elif pretrained_config['total_output_dim'] < current_config['total_output_dim']:
        # Current model is larger - can load partial weights
        mismatch_info['load_strategy'] = 'partial_load'
        mismatch_info['can_load'] = True
    
    has_mismatch = not (
        mismatch_info['total_confounders_match'] and 
        mismatch_info['output_dim_match']
    )
    
    return has_mismatch, mismatch_info


def _load_feature_extractor_with_size_adjustment(
    current_fe,
    pretrained_state_dict: Dict[str, Any],
    pretrained_config: Dict[str, Any],
    current_config: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """
    Load feature extractor weights with size adjustment for mismatched parameters.
    
    This handles cases where latent_confounders or confounder_projection have
    different sizes between pretrained and current models.
    
    Args:
        current_fe: Current FeatureExtractor module
        pretrained_state_dict: State dict from pretrained model
        pretrained_config: Config extracted from pretrained checkpoint
        current_config: Config from current model
    
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    # Create a modified state dict that handles size mismatches
    adjusted_state_dict = {}
    missing_keys = []
    unexpected_keys = []
    
    current_state = current_fe.state_dict()
    
    for key, pretrained_param in pretrained_state_dict.items():
        if key not in current_state:
            unexpected_keys.append(key)
            continue
        
        current_param = current_state[key]
        
        # Handle size mismatches for specific parameters
        if key == 'latent_confounders':
            # Copy subset of pretrained latent confounders
            pretrained_latents = pretrained_config['num_latent_confounders']
            current_latents = current_config['num_latent_confounders']
            
            if pretrained_latents != current_latents:
                if current_latents < pretrained_latents:
                    # Current has fewer - take first N confounders from pretrained
                    logger.info(f"Copying {current_latents} of {pretrained_latents} pretrained latent confounders")
                    adjusted_state_dict[key] = pretrained_param[:current_latents].clone()
                else:
                    # Current has more - copy what we have, rest stays random
                    logger.info(f"Copying {pretrained_latents} pretrained latent confounders, keeping {current_latents - pretrained_latents} random")
                    adjusted_state_dict[key] = current_param.clone()
                    adjusted_state_dict[key][:pretrained_latents] = pretrained_param
                continue
        
        elif key == 'confounder_projection':
            # Handle confounder_projection size mismatches
            if pretrained_param.shape[0] != current_param.shape[0]:
                pretrained_conf = pretrained_param.shape[0]
                current_conf = current_param.shape[0]
                
                if current_conf < pretrained_conf:
                    # Take subset
                    logger.info(f"Copying {current_conf} of {pretrained_conf} confounder projections")
                    adjusted_state_dict[key] = pretrained_param[:current_conf].clone()
                else:
                    # Copy what we have, rest stays random
                    logger.info(f"Copying {pretrained_conf} confounder projections, keeping {current_conf - pretrained_conf} random")
                    adjusted_state_dict[key] = current_param.clone()
                    adjusted_state_dict[key][:pretrained_conf] = pretrained_param
                continue
        
        # For parameters with matching shapes, or other parameters, copy directly
        if pretrained_param.shape == current_param.shape:
            adjusted_state_dict[key] = pretrained_param
        else:
            # Shape mismatch for unexpected parameter - skip it
            logger.warning(f"Skipping parameter '{key}' due to shape mismatch: "
                          f"pretrained {pretrained_param.shape} vs current {current_param.shape}")
            unexpected_keys.append(key)
    
    # Check for missing keys
    for key in current_state.keys():
        if key not in adjusted_state_dict and key not in ['explicit_confounders']:
            # explicit_confounders is a buffer, not a parameter to load
            missing_keys.append(key)
    
    # Load the adjusted state dict
    current_fe.load_state_dict(adjusted_state_dict, strict=False)
    
    return missing_keys, unexpected_keys


def load_pretrained_with_dimension_matching(
    current_model,
    pretrained_checkpoint: Dict[str, Any],
    strict: bool = False,
    auto_adjust: bool = True
) -> Dict[str, Any]:
    """
    Load pretrained weights with automatic dimension matching.
    
    This function handles dimension mismatches between pretrained and current models
    by automatically adjusting the current model's feature extractor when possible.
    
    Args:
        current_model: Current model to load weights into (CausalDragonnetText)
        pretrained_checkpoint: Pretrained checkpoint dictionary
        strict: Whether to require exact dimension match
        auto_adjust: Whether to automatically adjust current model for dimension mismatch
    
    Returns:
        Dictionary with loading information and results
    
    Raises:
        ValueError: If dimensions are incompatible and cannot be resolved
    """
    # Extract configurations
    pretrained_config = extract_feature_extractor_config(pretrained_checkpoint)
    
    current_fe = current_model.feature_extractor
    current_config = {
        'embedding_dim': current_fe.embedding_dim,
        'num_latent_confounders': current_fe.num_latent,
        'num_explicit_confounders': current_fe.num_explicit,
        'num_total_confounders': current_fe.num_total_confounders,
        'features_per_confounder': current_fe.features_per_confounder,
        'aggregator_mode': current_fe.aggregator.mode,
        'aggregator_features_per_conf': getattr(current_fe.aggregator, 'features_per_conf', 1),
        'total_output_dim': current_fe.output_dim
    }
    
    # Compute dimension mismatch
    has_mismatch, mismatch_info = compute_dimension_mismatch(
        pretrained_config, current_config
    )
    
    result = {
        'has_mismatch': has_mismatch,
        'mismatch_info': mismatch_info,
        'loaded': False,
        'adjustments_made': []
    }
    
    # Log mismatch information
    if has_mismatch:
        logger.warning("="*80)
        logger.warning("DIMENSION MISMATCH DETECTED")
        logger.warning("="*80)
        logger.warning(f"Pretrained model: {pretrained_config['num_total_confounders']} confounders, "
                      f"{pretrained_config['total_output_dim']} output dim")
        logger.warning(f"Current model: {current_config['num_total_confounders']} confounders, "
                      f"{current_config['total_output_dim']} output dim")
        logger.warning(f"Load strategy: {mismatch_info['load_strategy']}")
    
    # Handle incompatible dimensions
    if not mismatch_info['can_load']:
        error_msg = f"Cannot load pretrained weights: {mismatch_info['load_strategy']}"
        if strict:
            raise ValueError(error_msg)
        else:
            logger.error(error_msg)
            logger.error("Skipping pretrained weights - model will use random initialization")
            return result
    
    # Apply dimension matching strategy
    if mismatch_info['load_strategy'] == 'direct':
        # No mismatch - direct load
        logger.info("No dimension mismatch - loading weights directly")
        missing_keys, unexpected_keys = current_model.feature_extractor.load_state_dict(
            pretrained_checkpoint['feature_extractor'],
            strict=False
        )
        result['loaded'] = True
        result['missing_keys'] = missing_keys
        result['unexpected_keys'] = unexpected_keys
    
    elif mismatch_info['load_strategy'] == 'phantom_padding' and auto_adjust:
        # Add phantom confounders to match dimensions
        phantom_count = mismatch_info['required_phantom_confounders']
        logger.info(f"Adding {phantom_count} phantom confounders to match pretrained model dimensions")
        
        # CRITICAL FIX: Set phantom_confounders BEFORE reinitializing DragonNet
        current_fe.phantom_confounders = phantom_count
        result['adjustments_made'].append(f'phantom_confounders={phantom_count}')
        
        # Get the NEW output dimension after adding phantom confounders
        new_output_dim = current_fe.output_dim
        logger.info(f"Feature extractor output_dim updated: {current_config['total_output_dim']} -> {new_output_dim}")
        
        # CRITICAL FIX: Reinitialize DragonNet with correct input dimension
        _reinitialize_dragonnet(current_model, new_output_dim)
        result['adjustments_made'].append(f'dragonnet_reinitialized_with_input_dim={new_output_dim}')
        
        # Load feature extractor weights with size adjustment for mismatched parameters
        missing_keys, unexpected_keys = _load_feature_extractor_with_size_adjustment(
            current_model.feature_extractor,
            pretrained_checkpoint['feature_extractor'],
            pretrained_config,
            current_config
        )
        result['loaded'] = True
        result['missing_keys'] = missing_keys
        result['unexpected_keys'] = unexpected_keys
        
        # Try to load DragonNet representation layers
        if hasattr(current_model, 'dragonnet'):
            dragonnet_loaded = current_model.dragonnet.load_pretrained_representation(
                pretrained_checkpoint
            )
            if dragonnet_loaded:
                logger.info("Successfully loaded pretrained DragonNet representation layers")
                result['adjustments_made'].append('dragonnet_representation_loaded')
            else:
                logger.warning("Could not load DragonNet representation layers (dimension mismatch)")
    
    elif mismatch_info['load_strategy'] == 'partial_load':
        # Current model is larger - load what we can with size adjustment
        logger.info("Loading partial weights (current model is larger than pretrained)")
        missing_keys, unexpected_keys = _load_feature_extractor_with_size_adjustment(
            current_model.feature_extractor,
            pretrained_checkpoint['feature_extractor'],
            pretrained_config,
            current_config
        )
        result['loaded'] = True
        result['missing_keys'] = missing_keys
        result['unexpected_keys'] = unexpected_keys
        result['adjustments_made'].append('partial_load')
    
    # Verify loading was successful
    if result['loaded']:
        logger.info("="*80)
        logger.info("✓ Pretrained weights loaded successfully")
        if result['adjustments_made']:
            logger.info(f"  Adjustments: {', '.join(result['adjustments_made'])}")
        logger.info("="*80)
    
    return result


def _reinitialize_dragonnet(model, new_input_dim: int) -> None:
    """
    Reinitialize the DragonNet module with a new input dimension.
    
    This is necessary when phantom_confounders are added to the feature extractor,
    which changes its output dimension.
    
    Args:
        model: The CausalDragonnetText or MultiTreatmentDragonnetText model
        new_input_dim: New input dimension for DragonNet
    """
    logger.info(f"Reinitializing DragonNet with input_dim={new_input_dim}")
    
    # Determine model type and reinitialize appropriately
    if hasattr(model, 'dragonnet'):
        old_dragonnet = model.dragonnet
        
        # Check if it's binary or multi-treatment DragonNet
        if hasattr(old_dragonnet, 'num_treatments'):
            # Multi-treatment model
            from ..models.multitreatment import MultiTreatmentDragonNetInternal
            new_dragonnet = MultiTreatmentDragonNetInternal(
                input_dim=new_input_dim,
                num_treatments=old_dragonnet.num_treatments,
                representation_dim=old_dragonnet.representation_dim,
                hidden_outcome_dim=64  # Default value
            )
        else:
            # Binary treatment model
            from ..models.dragonnet import DragonNet
            # Get dimensions from old model
            repr_dim = old_dragonnet.representation_fc1.out_features
            hidden_outcome_dim = old_dragonnet.outcome0_fc1.out_features
            
            new_dragonnet = DragonNet(
                input_dim=new_input_dim,
                representation_dim=repr_dim,
                hidden_outcome_dim=hidden_outcome_dim
            )
        
        # Replace the old DragonNet with the new one
        model.dragonnet = new_dragonnet.to(model._device)
        logger.info(f"✓ DragonNet reinitialized (input: {new_input_dim})")
    else:
        logger.warning("Model does not have 'dragonnet' attribute - cannot reinitialize")


def create_compatible_model_from_checkpoint(
    checkpoint_path: Path,
    model_class,
    override_config: Optional[Dict[str, Any]] = None,
    device: str = 'cuda:0'
):
    """
    Create a model that's compatible with a pretrained checkpoint.
    
    Useful when you want to load a checkpoint but don't know the exact
    configuration it was trained with.
    
    Args:
        checkpoint_path: Path to checkpoint
        model_class: Model class (CausalDragonnetText or MultiTreatmentDragonnetText)
        override_config: Optional config parameters to override
        device: Device string
    
    Returns:
        Model instance with compatible dimensions
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = extract_feature_extractor_config(checkpoint)
    
    # Apply overrides
    if override_config:
        config.update(override_config)
    
    # Determine model type from checkpoint
    if 'num_treatments' in checkpoint.get('config', {}):
        from ..models.multitreatment import MultiTreatmentDragonnetText
        model = MultiTreatmentDragonnetText(
            num_treatments=checkpoint['config']['num_treatments'],
            sentence_transformer_model_name=config.get('sentence_transformer_model_name', 'all-MiniLM-L6-v2'),
            num_latent_confounders=config['num_latent_confounders'],
            features_per_confounder=config['features_per_confounder'],
            device=device
        )
    else:
        from ..models.causal_dragonnet import CausalDragonnetText
        model = CausalDragonnetText(
            sentence_transformer_model_name=config.get('sentence_transformer_model_name', 'all-MiniLM-L6-v2'),
            num_latent_confounders=config['num_latent_confounders'],
            features_per_confounder=config['features_per_confounder'],
            device=device
        )
    
    # Load weights
    load_result = load_pretrained_with_dimension_matching(
        model, checkpoint, strict=False, auto_adjust=True
    )
    
    if not load_result['loaded']:
        logger.warning("Failed to load weights - model has random initialization")
    
    return model


def check_checkpoint_compatibility(
    checkpoint_path: Path,
    target_num_confounders: int,
    target_features_per_confounder: int = 1
) -> Dict[str, Any]:
    """
    Check if a checkpoint is compatible with target dimensions.
    
    Args:
        checkpoint_path: Path to checkpoint
        target_num_confounders: Desired number of confounders
        target_features_per_confounder: Desired features per confounder
    
    Returns:
        Dictionary with compatibility information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pretrained_config = extract_feature_extractor_config(checkpoint)
    
    target_config = {
        'embedding_dim': pretrained_config['embedding_dim'],  # Must match
        'num_total_confounders': target_num_confounders,
        'features_per_confounder': target_features_per_confounder,
        'aggregator_features_per_conf': pretrained_config['aggregator_features_per_conf'],
        'total_output_dim': target_num_confounders * target_features_per_confounder * pretrained_config['aggregator_features_per_conf']
    }
    
    has_mismatch, mismatch_info = compute_dimension_mismatch(
        pretrained_config, target_config
    )
    
    return {
        'compatible': mismatch_info['can_load'],
        'pretrained_confounders': pretrained_config['num_total_confounders'],
        'target_confounders': target_num_confounders,
        'load_strategy': mismatch_info['load_strategy'],
        'requires_adjustment': has_mismatch,
        'phantom_confounders_needed': mismatch_info['required_phantom_confounders']
    }