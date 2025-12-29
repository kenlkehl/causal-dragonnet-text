# cdt/utils/diagnostics.py
"""Diagnostic utilities for confounder analysis."""

import logging
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def compute_confounder_feature_stats(
    features: np.ndarray,
    num_confounders: int,
    features_per_confounder: int = 1,
    explicit_confounder_texts: Optional[List[str]] = None,
    num_latent: int = 0
) -> pd.DataFrame:
    """
    Compute statistics for each confounder feature.
    
    Args:
        features: Array of shape (n_samples, n_features) where 
                  n_features = (num_confounders * features_per_confounder)
        num_confounders: Total number of confounders (explicit + latent)
        features_per_confounder: Number of feature dimensions per confounder
        explicit_confounder_texts: Optional list of explicit confounder query texts
        num_latent: Number of latent confounders
    
    Returns:
        DataFrame with per-feature statistics
    """
    n_samples, n_features = features.shape
    expected_features = num_confounders * features_per_confounder
    
    if n_features != expected_features:
        logger.warning(f"Feature dimension mismatch: got {n_features}, expected {expected_features}")
    
    # Build confounder names
    num_explicit = len(explicit_confounder_texts) if explicit_confounder_texts else 0
    
    records = []
    for feat_idx in range(n_features):
        confounder_idx = feat_idx // features_per_confounder
        sub_feature_idx = feat_idx % features_per_confounder
        
        # Determine confounder name
        if confounder_idx < num_explicit and explicit_confounder_texts:
            # Truncate long explicit confounder texts
            text = explicit_confounder_texts[confounder_idx]
            confounder_name = f"explicit_{confounder_idx}: {text[:50]}..."
        else:
            latent_idx = confounder_idx - num_explicit
            confounder_name = f"latent_{latent_idx}"
        
        # Compute stats
        feat_values = features[:, feat_idx]
        records.append({
            'confounder_idx': confounder_idx,
            'sub_feature_idx': sub_feature_idx,
            'confounder_name': confounder_name,
            'mean': np.mean(feat_values),
            'std': np.std(feat_values),
            'min': np.min(feat_values),
            'max': np.max(feat_values),
            'range': np.max(feat_values) - np.min(feat_values)
        })
    
    return pd.DataFrame(records)


def compute_latent_drift(
    initial_latents: torch.Tensor,
    final_latents: torch.Tensor
) -> pd.DataFrame:
    """
    Compute cosine similarity between initial and final latent confounder vectors.
    
    A value of 1.0 means no drift (vectors are identical after training).
    Lower values indicate the latents have been adapted to the dataset.
    
    Args:
        initial_latents: Tensor of shape (num_latent, embedding_dim)
        final_latents: Tensor of shape (num_latent, embedding_dim)
    
    Returns:
        DataFrame with per-latent cosine similarity
    """
    if initial_latents is None or final_latents is None:
        return pd.DataFrame()
    
    with torch.no_grad():
        # Normalize both
        initial_norm = F.normalize(initial_latents, p=2, dim=1)
        final_norm = F.normalize(final_latents, p=2, dim=1)
        
        # Compute per-latent cosine similarity
        cos_sim = (initial_norm * final_norm).sum(dim=1).cpu().numpy()
    
    records = []
    for idx, sim in enumerate(cos_sim):
        records.append({
            'latent_idx': idx,
            'cosine_similarity': sim,
            'drift': 1.0 - sim  # Higher = more drift
        })
    
    return pd.DataFrame(records)


def log_confounder_stats(stats_df: pd.DataFrame, prefix: str = "") -> None:
    """Log confounder feature statistics in a readable format."""
    if stats_df.empty:
        return
    
    logger.info(f"{prefix}Confounder Feature Statistics (n={len(stats_df)} features):")
    
    # Summary by confounder
    summary = stats_df.groupby('confounder_idx').agg({
        'mean': 'mean',
        'std': 'mean', 
        'range': 'mean',
        'confounder_name': 'first'
    }).reset_index()
    
    for _, row in summary.iterrows():
        logger.info(
            f"  [{row['confounder_idx']:2d}] {row['confounder_name'][:60]:60s} | "
            f"mean={row['mean']:+.3f} std={row['std']:.3f} range={row['range']:.3f}"
        )


def log_latent_drift(drift_df: pd.DataFrame, prefix: str = "") -> None:
    """Log latent drift statistics."""
    if drift_df.empty:
        logger.info(f"{prefix}Latent Drift: No latent confounders to track")
        return
    
    mean_sim = drift_df['cosine_similarity'].mean()
    min_sim = drift_df['cosine_similarity'].min()
    max_sim = drift_df['cosine_similarity'].max()
    
    logger.info(f"{prefix}Latent Drift Summary:")
    logger.info(f"  Mean cosine similarity: {mean_sim:.4f} (1.0 = no change)")
    logger.info(f"  Range: [{min_sim:.4f}, {max_sim:.4f}]")
    
    # Log individual latents with significant drift (similarity < 0.95)
    drifted = drift_df[drift_df['cosine_similarity'] < 0.95]
    if len(drifted) > 0:
        logger.info(f"  Latents with significant drift (sim < 0.95):")
        for _, row in drifted.iterrows():
            logger.info(f"    latent_{row['latent_idx']}: cos_sim={row['cosine_similarity']:.4f}")


def extract_confounder_features_batch(
    model,
    loader,
    device: torch.device
) -> np.ndarray:
    """
    Extract confounder features for all samples in a DataLoader.
    
    Args:
        model: CausalDragonnetText model
        loader: DataLoader
        device: torch device
    
    Returns:
        Array of shape (n_samples, n_features)
    """
    all_features = []
    model.eval()
    
    with torch.no_grad():
        for batch in loader:
            chunk_embeddings_list = [
                batch['chunk_embeddings'][i, :, :].to(device).contiguous()
                for i in range(batch['chunk_embeddings'].size(0))
            ]
            features = model.feature_extractor(chunk_embeddings_list)
            all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)
