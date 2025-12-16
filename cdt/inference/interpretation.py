# cdt/inference/interpretation.py
"""Confounder interpretation and analysis tools."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from ..models.causal_dragonnet import CausalDragonnetText
from ..models.multitreatment import MultiTreatmentDragonnetText
from ..data import ClinicalTextDataset, collate_batch, EmbeddingCache
from ..data.preprocessing import process_text


logger = logging.getLogger(__name__)


def interpret_confounders(
    model_path: Path,
    dataset: pd.DataFrame,
    text_column: str,
    cache: Optional[EmbeddingCache] = None,
    device: str = "cuda:0",
    top_k_chunks: int = 10
) -> pd.DataFrame:
    """
    Interpret learned confounders by finding top activating text chunks.
    
    Args:
        model_path: Path to trained model checkpoint
        dataset: Dataset with clinical texts
        text_column: Name of text column
        cache: Optional embedding cache
        device: Device string
        top_k_chunks: Number of top chunks to return per confounder
    
    Returns:
        DataFrame with confounder interpretations
    """
    logger.info("="*80)
    logger.info("CONFOUNDER INTERPRETATION")
    logger.info("="*80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {len(dataset)} samples")
    
    # Load model
    device_obj = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device_obj, weights_only=False)
    
    # Determine model type and load
    if 'num_treatments' in checkpoint.get('config', {}):
        # Multi-treatment model
        model = MultiTreatmentDragonnetText.load_from_checkpoint(
            str(model_path),
            device=device
        )
        logger.info("Loaded MultiTreatmentDragonnetText model")
    else:
        # Binary treatment model
        model = CausalDragonnetText.load_from_checkpoint(
            str(model_path),
            device=device
        )
        logger.info("Loaded CausalDragonnetText model")
    
    model.eval()
    
    # Get confounder information
    num_confounders = model.feature_extractor.num_total_confounders
    logger.info(f"Model has {num_confounders} confounders:")
    logger.info(f"  Latent: {model.feature_extractor.num_latent}")
    logger.info(f"  Explicit: {model.feature_extractor.num_explicit}")
    
    if model.feature_extractor.explicit_confounder_texts:
        logger.info("Explicit confounder queries:")
        for i, text in enumerate(model.feature_extractor.explicit_confounder_texts):
            logger.info(f"  {i}: {text}")
    
    # Extract confounder activations across dataset
    logger.info("\nExtracting confounder activations...")
    
    all_activations = []
    all_texts = []
    all_chunks = []
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing texts"):
        text = row[text_column]
        
        # Process text
        if cache is not None:
            chunks, embeddings = cache.get_or_compute(
                text,
                lambda t: process_text(
                    t,
                    model.sentence_transformer_model,
                    device_obj,
                    model.chunk_size,
                    model.chunk_overlap
                )
            )
        else:
            chunks, embeddings = process_text(
                text,
                model.sentence_transformer_model,
                device_obj,
                model.chunk_size,
                model.chunk_overlap
            )
        
        if len(chunks) == 0:
            continue
        
        # Get confounder activations
        with torch.no_grad():
            activations = _get_chunk_confounder_activations(
                model,
                embeddings,
                device_obj
            )  # (num_chunks, num_confounders)
        
        # Store
        for chunk_idx, (chunk_text, chunk_acts) in enumerate(zip(chunks, activations)):
            all_texts.append(text)
            all_chunks.append(chunk_text)
            all_activations.append(chunk_acts)
    
    # Convert to arrays
    all_activations = np.array(all_activations)  # (total_chunks, num_confounders)
    
    logger.info(f"Collected activations from {len(all_chunks)} chunks")
    
    # For each confounder, find top-k activating chunks
    logger.info(f"\nFinding top {top_k_chunks} chunks per confounder...")
    
    interpretations = []
    
    for conf_idx in range(num_confounders):
        conf_activations = all_activations[:, conf_idx]
        
        # Get top-k chunks
        top_indices = np.argsort(conf_activations)[-top_k_chunks:][::-1]
        
        for rank, chunk_idx in enumerate(top_indices):
            interpretations.append({
                'confounder_idx': conf_idx,
                'confounder_type': _get_confounder_type(model, conf_idx),
                'rank': rank + 1,
                'activation': conf_activations[chunk_idx],
                'chunk_text': all_chunks[chunk_idx],
                'source_text_idx': chunk_idx
            })
    
    interp_df = pd.DataFrame(interpretations)
    
    logger.info("\nInterpretation complete!")
    logger.info(f"Top activations per confounder:")
    
    for conf_idx in range(min(5, num_confounders)):
        top_chunk = interp_df[interp_df['confounder_idx'] == conf_idx].iloc[0]
        logger.info(f"\nConfounder {conf_idx} ({top_chunk['confounder_type']}):")
        logger.info(f"  Activation: {top_chunk['activation']:.4f}")
        logger.info(f"  Top chunk: {top_chunk['chunk_text'][:100]}...")
    
    return interp_df


def _get_chunk_confounder_activations(
    model,
    chunk_embeddings: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Get confounder activations for each chunk.
    
    Args:
        model: Trained model
        chunk_embeddings: Tensor of shape (num_chunks, embed_dim)
        device: Device
    
    Returns:
        Activations array of shape (num_chunks, num_confounders)
    """
    if chunk_embeddings.size(0) == 0:
        return np.zeros((0, model.feature_extractor.num_total_confounders))
    
    chunk_embeddings = chunk_embeddings.to(device)
    
    # Get confounder filters
    filters_list = []
    if model.feature_extractor.explicit_confounders is not None:
        filters_list.append(model.feature_extractor.explicit_confounders)
    if model.feature_extractor.latent_confounders is not None:
        filters_list.append(model.feature_extractor.latent_confounders)
    
    if not filters_list:
        return np.zeros((chunk_embeddings.size(0), 0))
    
    filters = torch.cat(filters_list, dim=0)  # (C, D)
    
    # Compute similarity between chunks and confounders
    chunk_norm = F.normalize(chunk_embeddings, p=2, dim=1)  # (L, D)
    filter_norm = F.normalize(filters, p=2, dim=1)  # (C, D)
    
    # Similarity matrix: (L, C)
    similarities = torch.mm(chunk_norm, filter_norm.t())
    
    return similarities.cpu().numpy()


def _get_confounder_type(model, conf_idx: int) -> str:
    """Get confounder type (explicit or latent)."""
    if conf_idx < model.feature_extractor.num_explicit:
        return "explicit"
    else:
        return "latent"


def analyze_confounder_correlations(
    model_path: Path,
    dataset: pd.DataFrame,
    text_column: str,
    cache: Optional[EmbeddingCache] = None,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Analyze correlations between confounders.
    
    Args:
        model_path: Path to trained model
        dataset: Dataset with texts
        text_column: Text column name
        cache: Optional cache
        device: Device string
    
    Returns:
        Dictionary with correlation analysis
    """
    logger.info("Analyzing confounder correlations...")
    
    # Load model
    device_obj = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device_obj, weights_only=False)
    
    if 'num_treatments' in checkpoint.get('config', {}):
        model = MultiTreatmentDragonnetText.load_from_checkpoint(
            str(model_path),
            device=device
        )
    else:
        model = CausalDragonnetText.load_from_checkpoint(
            str(model_path),
            device=device
        )
    
    model.eval()
    
    # Extract confounder features for all samples
    all_features = []
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Extracting features"):
        text = row[text_column]
        
        if cache is not None:
            chunks, embeddings = cache.get_or_compute(
                text,
                lambda t: process_text(
                    t,
                    model.sentence_transformer_model,
                    device_obj,
                    model.chunk_size,
                    model.chunk_overlap
                )
            )
        else:
            chunks, embeddings = process_text(
                text,
                model.sentence_transformer_model,
                device_obj,
                model.chunk_size,
                model.chunk_overlap
            )
        
        if embeddings.size(0) == 0:
            continue
        
        with torch.no_grad():
            features = model.feature_extractor([embeddings])
            all_features.append(features.cpu())
    
    # Concatenate
    all_features = torch.cat(all_features, dim=0).numpy()  # (n_samples, feature_dim)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(all_features.T)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    num_features = corr_matrix.shape[0]
    
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if abs(corr_matrix[i, j]) > 0.7:
                high_corr_pairs.append({
                    'feature_i': i,
                    'feature_j': j,
                    'correlation': corr_matrix[i, j]
                })
    
    logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
    
    return {
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'feature_means': all_features.mean(axis=0),
        'feature_stds': all_features.std(axis=0)
    }


def visualize_confounder_embeddings(
    model_path: Path,
    output_path: Path,
    method: str = "pca"
) -> None:
    """
    Visualize learned confounder embeddings in 2D.
    
    Args:
        model_path: Path to trained model
        output_path: Path to save visualization
        method: Dimensionality reduction method ('pca', 'tsne', 'umap')
    """
    logger.info(f"Visualizing confounder embeddings using {method.upper()}...")
    
    # Load model
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'num_treatments' in checkpoint.get('config', {}):
        model = MultiTreatmentDragonnetText.load_from_checkpoint(
            str(model_path),
            device="cpu"
        )
    else:
        model = CausalDragonnetText.load_from_checkpoint(
            str(model_path),
            device="cpu"
        )
    
    # Get confounder embeddings
    filters_list = []
    labels = []
    
    if model.feature_extractor.explicit_confounders is not None:
        explicit = model.feature_extractor.explicit_confounders.cpu().numpy()
        filters_list.append(explicit)
        labels.extend([f"Explicit_{i}" for i in range(explicit.shape[0])])
    
    if model.feature_extractor.latent_confounders is not None:
        latent = model.feature_extractor.latent_confounders.detach().cpu().numpy()
        filters_list.append(latent)
        labels.extend([f"Latent_{i}" for i in range(latent.shape[0])])
    
    if not filters_list:
        logger.warning("No confounders to visualize")
        return
    
    embeddings = np.vstack(filters_list)
    
    # Apply dimensionality reduction
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        coords_2d = reducer.fit_transform(embeddings)
        logger.info(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
    
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
    
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Split by type
        n_explicit = model.feature_extractor.num_explicit
        
        if n_explicit > 0:
            ax.scatter(
                coords_2d[:n_explicit, 0],
                coords_2d[:n_explicit, 1],
                c='blue',
                label='Explicit',
                s=100,
                alpha=0.6
            )
        
        if n_explicit < len(labels):
            ax.scatter(
                coords_2d[n_explicit:, 0],
                coords_2d[n_explicit:, 1],
                c='red',
                label='Latent',
                s=100,
                alpha=0.6
            )
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (coords_2d[i, 0], coords_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        ax.set_title(f"Confounder Embeddings ({method.upper()})")
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
        
        plt.close()
    
    except ImportError as e:
        logger.warning(f"Cannot create visualization: {e}")
        logger.warning("Install matplotlib to enable visualization")
