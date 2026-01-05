# cdt/models/feature_extractor.py
"""Feature extractor for learning confounder representations from text embeddings."""

import logging
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .components import CrossAttentionAggregator


logger = logging.getLogger(__name__)


def pad_chunks(
    list_of_chunk_embeddings: List[torch.Tensor],
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length chunk embeddings to create a batch.

    Args:
        list_of_chunk_embeddings: List of tensors, each (num_chunks_i, embed_dim)
        device: Device to create tensors on

    Returns:
        padded_chunks: (batch, max_len, embed_dim)
        mask: (batch, 1, max_len) boolean mask where True = padding
    """
    if not list_of_chunk_embeddings:
        raise ValueError("Empty list of embeddings")

    batch_size = len(list_of_chunk_embeddings)
    embed_dim = list_of_chunk_embeddings[0].size(-1)
    max_len = max(emb.size(0) for emb in list_of_chunk_embeddings)

    padded = torch.zeros(batch_size, max_len, embed_dim, device=device)
    mask = torch.ones(batch_size, 1, max_len, dtype=torch.bool, device=device)

    for i, emb in enumerate(list_of_chunk_embeddings):
        seq_len = emb.size(0)
        padded[i, :seq_len] = emb
        mask[i, 0, :seq_len] = False

    return padded, mask


class FeatureExtractor(nn.Module):
    """
    Extract confounder representations from text chunks using cross-attention.

    Architecture:
    1. Compute cosine similarities between chunk embeddings and confounder vectors
    2. Use similarities as attention weights in cross-attention aggregation
    3. Project chunks through learned value transformation
    4. Aggregate weighted values per confounder
    5. Apply MLP projection to reduce dimensionality
    6. Apply BatchNorm for normalization
    """

    def __init__(
        self,
        embedding_dim: int,
        num_latent_confounders: int,
        explicit_confounder_texts: Optional[List[str]],
        value_dim: int,
        num_attention_heads: int,
        attention_dropout: float,
        projection_dim: int,
        sentence_transformer_model: SentenceTransformer,
        phantom_confounders: int = 0,
        device: Optional[torch.device] = None,
        explicit_confounder_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize feature extractor.

        Args:
            embedding_dim: Dimension of sentence embeddings (e.g., 384 for MiniLM)
            num_latent_confounders: Number of learnable latent confounders
            explicit_confounder_texts: Optional list of explicit confounder query texts
            value_dim: Output dimension per confounder in cross-attention (e.g., 128)
            num_attention_heads: Number of attention heads per confounder
            attention_dropout: Dropout rate on attention weights
            projection_dim: Final output dimension after MLP projection (e.g., 200)
            sentence_transformer_model: Pre-loaded SentenceTransformer for encoding explicit texts
            phantom_confounders: Number of "phantom" confounders to pad output with zeros
                                 (used when current model has fewer confounders than pretrained)
            device: Device to create tensors on (default: CPU, will be moved later)
            explicit_confounder_embeddings: Optional pre-computed embeddings tensor (avoids re-encoding)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_latent = num_latent_confounders
        self.explicit_confounder_texts = explicit_confounder_texts
        self.value_dim = value_dim
        self.num_attention_heads = num_attention_heads
        self.projection_dim = projection_dim
        self.phantom_confounders = phantom_confounders

        # Use CPU as default device for initialization, will be moved later
        if device is None:
            device = torch.device('cpu')

        # Initialize latent confounders as learnable parameters
        if num_latent_confounders > 0:
            self.latent_confounders = nn.Parameter(
                torch.randn(num_latent_confounders, embedding_dim, device=device) * 0.1
            )
        else:
            self.latent_confounders = None

        # Initialize explicit confounders
        if explicit_confounder_embeddings is not None:
            # Use pre-computed embeddings (already encoded)
            logger.info(f"FeatureExtractor: Using {len(explicit_confounder_embeddings)} pre-computed explicit confounder embeddings.")
            explicit_embeddings = explicit_confounder_embeddings.to(device)
            self.register_buffer('explicit_confounders', explicit_embeddings)
            self.num_explicit = explicit_embeddings.size(0)

        elif explicit_confounder_texts:
            # Fallback: Encode from texts (Force CPU to avoid CUDA init crashes in workers)
            logger.info(f"FeatureExtractor: Encoding {len(explicit_confounder_texts)} explicit confounders (CPU Fallback).")
            explicit_embeddings = sentence_transformer_model.encode(
                explicit_confounder_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device='cpu'
            )
            explicit_embeddings = explicit_embeddings.to(device)
            self.register_buffer('explicit_confounders', explicit_embeddings)
            self.num_explicit = len(explicit_confounder_texts)
        else:
            self.explicit_confounders = None
            self.num_explicit = 0

        # Total number of confounders
        self.num_total_confounders = num_latent_confounders + self.num_explicit

        # Cross-attention aggregator
        self.aggregator = CrossAttentionAggregator(
            embedding_dim=embedding_dim,
            value_dim=value_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout
        )

        # MLP projection to reduce dimensionality
        # From (C * value_dim) to projection_dim
        cross_attn_dim = self.num_total_confounders * value_dim
        self.projection_mlp = nn.Sequential(
            nn.Linear(cross_attn_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # BatchNorm to standardize output across samples
        self.feature_norm = nn.BatchNorm1d(self.output_dim, affine=False)

    @property
    def output_dim(self):
        """Total output dimension after projection."""
        # projection_dim + phantom padding
        return self.projection_dim + (self.phantom_confounders * self.value_dim if self.phantom_confounders > 0 else 0)

    def forward(self, list_of_chunk_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract confounder features from chunk embeddings.

        Args:
            list_of_chunk_embeddings: List of tensors, each (num_chunks_i, embed_dim)

        Returns:
            Confounder features: (batch, output_dim)
        """
        device = list_of_chunk_embeddings[0].device
        padded_chunks, mask = pad_chunks(list_of_chunk_embeddings, device)
        B, L, D = padded_chunks.shape

        # If no confounders defined, just return mean pooling projected
        if self.num_total_confounders == 0:
            valid_mask = (~mask.squeeze(1)).unsqueeze(-1).float()  # (B, L, 1)
            pooled = (padded_chunks * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            # Project to output dimension
            pooled = self.projection_mlp(pooled.repeat(1, self.projection_dim // D + 1)[:, :self.projection_dim])
            if self.phantom_confounders > 0:
                phantom_pad = torch.zeros(B, self.phantom_confounders * self.value_dim, device=device, dtype=pooled.dtype)
                pooled = torch.cat([pooled, phantom_pad], dim=1)
            return self.feature_norm(pooled)

        # Concatenate all confounder embeddings
        filters_list = []
        if self.explicit_confounders is not None:
            filters_list.append(self.explicit_confounders)
        if self.latent_confounders is not None:
            filters_list.append(self.latent_confounders)

        filters = torch.cat(filters_list, dim=0)  # (C, D)
        C = filters.shape[0]

        # Compute cosine similarities (attention scores)
        x_norm = F.normalize(padded_chunks, p=2, dim=2)  # (B, L, D)
        f_norm = F.normalize(filters, p=2, dim=1)  # (C, D)
        attn_scores = torch.einsum('bld,cd->bcl', x_norm, f_norm)  # (B, C, L)

        # Cross-attention aggregation
        # Output: (B, C, value_dim)
        cross_attn_output = self.aggregator(attn_scores, padded_chunks, mask)

        # Flatten: (B, C * value_dim)
        flat_output = cross_attn_output.reshape(B, C * self.value_dim)

        # MLP projection: (B, projection_dim)
        projected = self.projection_mlp(flat_output)

        # Add phantom padding if needed (zeros for "missing" confounders from pretrained model)
        if self.phantom_confounders > 0:
            phantom_pad = torch.zeros(
                B, self.phantom_confounders * self.value_dim,
                device=device, dtype=projected.dtype
            )
            projected = torch.cat([projected, phantom_pad], dim=1)

        # Apply BatchNorm to standardize across samples
        output = self.feature_norm(projected)

        return output
