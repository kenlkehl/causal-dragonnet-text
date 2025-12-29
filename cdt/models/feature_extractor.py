# cdt/models/feature_extractor.py
"""Feature extractor for learning confounder representations from text embeddings."""

import logging
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .components import ConfounderAggregator


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
    Extract confounder representations from text chunks.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_latent_confounders: int,
        explicit_confounder_texts: Optional[List[str]],
        features_per_confounder: int,
        aggregator_mode: str,
        sentence_transformer_model: SentenceTransformer,
        phantom_confounders: int = 0,
        device: Optional[torch.device] = None,
        explicit_confounder_embeddings: Optional[torch.Tensor] = None,
        arctanh_transform: bool = False
    ):
        """
        Initialize feature extractor.
        
        Args:
            embedding_dim: Dimension of sentence embeddings
            num_latent_confounders: Number of learnable latent confounders
            explicit_confounder_texts: Optional list of explicit confounder query texts
            features_per_confounder: Number of feature detectors per confounder (default 1)
            aggregator_mode: Aggregation mode ('attn', 'max', 'lsep', etc.)
            sentence_transformer_model: Pre-loaded SentenceTransformer for encoding explicit texts
            phantom_confounders: Number of "phantom" confounders to pad output with zeros
                                 (used when current model has fewer confounders than pretrained)
            device: Device to create tensors on (default: CPU, will be moved later)
            explicit_confounder_embeddings: Optional pre-computed embeddings tensor (avoids re-encoding)
            arctanh_transform: If True, apply arctanh to cosine similarities before BatchNorm.
                               This stretches values near ±1 towards ±∞, expanding compressed ranges.
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_latent = num_latent_confounders
        self.explicit_confounder_texts = explicit_confounder_texts
        self.features_per_confounder = features_per_confounder
        self.phantom_confounders = phantom_confounders
        self.arctanh_transform = arctanh_transform
        
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
            # Ensure they are on the correct device
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
            # Ensure explicit embeddings are on the correct device
            explicit_embeddings = explicit_embeddings.to(device)
            # Store as non-trainable parameter
            self.register_buffer('explicit_confounders', explicit_embeddings)
            self.num_explicit = len(explicit_confounder_texts)
        else:
            self.explicit_confounders = None
            self.num_explicit = 0
        
        # Total number of confounders
        self.num_total_confounders = num_latent_confounders + self.num_explicit
        
        # Initialize aggregator
        self.aggregator = ConfounderAggregator(
            mode=aggregator_mode,
            per_confounder_params=True
        )
        
        # Multi-head projection for extracting multiple features per confounder
        if features_per_confounder > 1 and self.num_total_confounders > 0:
            # Construct base confounder filters
            base_filters_list = []
            if self.explicit_confounders is not None:
                base_filters_list.append(self.explicit_confounders)
            if self.latent_confounders is not None:
                base_filters_list.append(self.latent_confounders)
            
            base_filters = torch.cat(base_filters_list, dim=0)  # (C, D)
            
            # Initialize projections as variations of base filters
            init_proj = base_filters.unsqueeze(1).repeat(1, features_per_confounder, 1)
            init_proj = init_proj + torch.randn_like(init_proj) * 0.1
            
            self.confounder_projection = nn.Parameter(init_proj)
        else:
            self.confounder_projection = None
        
        # BatchNorm to standardize confounder features per-feature across samples
        # This normalizes each confounder independently (unlike LayerNorm which averages across features)
        self.feature_norm = nn.BatchNorm1d(self.output_dim, affine=False)

    @property
    def out_per_conf(self):
        """Output dimension per confounder after aggregation."""
        base_out = getattr(self.aggregator, 'features_per_conf', 1)
        return base_out * self.features_per_confounder

    @property
    def output_dim(self):
        """Total output dimension including phantom confounders."""
        return (self.num_total_confounders + self.phantom_confounders) * self.out_per_conf

    def forward(self, list_of_chunk_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract confounder features from chunk embeddings.
        
        Args:
            list_of_chunk_embeddings: List of tensors, each (num_chunks_i, embed_dim)
        
        Returns:
            Confounder features: (batch, total_confounders * out_per_conf)
        """
        device = list_of_chunk_embeddings[0].device
        padded_chunks, mask = pad_chunks(list_of_chunk_embeddings, device)
        B, L, D = padded_chunks.shape

        # If no confounders defined, just return mean pooling
        if self.num_total_confounders == 0:
            pooled = torch.mean(
                padded_chunks.masked_fill(mask.squeeze(1).unsqueeze(-1), 0.0),
                dim=1
            )
            # Add phantom padding if needed
            if self.phantom_confounders > 0:
                phantom_pad = torch.zeros(
                    B, self.phantom_confounders * self.out_per_conf,
                    device=device, dtype=pooled.dtype
                )
                pooled = torch.cat([pooled, phantom_pad], dim=1)
            return pooled

        # Concatenate all confounder embeddings
        filters_list = []
        if self.explicit_confounders is not None:
            filters_list.append(self.explicit_confounders)
        if self.latent_confounders is not None:
            filters_list.append(self.latent_confounders)
        
        filters = torch.cat(filters_list, dim=0)  # (C, D)
        C = filters.shape[0]

        # Apply multi-head projection if enabled
        if self.features_per_confounder > 1 and self.confounder_projection is not None:
            # filters: (C, D) -> projected: (C, K, D)
            # Compute similarity for each of K projections per confounder
            projected_filters = self.confounder_projection  # (C, K, D)

            projected_filters = F.normalize(projected_filters, p=2, dim=2)  # Normalize each projection
            
            # Reshape to (C*K, D, 1) for conv1d
            filters_expanded = projected_filters.reshape(C * self.features_per_confounder, D).unsqueeze(2)
            
            x = F.normalize(padded_chunks, p=2, dim=2)
            fm = F.conv1d(x.transpose(1, 2), filters_expanded)  # (B, C*K, L)
            
            # Expand mask to match: (B, C*K, L)
            mask_expanded = mask.repeat(1, C * self.features_per_confounder, 1)
            
            pooled = self.aggregator(fm, mask_expanded)  # (B, C*out_per_conf)
        else:
            # Original single-feature-per-confounder path
            x = F.normalize(padded_chunks, p=2, dim=2)
            f = F.normalize(filters, p=2, dim=1).unsqueeze(2)
            fm = F.conv1d(x.transpose(1, 2), f)  # (B, C, L)

            pooled = self.aggregator(fm, mask)  # (B, C*out_per_conf)
        
        # Add phantom padding if needed (zeros for "missing" confounders from pretrained model)
        if self.phantom_confounders > 0:
            phantom_pad = torch.zeros(
                B, self.phantom_confounders * self.out_per_conf,
                device=device, dtype=pooled.dtype
            )
            pooled = torch.cat([pooled, phantom_pad], dim=1)
        
        # Optional arctanh to stretch compressed cosine similarities before standardization
        if self.arctanh_transform:
            # Clamp to avoid ±∞ at exactly ±1
            pooled = torch.arctanh(torch.clamp(pooled, -0.999, 0.999))
        
        # Apply BatchNorm to standardize each confounder feature independently across samples
        pooled = self.feature_norm(pooled)
        
        return pooled