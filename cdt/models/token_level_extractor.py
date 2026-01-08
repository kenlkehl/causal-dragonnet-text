# cdt/models/token_level_extractor.py
"""Token-level feature extractor with frozen backbone and confounder-aligned projection."""

import logging
from typing import Optional, List, Dict, Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TokenLevelFeatureExtractor(nn.Module):
    """
    Extract confounder-aligned features at token level using frozen backbone.

    Architecture:
    1. Frozen transformer backbone (e.g., from SentenceTransformer) produces token embeddings
    2. Trainable projection layer maps tokens to confounder scores
    3. Projection initialized with confounder text embeddings for interpretability
    4. Aggregation reduces (B, T, C) to (B, C) for downstream causal model

    The key insight: each output dimension corresponds to a specific confounder,
    and the projection weights start as the semantic embedding of that confounder.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        explicit_confounder_texts: Optional[List[str]] = None,
        num_latent_confounders: int = 0,
        aggregation_method: Literal["max", "mean", "attention", "topk"] = "attention",
        topk: int = 5,
        anchor_strength: float = 0.0,
        projection_dim: Optional[int] = None,
        max_length: int = 10000,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize token-level feature extractor.

        Args:
            model_name: HuggingFace model name (or sentence-transformers model)
            explicit_confounder_texts: List of confounder descriptions for initialization
            num_latent_confounders: Number of additional learnable (randomly initialized) confounders
            aggregation_method: How to aggregate token scores to sample scores
                - "max": Max pooling over tokens
                - "mean": Mean pooling over tokens (excluding padding)
                - "attention": Learnable attention weights over tokens
                - "topk": Average of top-k scores per confounder
            topk: Number of top tokens for topk aggregation
            anchor_strength: Regularization strength to keep projection close to initial embeddings
                            (0 = no regularization, allows full drift)
            projection_dim: Optional MLP projection after aggregation (None = output is C-dimensional)
            max_length: Maximum sequence length for tokenization (default 10000)
            device: Device to create tensors on
        """
        super().__init__()

        if device is None:
            device = torch.device('cpu')
        self._device = device

        self.model_name = model_name
        self.explicit_confounder_texts = explicit_confounder_texts or []
        self.num_latent_confounders = num_latent_confounders
        self.aggregation_method = aggregation_method
        self.topk = topk
        self.anchor_strength = anchor_strength
        self.projection_dim = projection_dim
        self.max_length = max_length

        # Load backbone model and tokenizer
        logger.info(f"Loading backbone model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.backbone.config.hidden_size

        # Get backbone's max position embeddings for chunking
        self.backbone_max_length = getattr(
            self.backbone.config, 'max_position_embeddings', 512
        )
        logger.info(f"Backbone max position embeddings: {self.backbone_max_length}")

        # Move backbone to device
        self.backbone = self.backbone.to(device)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        logger.info(f"Backbone frozen. Embedding dim: {self.embedding_dim}")

        # Initialize confounder projection
        self.num_explicit = len(self.explicit_confounder_texts)
        self.num_total_confounders = self.num_explicit + num_latent_confounders

        if self.num_total_confounders == 0:
            raise ValueError("Must have at least one confounder (explicit or latent)")

        # Create projection layer (C, D) - each row is a confounder "query" vector
        self.confounder_projection = nn.Linear(
            self.embedding_dim,
            self.num_total_confounders,
            bias=False
        )

        # Initialize explicit confounders from text embeddings
        if self.num_explicit > 0:
            logger.info(f"Encoding {self.num_explicit} explicit confounder texts...")
            explicit_embs = self._encode_texts(self.explicit_confounder_texts)
            # Normalize for cosine similarity behavior
            explicit_embs = F.normalize(explicit_embs, p=2, dim=1)
            self.confounder_projection.weight.data[:self.num_explicit] = explicit_embs

            # Store anchors for regularization and interpretability
            self.register_buffer('confounder_anchors', explicit_embs.clone())
        else:
            self.register_buffer('confounder_anchors', torch.empty(0, self.embedding_dim))

        # Initialize latent confounders randomly
        if num_latent_confounders > 0:
            latent_init = torch.randn(num_latent_confounders, self.embedding_dim) * 0.1
            latent_init = F.normalize(latent_init, p=2, dim=1)
            self.confounder_projection.weight.data[self.num_explicit:] = latent_init
            logger.info(f"Initialized {num_latent_confounders} latent confounders")

        # Learnable temperature for attention aggregation
        if aggregation_method == "attention":
            self.attention_temperature = nn.Parameter(torch.ones(self.num_total_confounders))

        # Optional projection MLP after aggregation
        if projection_dim is not None:
            self.output_projection = nn.Sequential(
                nn.Linear(self.num_total_confounders, projection_dim),
                # nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(projection_dim, projection_dim),
                # nn.LayerNorm(projection_dim),
            )
            self._output_dim = projection_dim
        else:
            self.output_projection = None
            self._output_dim = self.num_total_confounders

        # Output normalization - commented out to allow full range of features
        # self.output_norm = nn.BatchNorm1d(self._output_dim, affine=True)
        self.output_norm = nn.Identity()

        # Store config for checkpointing
        self.config = {
            'model_name': model_name,
            'explicit_confounder_texts': explicit_confounder_texts,
            'num_latent_confounders': num_latent_confounders,
            'aggregation_method': aggregation_method,
            'topk': topk,
            'anchor_strength': anchor_strength,
            'projection_dim': projection_dim,
            'max_length': max_length,
        }

    @property
    def output_dim(self) -> int:
        """Output dimension of the feature extractor."""
        return self._output_dim

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using the backbone model (mean pooling)."""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        # Forward through backbone
        with torch.no_grad():
            outputs = self.backbone(**encoded)
            token_embs = outputs.last_hidden_state  # (B, T, D)

            # Mean pooling (excluding padding)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)  # (B, T, 1)
            masked_embs = token_embs * attention_mask
            summed = masked_embs.sum(dim=1)  # (B, D)
            counts = attention_mask.sum(dim=1).clamp(min=1)  # (B, 1)
            mean_embs = summed / counts  # (B, D)

        return mean_embs

    def _get_token_embeddings_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get token embeddings, chunking if sequence exceeds backbone max length.

        Args:
            input_ids: Token IDs (B, T)
            attention_mask: Attention mask (B, T)

        Returns:
            Token embeddings (B, T, D)
        """
        B, T = input_ids.shape
        max_len = self.backbone_max_length

        if T <= max_len:
            # Short sequence - process directly
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state

        # Long sequence - process in overlapping chunks
        # Use overlap to avoid boundary artifacts
        stride = max_len - 64  # 64 token overlap
        all_embeddings = torch.zeros(B, T, self.embedding_dim, device=input_ids.device)
        count_mask = torch.zeros(B, T, 1, device=input_ids.device)

        for start in range(0, T, stride):
            end = min(start + max_len, T)
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end]

            # Pad if chunk is shorter than max_len (last chunk)
            chunk_len = end - start
            if chunk_len < max_len:
                pad_len = max_len - chunk_len
                chunk_ids = F.pad(chunk_ids, (0, pad_len), value=self.tokenizer.pad_token_id or 0)
                chunk_mask = F.pad(chunk_mask, (0, pad_len), value=0)

            outputs = self.backbone(input_ids=chunk_ids, attention_mask=chunk_mask)
            chunk_embs = outputs.last_hidden_state[:, :chunk_len]  # (B, chunk_len, D)

            # Accumulate with count for averaging overlaps
            all_embeddings[:, start:end] += chunk_embs
            count_mask[:, start:end] += chunk_mask[:, :chunk_len].unsqueeze(-1).float()

        # Average overlapping regions
        count_mask = count_mask.clamp(min=1)
        token_embs = all_embeddings / count_mask

        return token_embs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: tokens -> confounder features.

        Args:
            input_ids: Token IDs (B, T)
            attention_mask: Attention mask (B, T), 1 = real token, 0 = padding

        Returns:
            Confounder features (B, output_dim)
        """
        B, T = input_ids.shape

        # Get token embeddings from frozen backbone (with chunking for long seqs)
        with torch.no_grad():
            token_embs = self._get_token_embeddings_chunked(input_ids, attention_mask)

        # Normalize token embeddings for cosine similarity behavior
        token_embs_norm = F.normalize(token_embs, p=2, dim=-1)

        # Project to confounder scores: (B, T, D) @ (D, C) -> (B, T, C)
        # Note: Linear layer computes x @ W.T, and W is (C, D), so this gives (B, T, C)
        confounder_scores = self.confounder_projection(token_embs_norm)

        # Aggregate over tokens
        # Create padding mask: True where padding
        padding_mask = (attention_mask == 0)  # (B, T)

        aggregated = self._aggregate(confounder_scores, padding_mask)  # (B, C)

        # Optional projection
        if self.output_projection is not None:
            aggregated = self.output_projection(aggregated)

        # Normalize output
        output = self.output_norm(aggregated)

        return output

    def _aggregate(
        self,
        scores: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate token-level scores to sample-level.

        Args:
            scores: (B, T, C) token-level confounder scores
            padding_mask: (B, T) True where padding

        Returns:
            (B, C) aggregated scores
        """
        B, T, C = scores.shape

        # Mask padding positions
        mask_expanded = padding_mask.unsqueeze(-1)  # (B, T, 1)

        if self.aggregation_method == "max":
            # Max pooling (set padding to -inf)
            masked_scores = scores.masked_fill(mask_expanded, float('-inf'))
            aggregated = masked_scores.max(dim=1).values  # (B, C)

        elif self.aggregation_method == "mean":
            # Mean pooling (exclude padding)
            masked_scores = scores.masked_fill(mask_expanded, 0.0)
            valid_counts = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            aggregated = masked_scores.sum(dim=1) / valid_counts  # (B, C)

        elif self.aggregation_method == "attention":
            # Attention-weighted aggregation with learnable temperature
            # Temperature per confounder: (C,) -> (1, 1, C)
            temp = self.attention_temperature.view(1, 1, C).clamp(min=0.1)
            attn_logits = scores / temp  # (B, T, C)

            # Mask padding
            attn_logits = attn_logits.masked_fill(mask_expanded, float('-inf'))

            # Softmax over tokens
            attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, C)

            # Weighted sum
            aggregated = (attn_weights * scores).sum(dim=1)  # (B, C)

        elif self.aggregation_method == "topk":
            # Top-k average per confounder
            masked_scores = scores.masked_fill(mask_expanded, float('-inf'))
            k = min(self.topk, T)
            topk_vals, _ = masked_scores.topk(k, dim=1)  # (B, k, C)
            aggregated = topk_vals.mean(dim=1)  # (B, C)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return aggregated

    def forward_from_text(self, texts: List[str]) -> torch.Tensor:
        """
        Convenience method: raw text -> confounder features.

        Args:
            texts: List of text strings

        Returns:
            Confounder features (B, output_dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        return self.forward(input_ids, attention_mask)

    def get_token_confounder_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get raw token-level confounder scores (for interpretability).

        Returns:
            (B, T, C) scores where score[b, t, c] is how much token t
            relates to confounder c in sample b
        """
        with torch.no_grad():
            token_embs = self._get_token_embeddings_chunked(input_ids, attention_mask)
            token_embs_norm = F.normalize(token_embs, p=2, dim=-1)
            scores = self.confounder_projection(token_embs_norm)
        return scores

    def anchor_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to keep projection close to anchors.

        Only applies to explicit confounders (latents have no anchors).

        Returns:
            Scalar loss (MSE between current and anchor embeddings)
        """
        if self.num_explicit == 0 or self.anchor_strength == 0:
            return torch.tensor(0.0, device=self._device)

        current_explicit = self.confounder_projection.weight[:self.num_explicit]
        current_explicit_norm = F.normalize(current_explicit, p=2, dim=1)

        # Cosine distance (1 - cosine_similarity)
        cosine_sim = (current_explicit_norm * self.confounder_anchors).sum(dim=1)
        loss = (1 - cosine_sim).mean()

        return self.anchor_strength * loss

    def confounder_drift(self) -> Dict[str, torch.Tensor]:
        """
        Measure how much each confounder has drifted from initialization.

        Returns:
            Dictionary with:
            - 'explicit_drift': (num_explicit,) cosine distance from anchors
            - 'explicit_names': list of confounder text names
        """
        result = {}

        if self.num_explicit > 0:
            with torch.no_grad():
                current = F.normalize(
                    self.confounder_projection.weight[:self.num_explicit],
                    p=2, dim=1
                )
                anchors = self.confounder_anchors
                cosine_sim = (current * anchors).sum(dim=1)
                drift = 1 - cosine_sim  # 0 = no drift, 1 = orthogonal, 2 = opposite

            result['explicit_drift'] = drift
            result['explicit_names'] = self.explicit_confounder_texts

        return result

    def interpret_latent_confounders(
        self,
        candidate_concepts: List[str],
        top_k: int = 5
    ) -> List[List[tuple]]:
        """
        Interpret latent confounders by finding nearest concept descriptions.

        Args:
            candidate_concepts: List of concept descriptions to compare against
            top_k: Number of top matches to return per latent confounder

        Returns:
            List of lists, one per latent confounder, each containing
            (concept, similarity) tuples sorted by similarity
        """
        if self.num_latent_confounders == 0:
            return []

        # Encode candidate concepts
        candidate_embs = self._encode_texts(candidate_concepts)
        candidate_embs = F.normalize(candidate_embs, p=2, dim=1)  # (N, D)

        # Get latent confounder weights
        with torch.no_grad():
            latent_weights = self.confounder_projection.weight[self.num_explicit:]  # (L, D)
            latent_weights = F.normalize(latent_weights, p=2, dim=1)

            # Compute similarities: (L, D) @ (D, N) -> (L, N)
            similarities = latent_weights @ candidate_embs.T

        results = []
        for i in range(self.num_latent_confounders):
            sims = similarities[i].cpu().numpy()
            sorted_indices = sims.argsort()[::-1][:top_k]
            matches = [(candidate_concepts[j], float(sims[j])) for j in sorted_indices]
            results.append(matches)

        return results

    def get_confounder_names(self) -> List[str]:
        """Get names for all confounders (explicit texts + 'latent_0', 'latent_1', ...)."""
        names = list(self.explicit_confounder_texts)
        names.extend([f"latent_{i}" for i in range(self.num_latent_confounders)])
        return names
