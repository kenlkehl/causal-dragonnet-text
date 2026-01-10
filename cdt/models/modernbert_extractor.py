# cdt/models/modernbert_extractor.py
"""Feature extractor using ModernBERT CLS token embedding."""

import logging
from typing import Optional, List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger(__name__)


class ModernBertFeatureExtractor(nn.Module):
    """
    Extract text representations using ModernBERT CLS token embedding.
    
    Architecture:
    1. Tokenize input text using ModernBERT tokenizer
    2. Forward through ModernBERT encoder
    3. Extract CLS token embedding (position 0 of last hidden state)
    4. Optional projection layer to match downstream dimension requirements
    
    ModernBERT supports up to 8192 tokens, so no chunking is needed for most clinical texts.
    """
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        projection_dim: Optional[int] = None,
        freeze_bert: bool = False,
        max_length: int = 8192,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ModernBERT feature extractor.
        
        Args:
            model_name: HuggingFace model name for ModernBERT
            projection_dim: If provided, project CLS embedding to this dimension.
                           If None, use raw ModernBERT hidden size (768).
            freeze_bert: If True, freeze ModernBERT weights during training
            max_length: Maximum sequence length for tokenization
            device: Device to place model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self._projection_dim = projection_dim
        
        if device is None:
            device = torch.device('cpu')
        self._device = device
        
        # Load tokenizer and model
        logger.info(f"Loading ModernBERT tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.bert.config.hidden_size
        logger.info(f"ModernBERT hidden size: {self.hidden_size}")
        
        # Freeze BERT weights if requested
        if freeze_bert:
            logger.info("Freezing ModernBERT weights")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(projection_dim, projection_dim),
                nn.LayerNorm(projection_dim),
            )
            logger.info(f"Added projection layer: {self.hidden_size} -> {projection_dim}")
        else:
            self.projection = None
        
        # Note: BatchNorm removed - unstable with small batch sizes
        # Using LayerNorm instead for stable normalization
        self.feature_norm = nn.LayerNorm(self.output_dim)

        logger.info(f"ModernBertFeatureExtractor initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Output dim: {self.output_dim}")
        logger.info(f"  Frozen: {freeze_bert}")
    
    @property
    def output_dim(self) -> int:
        """Total output dimension after optional projection."""
        if self._projection_dim is not None:
            return self._projection_dim
        return self.hidden_size
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract CLS token embeddings from texts.
        
        Args:
            texts: List of text strings to encode
        
        Returns:
            CLS embeddings: (batch, output_dim)
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
        
        # Forward through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract CLS token embedding (position 0)
        # last_hidden_state shape: (batch, seq_len, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        
        # Apply projection if configured
        if self.projection is not None:
            cls_embedding = self.projection(cls_embedding)
        
        # Apply BatchNorm
        output = self.feature_norm(cls_embedding)
        
        return output
    
    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
