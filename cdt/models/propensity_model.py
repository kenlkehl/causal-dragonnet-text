# cdt/models/propensity_model.py
"""Propensity-only model for dataset trimming before causal inference."""

import gc
import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_extractor import CNNFeatureExtractor
from .bert_extractor import BertFeatureExtractor


logger = logging.getLogger(__name__)


class PropensityNet(nn.Module):
    """
    Propensity prediction network with same representation as DragonNet.

    Uses 6-layer representation followed by a single propensity head.
    """

    def __init__(self, input_dim: int, representation_dim: int = 200):
        super().__init__()

        # Shared representation layers (same as DragonNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc3 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc4 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc5 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc6 = nn.Linear(representation_dim, representation_dim)

        # Single propensity head (same as DragonNet)
        self.propensity_fc1 = nn.Linear(representation_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the propensity network.

        Args:
            features: Feature tensor from feature extractor (batch, input_dim)

        Returns:
            t_logit: Propensity logits (batch, 1)
        """
        h = F.relu(self.representation_fc1(features))
        h = F.relu(self.representation_fc2(h))
        h = F.relu(self.representation_fc3(h))
        h = F.relu(self.representation_fc4(h))
        h = F.relu(self.representation_fc5(h))
        h = F.elu(self.representation_fc6(h))

        t_logit = self.propensity_fc1(h)

        return t_logit


class PropensityOnlyModel(nn.Module):
    """
    Propensity-score-only model for dataset trimming.

    Uses same architecture as CausalCNNText/DragonNet:
    - Feature extractor (CNN or BERT)
    - 6-layer representation network
    - Single propensity head

    This model is trained to predict P(T=1|X) using binary cross-entropy loss.
    Used for generating propensity scores for trimming before DragonNet training.
    """

    def __init__(
        self,
        # Feature extractor type
        feature_extractor_type: str = "cnn",
        # CNN-specific args
        embedding_dim: int = 128,
        kernel_sizes: List[int] = [3, 4, 5, 7],
        explicit_filter_concepts: Optional[Dict[str, List[str]]] = None,
        num_kmeans_filters: int = 64,
        num_random_filters: int = 0,
        cnn_dropout: float = 0.1,
        max_length: int = 2048,
        min_word_freq: int = 2,
        max_vocab_size: Optional[int] = 50000,
        projection_dim: Optional[int] = 128,
        # BERT-specific args
        bert_model_name: str = "bert-base-uncased",
        bert_max_length: int = 512,
        bert_projection_dim: Optional[int] = 128,
        bert_dropout: float = 0.1,
        bert_freeze_encoder: bool = False,
        bert_gradient_checkpointing: bool = False,
        # Propensity network args
        representation_dim: int = 128,
        device: str = "cuda:0"
    ):
        """
        Initialize propensity-only model.

        Args:
            feature_extractor_type: "cnn" or "bert"
            embedding_dim: (CNN) Dimension of word embeddings
            kernel_sizes: (CNN) List of kernel sizes for n-gram capture
            explicit_filter_concepts: (CNN) Dict mapping kernel_size to concept phrases
            num_kmeans_filters: (CNN) Number of k-means derived filters per kernel size
            num_random_filters: (CNN) Number of randomly initialized filters per kernel size
            cnn_dropout: (CNN) Dropout rate
            max_length: (CNN) Maximum sequence length in tokens
            min_word_freq: (CNN) Minimum word frequency for vocabulary inclusion
            max_vocab_size: (CNN) Maximum vocabulary size
            projection_dim: (CNN) Dimension to project CNN output to
            bert_model_name: (BERT) HuggingFace model name or path
            bert_max_length: (BERT) Maximum sequence length in subword tokens
            bert_projection_dim: (BERT) Projection dimension after CLS token
            bert_dropout: (BERT) Dropout rate for projection layer
            bert_freeze_encoder: (BERT) Whether to freeze transformer weights
            bert_gradient_checkpointing: (BERT) Enable gradient checkpointing
            representation_dim: Dimension of representation layers
            device: Device string
        """
        super().__init__()

        self._device = torch.device(device)
        self.feature_extractor_type = feature_extractor_type

        # Store config for checkpointing
        self.config = {
            'feature_extractor_type': feature_extractor_type,
            'embedding_dim': embedding_dim,
            'kernel_sizes': kernel_sizes,
            'explicit_filter_concepts': explicit_filter_concepts,
            'num_kmeans_filters': num_kmeans_filters,
            'num_random_filters': num_random_filters,
            'cnn_dropout': cnn_dropout,
            'max_length': max_length,
            'min_word_freq': min_word_freq,
            'max_vocab_size': max_vocab_size,
            'projection_dim': projection_dim,
            'bert_model_name': bert_model_name,
            'bert_max_length': bert_max_length,
            'bert_projection_dim': bert_projection_dim,
            'bert_dropout': bert_dropout,
            'bert_freeze_encoder': bert_freeze_encoder,
            'bert_gradient_checkpointing': bert_gradient_checkpointing,
            'representation_dim': representation_dim
        }

        # Initialize feature extractor based on type
        if feature_extractor_type == "bert":
            self.feature_extractor = BertFeatureExtractor(
                model_name=bert_model_name,
                projection_dim=bert_projection_dim,
                max_length=bert_max_length,
                dropout=bert_dropout,
                freeze_encoder=bert_freeze_encoder,
                device=self._device
            )
            if bert_gradient_checkpointing:
                self.feature_extractor.gradient_checkpointing_enable()
            logger.info(f"Propensity model using BERT feature extractor: {bert_model_name}")
        else:
            # CNN feature extractor (default)
            self.feature_extractor = CNNFeatureExtractor(
                embedding_dim=embedding_dim,
                kernel_sizes=kernel_sizes,
                explicit_filter_concepts=explicit_filter_concepts,
                num_kmeans_filters=num_kmeans_filters,
                num_random_filters=num_random_filters,
                projection_dim=projection_dim,
                dropout=cnn_dropout,
                max_length=max_length,
                min_word_freq=min_word_freq,
                max_vocab_size=max_vocab_size,
                device=self._device
            )
            logger.info("Propensity model using CNN feature extractor")

        # Propensity network
        input_dim = self.feature_extractor.output_dim
        self.propensity_net = PropensityNet(
            input_dim=input_dim,
            representation_dim=representation_dim
        )

        # Move to device
        self.to(self._device)

        logger.info(f"PropensityOnlyModel initialized:")
        logger.info(f"  Feature extractor: {feature_extractor_type}")
        logger.info(f"  Feature extractor output: {input_dim}")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Device: {self._device}")

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            texts: List of text strings

        Returns:
            t_logit: Propensity logits (batch, 1)
        """
        features = self.feature_extractor(texts)
        t_logit = self.propensity_net(features)
        return t_logit

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with 'texts' and 'treatment' keys

        Returns:
            Dictionary with loss and predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)

        # Forward pass
        t_logit = self.forward(texts)

        # Binary cross-entropy loss for treatment prediction
        loss = F.binary_cross_entropy_with_logits(
            t_logit.squeeze(-1),
            treatments
        )

        return {
            'loss': loss,
            't_logit': t_logit.detach()
        }

    def predict(self, texts: List[str]) -> torch.Tensor:
        """
        Predict propensity scores.

        Args:
            texts: List of text strings

        Returns:
            Propensity probabilities (batch,)
        """
        with torch.no_grad():
            t_logit = self.forward(texts)
            propensity = torch.sigmoid(t_logit).squeeze(-1)
            return propensity

    def fit_tokenizer(self, texts: List[str]) -> 'PropensityOnlyModel':
        """
        Fit the word tokenizer on training texts.

        For CNN: This MUST be called before using the model for training or inference.
        For BERT: This is a no-op (BERT uses its pretrained tokenizer).

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        if hasattr(self.feature_extractor, 'fit_tokenizer'):
            self.feature_extractor.fit_tokenizer(texts)
        # BERT uses pretrained tokenizer, no fitting needed
        return self

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)


def create_propensity_model_from_config(
    arch_config,
    representation_dim: int,
    device: torch.device
) -> PropensityOnlyModel:
    """
    Create a PropensityOnlyModel from architecture config.

    Args:
        arch_config: ModelArchitectureConfig instance
        representation_dim: Dimension for representation layers
        device: PyTorch device

    Returns:
        PropensityOnlyModel instance
    """
    feature_extractor_type = getattr(arch_config, 'feature_extractor_type', 'cnn')

    model = PropensityOnlyModel(
        feature_extractor_type=feature_extractor_type,
        # CNN args
        embedding_dim=arch_config.cnn_embedding_dim,
        kernel_sizes=arch_config.cnn_kernel_sizes,
        explicit_filter_concepts=arch_config.cnn_explicit_filter_concepts,
        num_kmeans_filters=arch_config.cnn_num_kmeans_filters,
        num_random_filters=arch_config.cnn_num_random_filters,
        cnn_dropout=arch_config.cnn_dropout,
        max_length=arch_config.cnn_max_length,
        min_word_freq=getattr(arch_config, 'cnn_min_word_freq', 2),
        max_vocab_size=getattr(arch_config, 'cnn_max_vocab_size', 50000),
        projection_dim=arch_config.dragonnet_representation_dim,
        # BERT args
        bert_model_name=getattr(arch_config, 'bert_model_name', 'bert-base-uncased'),
        bert_max_length=getattr(arch_config, 'bert_max_length', 512),
        bert_projection_dim=getattr(arch_config, 'bert_projection_dim', 128),
        bert_dropout=getattr(arch_config, 'bert_dropout', 0.1),
        bert_freeze_encoder=getattr(arch_config, 'bert_freeze_encoder', False),
        bert_gradient_checkpointing=getattr(arch_config, 'bert_gradient_checkpointing', False),
        # Propensity network args
        representation_dim=representation_dim,
        device=str(device)
    )

    return model
