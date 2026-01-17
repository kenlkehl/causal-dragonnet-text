# cdt/models/outcome_model.py
"""Outcome-only model for validating text→outcome signal before DragonNet training."""

import gc
import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_extractor import CNNFeatureExtractor
from .bert_extractor import BertFeatureExtractor


logger = logging.getLogger(__name__)


class OutcomeNet(nn.Module):
    """
    Outcome prediction network with same representation structure as DragonNet.

    Uses representation layers followed by a single outcome head for continuous
    survival prediction (Softplus activation for positivity).
    """

    def __init__(self, input_dim: int, representation_dim: int = 200, hidden_dim: int = 100):
        super().__init__()

        # Shared representation layers (same as DragonNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc6 = nn.Linear(representation_dim, representation_dim)

        # Single outcome head (same structure as DragonNet outcome heads)
        self.outcome_fc1 = nn.Linear(representation_dim, hidden_dim)
        self.outcome_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.outcome_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the outcome network.

        Args:
            features: Feature tensor from feature extractor (batch, input_dim)

        Returns:
            y_pred: Outcome predictions (batch, 1) - positive via Softplus
            representation: Final representation layer output (batch, representation_dim)
        """
        h = F.relu(self.representation_fc1(features))
        representation = F.elu(self.representation_fc6(h))

        y = F.relu(self.outcome_fc1(representation))
        y = F.elu(self.outcome_fc2(y))
        y_pred = F.softplus(self.outcome_fc3(y))  # Positive continuous outcome

        return y_pred, representation


class OutcomeOnlyModel(nn.Module):
    """
    Outcome-prediction-only model for validating text→outcome signal.

    Uses same architecture as CausalCNNText/DragonNet:
    - Feature extractor (CNN or BERT)
    - Representation layers
    - Single outcome head

    This model is trained to predict Y (continuous outcome) using MSE loss.
    Used to validate whether the CNN can learn meaningful text representations
    for outcome prediction before running full DragonNet training.
    """

    def __init__(
        self,
        # Feature extractor type
        feature_extractor_type: str = "cnn",
        # CNN-specific args
        embedding_dim: int = 128,
        kernel_sizes: List[int] = [3, 4, 5, 7],
        explicit_filter_concepts: Optional[Dict[str, List[str]]] = None,
        num_kmeans_filters: int = 0,
        num_random_filters: int = 256,
        cnn_dropout: float = 0.0,
        max_length: int = 8192,
        min_word_freq: int = 2,
        max_vocab_size: Optional[int] = 20000,
        projection_dim: Optional[int] = 128,
        # BERT-specific args
        bert_model_name: str = "bert-base-uncased",
        bert_max_length: int = 512,
        bert_projection_dim: Optional[int] = 128,
        bert_dropout: float = 0.1,
        bert_freeze_encoder: bool = False,
        bert_gradient_checkpointing: bool = False,
        # Outcome network args
        representation_dim: int = 128,
        hidden_dim: int = 64,
        device: str = "cuda:0"
    ):
        """
        Initialize outcome-only model.

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
            hidden_dim: Dimension of outcome hidden layers
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
            'representation_dim': representation_dim,
            'hidden_dim': hidden_dim
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
            logger.info(f"Outcome model using BERT feature extractor: {bert_model_name}")
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
            logger.info("Outcome model using CNN feature extractor")

        # Outcome network
        input_dim = self.feature_extractor.output_dim
        self.outcome_net = OutcomeNet(
            input_dim=input_dim,
            representation_dim=representation_dim,
            hidden_dim=hidden_dim
        )

        # Move to device
        self.to(self._device)

        logger.info(f"OutcomeOnlyModel initialized:")
        logger.info(f"  Feature extractor: {feature_extractor_type}")
        logger.info(f"  Feature extractor output: {input_dim}")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Device: {self._device}")

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            texts: List of text strings

        Returns:
            y_pred: Outcome predictions (batch, 1)
            representation: Representation layer output (batch, representation_dim)
        """
        features = self.feature_extractor(texts)
        y_pred, representation = self.outcome_net(features)
        return y_pred, representation

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with 'texts' and 'outcome' keys

        Returns:
            Dictionary with loss and predictions
        """
        texts = batch['texts']
        outcomes = batch['outcome']  # (batch,) - continuous survival in months

        # Forward pass
        y_pred, _ = self.forward(texts)

        # MSE loss for continuous outcome
        loss = F.mse_loss(y_pred.squeeze(-1), outcomes)

        return {
            'loss': loss,
            'y_pred': y_pred.detach()
        }

    def predict(self, texts: List[str]) -> torch.Tensor:
        """
        Predict outcomes.

        Args:
            texts: List of text strings

        Returns:
            Outcome predictions (batch,)
        """
        with torch.no_grad():
            y_pred, _ = self.forward(texts)
            return y_pred.squeeze(-1)

    def fit_tokenizer(self, texts: List[str]) -> 'OutcomeOnlyModel':
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


def create_outcome_model_from_config(
    arch_config,
    representation_dim: int,
    hidden_dim: int,
    device: torch.device
) -> OutcomeOnlyModel:
    """
    Create an OutcomeOnlyModel from architecture config.

    Args:
        arch_config: ModelArchitectureConfig instance
        representation_dim: Dimension for representation layers
        hidden_dim: Dimension for outcome hidden layers
        device: PyTorch device

    Returns:
        OutcomeOnlyModel instance
    """
    feature_extractor_type = getattr(arch_config, 'feature_extractor_type', 'cnn')

    model = OutcomeOnlyModel(
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
        # Outcome network args
        representation_dim=representation_dim,
        hidden_dim=hidden_dim,
        device=str(device)
    )

    return model
