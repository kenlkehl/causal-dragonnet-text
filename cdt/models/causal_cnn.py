# cdt/models/causal_cnn.py
"""Causal inference model using simple 1D CNN for text representation."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_extractor import CNNFeatureExtractor
from .bert_extractor import BertFeatureExtractor
from .dragonnet import DragonNet
from .uplift import UpliftNet


logger = logging.getLogger(__name__)


class CausalCNNText(nn.Module):
    """
    Causal inference model for text using CNN or BERT feature extraction.

    Architecture:
    - Feature extractor (CNN or BERT) encodes text into feature vector
    - DragonNet/UpliftNet predicts outcomes and propensity

    CNN mode:
    - 1D CNN with word-level tokenization
    - Much faster to train than transformers
    - IMPORTANT: Call fit_tokenizer(texts) with training data before use

    BERT mode:
    - HuggingFace transformer with CLS token extraction
    - Fine-tuning or frozen encoder options
    - No fit_tokenizer() needed (uses pretrained tokenizer)
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
        # DragonNet args
        dragonnet_representation_dim: int = 128,
        dragonnet_hidden_outcome_dim: int = 64,
        device: str = "cuda:0",
        model_type: str = "dragonnet"  # "dragonnet" or "uplift"
    ):
        """
        Initialize causal inference model with CNN or BERT feature extractor.

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
            dragonnet_representation_dim: DragonNet representation dimension
            dragonnet_hidden_outcome_dim: DragonNet outcome hidden dimension
            device: Device string
            model_type: Architecture type ("dragonnet" or "uplift")
        """
        super().__init__()

        self._device = torch.device(device)
        self.model_type = model_type
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
            'dragonnet_representation_dim': dragonnet_representation_dim,
            'dragonnet_hidden_outcome_dim': dragonnet_hidden_outcome_dim,
            'model_type': model_type
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
            logger.info(f"Using BERT feature extractor: {bert_model_name}")
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
            logger.info("Using CNN feature extractor")

        # Binary treatment Causal Inference Net
        input_dim = self.feature_extractor.output_dim

        if model_type == "uplift":
            self.net = UpliftNet(
                input_dim=input_dim,
                representation_dim=dragonnet_representation_dim,
                hidden_outcome_dim=dragonnet_hidden_outcome_dim
            )
            logger.info("Using UpliftNet architecture (Base + ITE parametrization)")
        else:
            self.net = DragonNet(
                input_dim=input_dim,
                representation_dim=dragonnet_representation_dim,
                hidden_outcome_dim=dragonnet_hidden_outcome_dim
            )
            logger.info("Using classic DragonNet architecture")

        # Alias for backward compatibility
        self.dragonnet = self.net

        # Move to device
        self.to(self._device)

        logger.info(f"CausalCNNText initialized:")
        logger.info(f"  Feature extractor: {feature_extractor_type}")
        logger.info(f"  Feature extractor output: {input_dim}")
        logger.info(f"  Device: {self._device}")

    def forward(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            texts: List of text strings

        Returns:
            y0_pred: (batch, 1) - predicted survival under control (positive, continuous)
            y1_pred: (batch, 1) - predicted survival under treatment (positive, continuous)
            t_logit: (batch, 1) - treatment propensity logit
            final_common_layer: (batch, representation_dim) - shared representation
        """
        # Extract features from texts using CNN
        features = self.feature_extractor(texts)

        if self.model_type == "uplift":
            # UpliftNet returns: y0_pred, tau, t_logit, final_common_layer
            y0_pred, tau, t_logit, final_common_layer = self.net(features)
            # Reconstruct y1_pred = y0_pred + tau
            y1_pred = y0_pred + tau
        else:
            # DragonNet returns: y0_pred, y1_pred, t_logit, final_common_layer
            y0_pred, y1_pred, t_logit, final_common_layer = self.net(features)

        return y0_pred, y1_pred, t_logit, final_common_layer

    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        beta_targreg: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step for continuous outcomes.

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys
            alpha_propensity: Weight for propensity loss
            beta_targreg: Weight for targeted regularization

        Returns:
            Dictionary with loss components and detached predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,) - continuous survival in months

        # Forward pass
        y0_pred, y1_pred, t_logit, phi = self.forward(texts)

        # Propensity loss (still binary cross-entropy for treatment)
        propensity_loss = F.binary_cross_entropy_with_logits(
            t_logit.squeeze(-1),
            treatments
        )

        # Outcome loss - MSE for continuous outcomes (factual only)
        factual_pred = torch.where(
            treatments.unsqueeze(1) > 0.5,
            y1_pred,
            y0_pred
        )

        outcome_loss = F.mse_loss(
            factual_pred.squeeze(-1),
            outcomes
        )

        # Targeted regularization (R-loss) for continuous outcomes
        if beta_targreg > 0:
            with torch.no_grad():
                propensity = torch.sigmoid(t_logit).clamp(1e-3, 1 - 1e-3)
                H = (treatments.unsqueeze(1) / propensity) - \
                    ((1 - treatments.unsqueeze(1)) / (1 - propensity))

            # Use raw predictions for continuous outcomes (no sigmoid)
            moment = torch.mean((outcomes.unsqueeze(1) - factual_pred) * H)
            targreg_loss = moment ** 2
        else:
            targreg_loss = torch.tensor(0.0, device=self._device)

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            beta_targreg * targreg_loss
        )

        return {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'targreg_loss': targreg_loss.detach() if isinstance(targreg_loss, torch.Tensor) else targreg_loss,
            'y0_pred': y0_pred.detach(),
            'y1_pred': y1_pred.detach(),
            't_logit': t_logit.detach()
        }

    def predict(
        self,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for inference with continuous outcomes.

        Args:
            texts: List of text strings

        Returns:
            Dictionary with prediction outputs:
            - y0_pred: Predicted survival under control (months)
            - y1_pred: Predicted survival under treatment (months)
            - ite: Individual treatment effect (y1 - y0, in months)
            - propensity: Treatment propensity probability
            - t_logit: Treatment propensity logit
            - final_common_layer: Shared representation
        """
        with torch.no_grad():
            features = self.feature_extractor(texts)

            if self.model_type == "uplift":
                y0_pred, tau, t_logit, final_common_layer = self.net(features)
                y1_pred = y0_pred + tau
                ite = tau.squeeze(-1)
            else:
                y0_pred, y1_pred, t_logit, final_common_layer = self.net(features)
                ite = (y1_pred - y0_pred).squeeze(-1)

            # Propensity is still binary (sigmoid)
            propensity = torch.sigmoid(t_logit).squeeze(-1)

            return {
                'y0_pred': y0_pred.squeeze(-1),  # Continuous survival prediction
                'y1_pred': y1_pred.squeeze(-1),  # Continuous survival prediction
                'ite': ite,  # Treatment effect in months
                'propensity': propensity,
                't_logit': t_logit.squeeze(-1),
                'final_common_layer': final_common_layer,
            }

    def get_features(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """
        Extract feature representations from texts.

        Args:
            texts: List of text strings

        Returns:
            Feature tensor: (batch, output_dim)
        """
        with torch.no_grad():
            return self.feature_extractor(texts)

    def fit_tokenizer(self, texts: List[str]) -> 'CausalCNNText':
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

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint including tokenizer state.
        """
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'feature_extractor': self.feature_extractor.state_dict(),
            'dragonnet': self.net.state_dict(),
            'feature_extractor_type': self.feature_extractor_type,
        }

        # Save tokenizer state for CNN, or extractor state for BERT
        if self.feature_extractor_type == "cnn":
            checkpoint['tokenizer_state'] = self.feature_extractor.get_tokenizer_state()
        else:
            checkpoint['extractor_state'] = self.feature_extractor.get_state()

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        device: Optional[str] = None
    ) -> 'CausalCNNText':
        """
        Load model from checkpoint including tokenizer state.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        if device is not None:
            config['device'] = device

        # Create model
        model = cls(**config)

        # Load tokenizer state for CNN (rebuilds embedding layer with correct vocab size)
        if model.feature_extractor_type == "cnn" and 'tokenizer_state' in checkpoint:
            model.feature_extractor.load_tokenizer_state(checkpoint['tokenizer_state'])

        # Load state dict (after tokenizer so embedding has correct size)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            if 'feature_extractor' in checkpoint:
                model.feature_extractor.load_state_dict(
                    checkpoint['feature_extractor'],
                    strict=False
                )
            if 'dragonnet' in checkpoint:
                model.net.load_state_dict(
                    checkpoint['dragonnet'],
                    strict=False
                )

        logger.info(f"Model loaded from {path}")
        return model

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
