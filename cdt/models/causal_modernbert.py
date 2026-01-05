# cdt/models/causal_modernbert.py
"""Causal inference model using ModernBERT for text representation."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modernbert_extractor import ModernBertFeatureExtractor
from .dragonnet import DragonNet
from .uplift import UpliftNet


logger = logging.getLogger(__name__)


class CausalModernBertText(nn.Module):
    """
    Causal inference model using ModernBERT CLS embedding.
    
    Simplified architecture:
    - ModernBERT encodes full text into CLS token embedding
    - DragonNet/UpliftNet predicts outcomes and propensity
    
    No chunking or confounder extraction - direct end-to-end learning.
    """
    
    def __init__(
        self,
        modernbert_model_name: str = "answerdotai/ModernBERT-base",
        projection_dim: Optional[int] = 128,
        freeze_bert: bool = False,
        max_length: int = 8192,
        dragonnet_representation_dim: int = 128,
        dragonnet_hidden_outcome_dim: int = 64,
        device: str = "cuda:0",
        model_type: str = "dragonnet"  # "dragonnet" or "uplift"
    ):
        """
        Initialize causal inference model with ModernBERT.
        
        Args:
            modernbert_model_name: HuggingFace model name for ModernBERT
            projection_dim: Dimension to project CLS embedding to (should match dragonnet input)
            freeze_bert: If True, freeze ModernBERT weights
            max_length: Maximum sequence length for tokenization
            dragonnet_representation_dim: DragonNet representation dimension
            dragonnet_hidden_outcome_dim: DragonNet outcome hidden dimension
            device: Device string
            model_type: Architecture type ("dragonnet" or "uplift")
        """
        super().__init__()
        
        self._device = torch.device(device)
        self.model_type = model_type
        
        # Store config for checkpointing
        self.config = {
            'modernbert_model_name': modernbert_model_name,
            'projection_dim': projection_dim,
            'freeze_bert': freeze_bert,
            'max_length': max_length,
            'dragonnet_representation_dim': dragonnet_representation_dim,
            'dragonnet_hidden_outcome_dim': dragonnet_hidden_outcome_dim,
            'model_type': model_type
        }
        
        # Feature extractor using ModernBERT
        self.feature_extractor = ModernBertFeatureExtractor(
            model_name=modernbert_model_name,
            projection_dim=projection_dim,
            freeze_bert=freeze_bert,
            max_length=max_length,
            device=self._device
        )
        
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
        
        logger.info(f"CausalModernBertText initialized:")
        logger.info(f"  ModernBERT model: {modernbert_model_name}")
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
            y0_logit: (batch, 1) - outcome prediction under control
            y1_logit: (batch, 1) - outcome prediction under treatment
            t_logit: (batch, 1) - treatment propensity logit
            final_common_layer: (batch, representation_dim) - shared representation
        """
        # Extract CLS features from texts
        features = self.feature_extractor(texts)
        
        if self.model_type == "uplift":
            # UpliftNet returns: y0_logit, tau_logit, t_logit, final_common_layer
            y0_logit, tau_logit, t_logit, final_common_layer = self.net(features)
            # Reconstruct y1_logit = y0_logit + tau_logit
            y1_logit = y0_logit + tau_logit
        else:
            # DragonNet returns: y0_logit, y1_logit, t_logit, final_common_layer
            y0_logit, y1_logit, t_logit, final_common_layer = self.net(features)
        
        return y0_logit, y1_logit, t_logit, final_common_layer
    
    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        beta_targreg: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.
        
        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys
            alpha_propensity: Weight for propensity loss
            beta_targreg: Weight for targeted regularization
        
        Returns:
            Dictionary with loss components and detached predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        
        # Forward pass
        y0_logit, y1_logit, t_logit, phi = self.forward(texts)
        
        # Propensity loss
        propensity_loss = F.binary_cross_entropy_with_logits(
            t_logit.squeeze(-1),
            treatments
        )
        
        # Outcome loss - factual outcome only
        factual_logit = torch.where(
            treatments.unsqueeze(1) > 0.5,
            y1_logit,
            y0_logit
        )
        
        outcome_loss = F.binary_cross_entropy_with_logits(
            factual_logit.squeeze(-1),
            outcomes
        )
        
        # Targeted regularization (R-loss)
        if beta_targreg > 0:
            with torch.no_grad():
                propensity = torch.sigmoid(t_logit).clamp(1e-3, 1 - 1e-3)
                H = (treatments.unsqueeze(1) / propensity) - \
                    ((1 - treatments.unsqueeze(1)) / (1 - propensity))
            
            factual_prob = torch.sigmoid(factual_logit)
            moment = torch.mean((outcomes.unsqueeze(1) - factual_prob) * H)
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
            'y0_logit': y0_logit.detach(),
            'y1_logit': y1_logit.detach(),
            't_logit': t_logit.detach()
        }
    
    def predict(
        self,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for inference.
        
        Args:
            texts: List of text strings
        
        Returns:
            Dictionary with prediction outputs
        """
        with torch.no_grad():
            features = self.feature_extractor(texts)
            
            if self.model_type == "uplift":
                y0_logit, tau_logit, t_logit, final_common_layer = self.net(features)
                y1_logit = y0_logit + tau_logit
                tau_pred = tau_logit.squeeze(-1)
            else:
                y0_logit, y1_logit, t_logit, final_common_layer = self.net(features)
                tau_pred = (y1_logit - y0_logit).squeeze(-1)
            
            # Convert to probabilities
            y0_prob = torch.sigmoid(y0_logit).squeeze(-1)
            y1_prob = torch.sigmoid(y1_logit).squeeze(-1)
            propensity = torch.sigmoid(t_logit).squeeze(-1)
            
            return {
                'y0_prob': y0_prob,
                'y1_prob': y1_prob,
                'propensity': propensity,
                'y0_logit': y0_logit.squeeze(-1),
                'y1_logit': y1_logit.squeeze(-1),
                't_logit': t_logit.squeeze(-1),
                'final_common_layer': final_common_layer,
                'tau_pred': tau_pred
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
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint.
        """
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'feature_extractor': self.feature_extractor.state_dict(),
            'dragonnet': self.net.state_dict(),
        }
        
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
    ) -> 'CausalModernBertText':
        """
        Load model from checkpoint.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        if device is not None:
            config['device'] = device
        
        # Create model
        model = cls(**config)
        
        # Load state dict
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
