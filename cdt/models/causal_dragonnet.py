# cdt/models/causal_dragonnet.py
"""Binary treatment causal inference model combining text processing and DragonNet."""

import logging
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .feature_extractor import FeatureExtractor
from .dragonnet import DragonNet
from .uplift import UpliftNet


logger = logging.getLogger(__name__)


class CausalDragonnetText(nn.Module):
    """
    Binary treatment DragonNet for clinical text.
    
    Proper nn.Module that integrates:
    - Text embedding (via SentenceTransformer)
    - Confounder extraction (FeatureExtractor)
    - Binary treatment causal inference (DragonNet or UpliftNet)
    """
    
    def __init__(
        self,
        sentence_transformer_model_name: str = 'all-MiniLM-L6-v2',
        num_latent_confounders: int = 20,
        explicit_confounder_texts: Optional[List[str]] = None,
        explicit_confounder_embeddings: Optional[torch.Tensor] = None,
        value_dim: int = 128,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        dragonnet_representation_dim: int = 128,
        dragonnet_hidden_outcome_dim: int = 64,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        device: str = "cuda:0",
        model_type: str = "dragonnet"  # "dragonnet" or "uplift"
    ):
        """
        Initialize binary treatment DragonNet for text.

        Args:
            sentence_transformer_model_name: Name of sentence transformer model
            num_latent_confounders: Number of learnable confounder patterns
            explicit_confounder_texts: Optional list of explicit confounder queries
            explicit_confounder_embeddings: Optional pre-computed embeddings
            value_dim: Output dimension per confounder in cross-attention
            num_attention_heads: Number of attention heads per confounder
            attention_dropout: Dropout rate on attention weights
            dragonnet_representation_dim: DragonNet representation dimension
            dragonnet_hidden_outcome_dim: DragonNet outcome hidden dimension
            chunk_size: Text chunk size in words
            chunk_overlap: Overlap between chunks
            device: Device string
            model_type: Architecture type ("dragonnet" or "uplift")
        """
        super().__init__()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._device = torch.device(device)
        self.model_type = model_type

        # Store config for checkpointing
        self.config = {
            'sentence_transformer_model_name': sentence_transformer_model_name,
            'num_latent_confounders': num_latent_confounders,
            'explicit_confounder_texts': explicit_confounder_texts,
            'value_dim': value_dim,
            'num_attention_heads': num_attention_heads,
            'attention_dropout': attention_dropout,
            'dragonnet_representation_dim': dragonnet_representation_dim,
            'dragonnet_hidden_outcome_dim': dragonnet_hidden_outcome_dim,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'model_type': model_type
        }
        
        # Load sentence transformer (not a PyTorch submodule)
        # Pass device directly to avoid initial allocation on cuda:0
        self.sentence_transformer_model = SentenceTransformer(
            sentence_transformer_model_name,
            device=self._device
        )
        self.embedding_dim = self.sentence_transformer_model.get_sentence_embedding_dimension()
        
        # Feature extractor - registered as submodule
        self.feature_extractor = FeatureExtractor(
            embedding_dim=self.embedding_dim,
            num_latent_confounders=num_latent_confounders,
            explicit_confounder_texts=explicit_confounder_texts,
            value_dim=value_dim,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            projection_dim=dragonnet_representation_dim,  # Match DragonNet input
            sentence_transformer_model=self.sentence_transformer_model,
            phantom_confounders=0,
            device=self._device,
            explicit_confounder_embeddings=explicit_confounder_embeddings
        )
        
        # Binary treatment Causal Inference Net
        if model_type == "uplift":
            self.net = UpliftNet(
                input_dim=self.feature_extractor.output_dim,
                representation_dim=dragonnet_representation_dim,
                hidden_outcome_dim=dragonnet_hidden_outcome_dim
            )
            logger.info("Using UpliftNet architecture (Base + ITE parametrization)")
        else:
            self.net = DragonNet(
                input_dim=self.feature_extractor.output_dim,
                representation_dim=dragonnet_representation_dim,
                hidden_outcome_dim=dragonnet_hidden_outcome_dim
            )
            logger.info("Using classic DragonNet architecture")

        # For backward compatibility, alias self.net to self.dragonnet
        # (though strictly it might be an UpliftNet now)
        self.dragonnet = self.net
        
        # Move to device
        self.to(self._device)
        
        logger.info(f"CausalDragonnetText initialized:")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Confounders: {num_latent_confounders}")
        logger.info(f"  Feature extractor output: {self.feature_extractor.output_dim}")
        logger.info(f"  Device: {self._device}")
    
    def forward(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Returns:
            y0_logit: (batch, 1) - outcome prediction under control
            y1_logit: (batch, 1) - outcome prediction under treatment
            t_logit: (batch, 1) - treatment propensity logit
            final_common_layer: (batch, representation_dim) - shared representation before outcome heads
        """
        # Extract confounder features from chunks
        confounder_features = self.feature_extractor(chunk_embeddings_list)
        
        if self.model_type == "uplift":
            # UpliftNet returns: y0_logit, tau_logit, t_logit, final_common_layer
            y0_logit, tau_logit, t_logit, final_common_layer = self.net(confounder_features)
            # Reconstruct y1_logit = y0_logit + tau_logit
            y1_logit = y0_logit + tau_logit
        else:
            # DragonNet returns: y0_logit, y1_logit, t_logit, final_common_layer
            y0_logit, y1_logit, t_logit, final_common_layer = self.net(confounder_features)
        
        return y0_logit, y1_logit, t_logit, final_common_layer
    
    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        beta_targreg: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.
        
        Returns:
            Dictionary with loss components and detached predictions
        """
        chunk_embeddings_list = batch['chunk_embeddings']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        
        # Forward pass
        y0_logit, y1_logit, t_logit, phi = self.forward(chunk_embeddings_list)
        
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
                # Compute H = t/e(x) - (1-t)/(1-e(x))
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
            # Return predictions for AUROC calculation
            'y0_logit': y0_logit.detach(),
            'y1_logit': y1_logit.detach(),
            't_logit': t_logit.detach()
        }
    
    def predict(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for inference.
        """
        with torch.no_grad():
            confounder_features = self.feature_extractor(chunk_embeddings_list)
            
            if self.model_type == "uplift":
                y0_logit, tau_logit, t_logit, final_common_layer = self.net(confounder_features)
                y1_logit = y0_logit + tau_logit
                
                # In uplift mode, we can return tau directly
                tau_pred = tau_logit.squeeze(-1)  # If linear activation, this IS the ITE on logit scale
            else:
                y0_logit, y1_logit, t_logit, final_common_layer = self.net(confounder_features)
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
                'tau_pred': tau_pred # Added explicit tau return
            }
    
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
            # Legacy component keys for backward compat tools
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
    ) -> 'CausalDragonnetText':
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
            # Legacy format - load components separately
            if 'feature_extractor' in checkpoint:
                model.feature_extractor.load_state_dict(
                    checkpoint['feature_extractor'],
                    strict=False
                )
            if 'dragonnet' in checkpoint:
                # This works because we alias self.net to self.dragonnet
                model.net.load_state_dict(
                    checkpoint['dragonnet'],
                    strict=False
                )
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def get_final_common_layer(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract the final common layer (shared representation before outcome heads).
        """
        with torch.no_grad():
            confounder_features = self.feature_extractor(chunk_embeddings_list)
            
            # Both models return final_common_layer as the last element
            if self.model_type == "uplift":
                _, _, _, final_common_layer = self.net(confounder_features)
            else:
                _, _, _, final_common_layer = self.net(confounder_features)
                
            return final_common_layer
    
    def get_confounder_features(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract confounder features (before DragonNet/UpliftNet representation layers).

        This is the raw output of FeatureExtractor with shape:
            (batch, projection_dim) where projection_dim = dragonnet_representation_dim
        """
        with torch.no_grad():
            confounder_features = self.feature_extractor(chunk_embeddings_list)
            return confounder_features
