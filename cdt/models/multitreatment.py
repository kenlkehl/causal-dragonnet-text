# cdt/models/multitreatment.py
"""Multi-treatment DragonNet models for pretraining."""

import logging
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .feature_extractor import FeatureExtractor


logger = logging.getLogger(__name__)


class MultiTreatmentDragonNetInternal(nn.Module):
    """
    Multi-treatment DragonNet architecture.
    
    Takes confounder representations and predicts:
    - Treatment propensities (multi-class)
    - Outcomes for each treatment arm
    """
    
    def __init__(
        self,
        input_dim: int,
        num_treatments: int,
        representation_dim: int = 200,
        hidden_outcome_dim: int = 100
    ):
        """
        Initialize multi-treatment DragonNet.
        
        Args:
            input_dim: Dimension of input confounder features
            num_treatments: Number of treatment options
            representation_dim: Dimension of shared representation
            hidden_outcome_dim: Dimension of outcome prediction hidden layers
        """
        super().__init__()
        
        self.num_treatments = num_treatments
        self.representation_dim = representation_dim
        
        # Shared representation layers
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc3 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc4 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc5 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc6 = nn.Linear(representation_dim, representation_dim)
        
        # Treatment propensity head (multi-class classification)
        self.propensity_fc1 = nn.Linear(representation_dim, representation_dim)
        self.propensity_fc2 = nn.Linear(representation_dim, representation_dim)
        self.propensity_fc3 = nn.Linear(representation_dim, representation_dim)
        self.propensity_fc4 = nn.Linear(representation_dim, num_treatments)
        
        # Outcome heads - one for each treatment
        self.outcome_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(representation_dim, hidden_outcome_dim),
                nn.ReLU(),
                nn.Linear(hidden_outcome_dim, hidden_outcome_dim),
                                nn.ReLU(),
                nn.Linear(hidden_outcome_dim, hidden_outcome_dim),
                                nn.ReLU(),
                nn.Linear(hidden_outcome_dim, hidden_outcome_dim),
                                nn.ReLU(),
                nn.Linear(hidden_outcome_dim, hidden_outcome_dim),
                nn.ELU(),
                nn.Linear(hidden_outcome_dim, 1)
            )
            for _ in range(num_treatments)
        ])
    
    def forward(
        self,
        x_representation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_representation: Confounder features (batch, input_dim)
        
        Returns:
            outcome_logits: (batch, num_treatments) - outcome predictions for each treatment
            treatment_logits: (batch, num_treatments) - treatment propensity logits
            phi: (batch, representation_dim) - learned representation
        """
        # Shared representation
        phi = F.relu(self.representation_fc1(x_representation))
        phi = F.relu(self.representation_fc2(phi))
        phi = F.relu(self.representation_fc3(phi))
        phi = F.relu(self.representation_fc4(phi))
        phi = F.relu(self.representation_fc5(phi))
        phi = F.elu(self.representation_fc6(phi))
        
        # Treatment propensity
        treatment = F.relu(self.propensity_fc1(phi))
        treatment = F.relu(self.propensity_fc2(treatment))
        treatment = F.relu(self.propensity_fc3(treatment))
        treatment_logits = self.propensity_fc4(treatment)
        
        # Outcome predictions for each treatment
        outcome_logits = torch.stack([
            head(phi).squeeze(-1) for head in self.outcome_heads
        ], dim=1)
        
        return outcome_logits, treatment_logits, phi


class MultiTreatmentDragonnetText(nn.Module):
    """
    Multi-treatment DragonNet for clinical text.
    
    Now a proper nn.Module that integrates:
    - Text embedding (via SentenceTransformer)
    - Confounder extraction (FeatureExtractor)
    - Multi-treatment causal inference (MultiTreatmentDragonNetInternal)
    """
    
    def __init__(
        self,
        num_treatments: int,
        sentence_transformer_model_name: str = "all-MiniLM-L6-v2",
        num_latent_confounders: int = 20,
        features_per_confounder: int = 1,
        explicit_confounder_texts: Optional[List[str]] = None,
        aggregator_mode: str = "attn",
        dragonnet_representation_dim: int = 128,
        dragonnet_hidden_outcome_dim: int = 64,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        device: str = "cuda:0"
    ):
        """
        Initialize multi-treatment DragonNet for text.
        
        Args:
            num_treatments: Number of treatment options
            sentence_transformer_model_name: Name of sentence transformer model
            num_latent_confounders: Number of learnable confounder patterns
            features_per_confounder: Features extracted per confounder
            explicit_confounder_texts: Optional list of explicit confounder queries
            aggregator_mode: Aggregation mode for chunks
            dragonnet_representation_dim: DragonNet representation dimension
            dragonnet_hidden_outcome_dim: DragonNet outcome hidden dimension
            chunk_size: Text chunk size in words
            chunk_overlap: Overlap between chunks
            device: Device string
        """
        super().__init__()
        
        self.num_treatments = num_treatments
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._device = torch.device(device)
        
        # Store config for checkpointing
        self.config = {
            'num_treatments': num_treatments,
            'sentence_transformer_model_name': sentence_transformer_model_name,
            'num_latent_confounders': num_latent_confounders,
            'features_per_confounder': features_per_confounder,
            'explicit_confounder_texts': explicit_confounder_texts,
            'aggregator_mode': aggregator_mode,
            'dragonnet_representation_dim': dragonnet_representation_dim,
            'dragonnet_hidden_outcome_dim': dragonnet_hidden_outcome_dim,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
        }
        
        # Load sentence transformer (not registered as a submodule)
        # This is stored as an attribute but not part of the PyTorch model graph
        # because SentenceTransformer has its own internal structure
        self.sentence_transformer_model = SentenceTransformer(
            sentence_transformer_model_name
        )
        self.sentence_transformer_model.to(self._device)
        self.embedding_dim = self.sentence_transformer_model.get_sentence_embedding_dimension()
        
        # Feature extractor - registered as submodule
        self.feature_extractor = FeatureExtractor(
            embedding_dim=self.embedding_dim,
            num_latent_confounders=num_latent_confounders,
            explicit_confounder_texts=explicit_confounder_texts,
            features_per_confounder=features_per_confounder,
            aggregator_mode=aggregator_mode,
            sentence_transformer_model=self.sentence_transformer_model,
            phantom_confounders=0,
            device=self._device
        )
        
        # Multi-treatment DragonNet - registered as submodule
        self.dragonnet = MultiTreatmentDragonNetInternal(
            input_dim=self.feature_extractor.output_dim,
            num_treatments=num_treatments,
            representation_dim=dragonnet_representation_dim,
            hidden_outcome_dim=dragonnet_hidden_outcome_dim
        )
        
        # Move to device
        self.to(self._device)
        
        logger.info(f"MultiTreatmentDragonnetText initialized:")
        logger.info(f"  Treatments: {num_treatments}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Confounders: {num_latent_confounders}")
        logger.info(f"  Feature extractor output: {self.feature_extractor.output_dim}")
        logger.info(f"  Device: {self._device}")
    
    def forward(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            chunk_embeddings_list: List of chunk embeddings, each (num_chunks, embed_dim)
        
        Returns:
            outcome_logits: (batch, num_treatments)
            treatment_logits: (batch, num_treatments)
            phi: (batch, representation_dim)
        """
        # Extract confounder features from chunks
        x_representation = self.feature_extractor(chunk_embeddings_list)
        
        # Pass through DragonNet
        outcome_logits, treatment_logits, phi = self.dragonnet(x_representation)
        
        return outcome_logits, treatment_logits, phi
    
    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        beta_targreg: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.
        
        Args:
            batch: Batch dictionary with chunk_embeddings, treatment, outcome
            alpha_propensity: Weight for propensity loss
            beta_targreg: Weight for targeted regularization
        
        Returns:
            Dictionary with loss components
        """
        chunk_embeddings_list = batch['chunk_embeddings']
        treatments = batch['treatment'].long()  # (batch,) - treatment indices
        outcomes = batch['outcome']  # (batch,)
        batch_size = treatments.size(0)
        
        # Forward pass
        outcome_logits, treatment_logits, phi = self.forward(chunk_embeddings_list)
        
        # Propensity loss (multi-class cross-entropy)
        propensity_loss = F.cross_entropy(treatment_logits, treatments)
        
        # Outcome loss - only for observed treatment
        # Gather the outcome prediction for each sample's actual treatment
        observed_outcome_logits = outcome_logits[
            torch.arange(batch_size, device=self._device),
            treatments
        ]
        outcome_loss = F.binary_cross_entropy_with_logits(
            observed_outcome_logits,
            outcomes
        )
        
        # Targeted regularization
        # Encourages separation between potential outcomes
        if beta_targreg > 0:
            # For each sample, get predictions for all treatments
            all_preds = torch.sigmoid(outcome_logits)  # (batch, num_treatments)
            
            # Compute variance across treatments as regularization
            targreg_loss = -torch.mean(torch.var(all_preds, dim=1))
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
            'targreg_loss': targreg_loss.detach() if isinstance(targreg_loss, torch.Tensor) else targreg_loss
        }
    
    def predict(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for inference.
        
        Args:
            chunk_embeddings_list: List of chunk embeddings
        
        Returns:
            Dictionary with predictions
        """
        with torch.no_grad():
            outcome_logits, treatment_logits, phi = self.forward(chunk_embeddings_list)
            
            # Convert to probabilities
            outcome_probs = torch.sigmoid(outcome_logits)  # (batch, num_treatments)
            treatment_probs = F.softmax(treatment_logits, dim=1)  # (batch, num_treatments)
            
            return {
                'outcome_probs': outcome_probs,
                'outcome_logits': outcome_logits,
                'treatment_probs': treatment_probs,
                'treatment_logits': treatment_logits,
                'representation': phi
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
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state
            epoch: Optional epoch number
            metrics: Optional training metrics
        """
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'feature_extractor': self.feature_extractor.state_dict(),
            'dragonnet': self.dragonnet.state_dict(),
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
    ) -> 'MultiTreatmentDragonnetText':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Optional device override
        
        Returns:
            Loaded model instance
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
                model.dragonnet.load_state_dict(
                    checkpoint['dragonnet'],
                    strict=False
                )
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def get_representation(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract learned representations (phi).
        
        Args:
            chunk_embeddings_list: List of chunk embeddings
        
        Returns:
            Representation tensor (batch, representation_dim)
        """
        with torch.no_grad():
            x_representation = self.feature_extractor(chunk_embeddings_list)
            _, _, phi = self.dragonnet(x_representation)
            return phi
    
    def get_confounder_features(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract confounder features (before DragonNet).
        
        Args:
            chunk_embeddings_list: List of chunk embeddings
        
        Returns:
            Confounder features (batch, feature_extractor_output_dim)
        """
        with torch.no_grad():
            return self.feature_extractor(chunk_embeddings_list)
