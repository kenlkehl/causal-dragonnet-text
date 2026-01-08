# cdt/models/causal_dragonnet_token.py
"""Token-level causal inference model with frozen backbone and interpretable confounder projection."""

import logging
from typing import Optional, List, Dict, Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_level_extractor import TokenLevelFeatureExtractor
from .dragonnet import DragonNet
from .uplift import UpliftNet


logger = logging.getLogger(__name__)


class CausalDragonnetTokenLevel(nn.Module):
    """
    Token-level DragonNet for clinical text with interpretable confounders.

    Key differences from CausalDragonnetText:
    1. Uses frozen transformer backbone (no fine-tuning of embeddings)
    2. Trainable projection layer initialized with confounder text embeddings
    3. Each output dimension has semantic meaning (maps to a confounder)
    4. Operates on tokenized text directly (not pre-computed chunk embeddings)

    Architecture:
        text -> tokenize -> frozen_backbone -> (B,T,D) token embeddings
             -> confounder_projection -> (B,T,C) token-confounder scores
             -> aggregation -> (B,C) confounder features
             -> optional MLP -> (B, projection_dim)
             -> DragonNet/UpliftNet -> causal predictions
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        explicit_confounder_texts: Optional[List[str]] = None,
        num_latent_confounders: int = 20,
        aggregation_method: Literal["max", "mean", "attention", "topk"] = "attention",
        topk: int = 10000,
        anchor_strength: float = 0.0,
        dragonnet_representation_dim: int = 128,
        dragonnet_hidden_outcome_dim: int = 64,
        max_length: int = 10000,
        model_type: str = "dragonnet",
        device: str = "cuda:0",
    ):
        """
        Initialize token-level causal model.

        Args:
            model_name: HuggingFace model name for backbone
            explicit_confounder_texts: List of confounder descriptions
            num_latent_confounders: Number of learnable latent confounders
            aggregation_method: How to aggregate token scores ("max", "mean", "attention", "topk")
            topk: K value for topk aggregation
            anchor_strength: Regularization to keep projection near initial embeddings
            dragonnet_representation_dim: DragonNet shared representation dimension
            dragonnet_hidden_outcome_dim: DragonNet outcome head hidden dimension
            max_length: Maximum sequence length for tokenization (default 10000)
            model_type: "dragonnet" or "uplift"
            device: Device string
        """
        super().__init__()

        self._device = torch.device(device)
        self.model_type = model_type
        self.anchor_strength = anchor_strength
        self.max_length = max_length

        # Store config for checkpointing
        self.config = {
            'model_name': model_name,
            'explicit_confounder_texts': explicit_confounder_texts,
            'num_latent_confounders': num_latent_confounders,
            'aggregation_method': aggregation_method,
            'topk': topk,
            'anchor_strength': anchor_strength,
            'dragonnet_representation_dim': dragonnet_representation_dim,
            'dragonnet_hidden_outcome_dim': dragonnet_hidden_outcome_dim,
            'max_length': max_length,
            'model_type': model_type,
        }

        # Token-level feature extractor
        self.feature_extractor = TokenLevelFeatureExtractor(
            model_name=model_name,
            explicit_confounder_texts=explicit_confounder_texts,
            num_latent_confounders=num_latent_confounders,
            aggregation_method=aggregation_method,
            topk=topk,
            anchor_strength=anchor_strength,
            projection_dim=dragonnet_representation_dim,
            max_length=max_length,
            device=self._device,
        )

        # Causal inference network
        if model_type == "uplift":
            self.net = UpliftNet(
                input_dim=self.feature_extractor.output_dim,
                representation_dim=dragonnet_representation_dim,
                hidden_outcome_dim=dragonnet_hidden_outcome_dim,
            )
            logger.info("Using UpliftNet architecture")
        else:
            self.net = DragonNet(
                input_dim=self.feature_extractor.output_dim,
                representation_dim=dragonnet_representation_dim,
                hidden_outcome_dim=dragonnet_hidden_outcome_dim,
            )
            logger.info("Using DragonNet architecture")

        # Alias for compatibility
        self.dragonnet = self.net

        # Move to device
        self.to(self._device)

        logger.info(f"CausalDragonnetTokenLevel initialized:")
        logger.info(f"  Backbone: {model_name}")
        logger.info(f"  Explicit confounders: {len(explicit_confounder_texts or [])}")
        logger.info(f"  Latent confounders: {num_latent_confounders}")
        logger.info(f"  Feature extractor output: {self.feature_extractor.output_dim}")
        logger.info(f"  Aggregation: {aggregation_method}")
        logger.info(f"  Device: {self._device}")

    @property
    def tokenizer(self):
        """Access to tokenizer for external use."""
        return self.feature_extractor.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            input_ids: Token IDs (B, T)
            attention_mask: Attention mask (B, T)

        Returns:
            y0_logit: (B, 1) outcome prediction under control
            y1_logit: (B, 1) outcome prediction under treatment
            t_logit: (B, 1) treatment propensity logit
            phi: (B, representation_dim) shared representation
        """
        # Extract confounder features
        confounder_features = self.feature_extractor(input_ids, attention_mask)

        # Causal predictions
        if self.model_type == "uplift":
            y0_logit, tau_logit, t_logit, phi = self.net(confounder_features)
            y1_logit = y0_logit + tau_logit
        else:
            y0_logit, y1_logit, t_logit, phi = self.net(confounder_features)

        return y0_logit, y1_logit, t_logit, phi

    def forward_from_text(
        self,
        texts: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass from raw text.

        Args:
            texts: List of text strings

        Returns:
            Same as forward()
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        return self.forward(input_ids, attention_mask)

    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        beta_targreg: float = 0.1,
        outcome_type: str = "binary",
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with input_ids, attention_mask, treatment, outcome
            alpha_propensity: Weight for propensity loss
            beta_targreg: Weight for targeted regularization
            outcome_type: "binary" or "continuous"

        Returns:
            Dictionary with loss components and predictions
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        treatments = batch['treatment']
        outcomes = batch['outcome']

        # Forward pass
        y0_logit, y1_logit, t_logit, phi = self.forward(input_ids, attention_mask)

        # Propensity loss
        propensity_loss = F.binary_cross_entropy_with_logits(
            t_logit.squeeze(-1),
            treatments
        )

        # Outcome loss (factual only)
        factual_pred = torch.where(
            treatments.unsqueeze(1) > 0.5,
            y1_logit,
            y0_logit
        )

        if outcome_type == "continuous":
            outcome_loss = F.mse_loss(factual_pred.squeeze(-1), outcomes)
        else:
            outcome_loss = F.binary_cross_entropy_with_logits(
                factual_pred.squeeze(-1),
                outcomes
            )

        # Targeted regularization
        if beta_targreg > 0:
            with torch.no_grad():
                propensity = torch.sigmoid(t_logit).clamp(1e-3, 1 - 1e-3)
                H = (treatments.unsqueeze(1) / propensity) - \
                    ((1 - treatments.unsqueeze(1)) / (1 - propensity))

            if outcome_type == "continuous":
                residual = outcomes.unsqueeze(1) - factual_pred
                moment = torch.mean(residual * H)
            else:
                factual_prob = torch.sigmoid(factual_pred)
                moment = torch.mean((outcomes.unsqueeze(1) - factual_prob) * H)

            targreg_loss = moment ** 2
        else:
            targreg_loss = torch.tensor(0.0, device=self._device)

        # Anchor regularization (keep projection interpretable)
        anchor_loss = self.feature_extractor.anchor_loss()

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            beta_targreg * targreg_loss +
            anchor_loss
        )

        return {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'targreg_loss': targreg_loss.detach() if isinstance(targreg_loss, torch.Tensor) else targreg_loss,
            'anchor_loss': anchor_loss.detach(),
            'y0_logit': y0_logit.detach(),
            'y1_logit': y1_logit.detach(),
            't_logit': t_logit.detach(),
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        outcome_type: str = "binary",
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for inference.

        Args:
            input_ids: Token IDs (B, T)
            attention_mask: Attention mask (B, T)
            outcome_type: "binary" or "continuous"

        Returns:
            Dictionary with predictions
        """
        with torch.no_grad():
            y0_logit, y1_logit, t_logit, phi = self.forward(input_ids, attention_mask)

            propensity = torch.sigmoid(t_logit).squeeze(-1)
            tau_pred = (y1_logit - y0_logit).squeeze(-1)

            if outcome_type == "continuous":
                return {
                    'y0_pred': y0_logit.squeeze(-1),
                    'y1_pred': y1_logit.squeeze(-1),
                    'propensity': propensity,
                    'tau_pred': tau_pred,
                    'phi': phi,
                }
            else:
                y0_prob = torch.sigmoid(y0_logit).squeeze(-1)
                y1_prob = torch.sigmoid(y1_logit).squeeze(-1)

                return {
                    'y0_prob': y0_prob,
                    'y1_prob': y1_prob,
                    'propensity': propensity,
                    'tau_pred': tau_pred,
                    'phi': phi,
                }

    def predict_from_text(
        self,
        texts: List[str],
        outcome_type: str = "binary",
    ) -> Dict[str, torch.Tensor]:
        """Predict from raw text."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        return self.predict(input_ids, attention_mask, outcome_type)

    def get_token_attributions(
        self,
        texts: List[str],
        confounder_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get token-level attributions for interpretability.

        Args:
            texts: List of text strings
            confounder_idx: If specified, return scores for this confounder only

        Returns:
            Dictionary with:
            - 'tokens': List of token strings per sample
            - 'scores': (B, T, C) or (B, T) token-confounder scores
            - 'confounder_names': List of confounder names
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        # Get token-level scores
        scores = self.feature_extractor.get_token_confounder_scores(
            input_ids, attention_mask
        )

        # Decode tokens
        tokens = [
            self.tokenizer.convert_ids_to_tokens(ids)
            for ids in input_ids.cpu().tolist()
        ]

        if confounder_idx is not None:
            scores = scores[:, :, confounder_idx]

        return {
            'tokens': tokens,
            'scores': scores.cpu(),
            'attention_mask': attention_mask.cpu(),
            'confounder_names': self.feature_extractor.get_confounder_names(),
        }

    def get_confounder_drift(self) -> Dict[str, Any]:
        """Get interpretability metrics for confounder drift."""
        return self.feature_extractor.confounder_drift()

    def interpret_latent_confounders(
        self,
        candidate_concepts: List[str],
        top_k: int = 5,
    ) -> List[List[tuple]]:
        """Find nearest concepts for latent confounders."""
        return self.feature_extractor.interpret_latent_confounders(
            candidate_concepts, top_k
        )

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'feature_extractor': self.feature_extractor.state_dict(),
            'net': self.net.state_dict(),
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
        device: Optional[str] = None,
    ) -> 'CausalDragonnetTokenLevel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        if device is not None:
            config['device'] = device

        model = cls(**config)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        logger.info(f"Model loaded from {path}")
        return model
