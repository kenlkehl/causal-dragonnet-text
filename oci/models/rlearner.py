# oci/models/rlearner.py
"""R-Learner network for direct treatment effect optimization."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class RLearnerNet(nn.Module):
    """
    R-Learner Network with three heads for direct treatment effect optimization.

    Architecture:
    - Shared representation layers (same as DragonNet)
    - Propensity head: e(X) = P(T=1|X)
    - Marginal outcome head: m(X) = E[Y|X]
    - Treatment effect head: τ(X) = E[Y(1)-Y(0)|X]

    The τ head is trained with R-learner loss that directly optimizes
    treatment effect estimation by minimizing:
        L_R = E[(Y - m(X) - τ(X)(T - e(X)))^2]

    Key advantages over DragonNet:
    - Direct gradient signal to τ(X) from treatment effect loss
    - Nuisance functions (e, m) are detached in R-loss, preventing
      interference with effect estimation
    - τ(X) is unbounded (can be negative) - represents true effect

    References:
        Nie & Wager (2021). Quasi-oracle estimation of heterogeneous
        treatment effects. Biometrika.
    """

    def __init__(
        self,
        input_dim: int,
        representation_dim: int = 200,
        hidden_outcome_dim: int = 100,
        dropout: float = 0.2
    ):
        """
        Initialize R-Learner network.

        Args:
            input_dim: Dimension of input features from feature extractor
            representation_dim: Dimension of shared representation
            hidden_outcome_dim: Hidden dimension for outcome/effect heads
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout_rate = dropout

        # Shared representation layers (2 layers like simple DragonNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)
        self.rep_dropout = nn.Dropout(dropout)

        # Nuisance branch: hidden state W jointly supports e(X) and m(X).
        self.nuisance_fc = nn.Linear(representation_dim, hidden_outcome_dim)
        self.propensity_fc = nn.Linear(hidden_outcome_dim, 1)
        self.outcome_fc = nn.Linear(hidden_outcome_dim, 1)

        # Treatment effect branch: hidden state X supports τ(X).
        # Note: τ is unbounded (no final activation), can be positive or negative
        self.effect_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.effect_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.effect_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Dropout for outcome/effect heads
        self.outcome_dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor):
        """
        Forward pass through R-Learner network.

        Args:
            features: Output from feature extractor, shape (batch, input_dim)

        Returns:
            m_logit: Marginal outcome logit E[Y|X], shape (batch, 1)
            tau: Treatment effect τ(X), shape (batch, 1) - unbounded
            t_logit: Propensity logit, shape (batch, 1)
            phi: Shared representation, shape (batch, representation_dim)
        """
        # Shared representation
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        phi = F.elu(self.representation_fc2(h))
        phi = self.rep_dropout(phi)

        # Nuisance branch: W is the common hidden state for propensity/outcome.
        w_hidden = F.relu(self.nuisance_fc(phi))
        w_hidden = self.outcome_dropout(w_hidden)
        t_logit = self.propensity_fc(w_hidden)
        m_logit = self.outcome_fc(w_hidden)

        # Treatment effect head (no final activation - τ can be negative)
        x_hidden = F.relu(self.effect_fc1(phi))
        tau = self.outcome_dropout(x_hidden)
        tau = F.elu(self.effect_fc2(tau))
        tau = self.outcome_dropout(tau)
        tau_out = self.effect_fc3(tau)

        return m_logit, tau_out, t_logit, phi

    def forward_with_activations(self, features: torch.Tensor):
        """Forward pass returning role-specific W/X branch activations."""
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        phi = F.elu(self.representation_fc2(h))
        phi = self.rep_dropout(phi)

        w_hidden = F.relu(self.nuisance_fc(phi))
        w_hidden = self.outcome_dropout(w_hidden)
        t_logit = self.propensity_fc(w_hidden)
        m_logit = self.outcome_fc(w_hidden)

        x_hidden = F.relu(self.effect_fc1(phi))
        tau = self.outcome_dropout(x_hidden)
        tau = F.elu(self.effect_fc2(tau))
        tau = self.outcome_dropout(tau)
        tau_out = self.effect_fc3(tau)

        return m_logit, tau_out, t_logit, phi, w_hidden, x_hidden

    def get_representation(self, features):
        """Compute shared representation from input features."""
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        phi = F.elu(self.representation_fc2(h))
        phi = self.rep_dropout(phi)
        return phi

    def propensity_from_representation(self, phi):
        """Compute propensity logit from shared representation."""
        w_hidden = F.relu(self.nuisance_fc(phi))
        w_hidden = self.outcome_dropout(w_hidden)
        return self.propensity_fc(w_hidden)


class RoleGatedSlotRLearner(nn.Module):
    """R-learner head with separate nuisance and effect gates over slots."""

    def __init__(
        self,
        input_dim: int,
        num_slots: int,
        slot_feature_dim: int,
        representation_dim: int = 200,
        hidden_outcome_dim: int = 100,
        dropout: float = 0.2,
        gate_l1_weight: float = 0.0,
    ):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if slot_feature_dim < 1:
            raise ValueError("slot_feature_dim must be >= 1")

        self.num_slots = int(num_slots)
        self.slot_feature_dim = int(slot_feature_dim)
        self.slot_flat_dim = self.num_slots * self.slot_feature_dim
        if input_dim < self.slot_flat_dim:
            raise ValueError(
                f"input_dim {input_dim} is smaller than slot flat dim {self.slot_flat_dim}"
            )
        self.extra_dim = int(input_dim - self.slot_flat_dim)
        self.gate_l1_weight = float(gate_l1_weight)

        role_input_dim = self.slot_feature_dim + self.extra_dim
        self.nuisance_gate_logits = nn.Parameter(torch.zeros(self.num_slots))
        self.effect_gate_logits = nn.Parameter(torch.zeros(self.num_slots))

        self.nuisance_rep = nn.Sequential(
            nn.Linear(role_input_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.effect_rep = nn.Sequential(
            nn.Linear(role_input_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.nuisance_fc = nn.Sequential(
            nn.Linear(representation_dim, hidden_outcome_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.propensity_fc = nn.Linear(hidden_outcome_dim, 1)
        self.outcome_fc = nn.Linear(hidden_outcome_dim, 1)

        self.effect_fc = nn.Sequential(
            nn.Linear(representation_dim, hidden_outcome_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_outcome_dim, hidden_outcome_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.effect_out = nn.Linear(hidden_outcome_dim, 1)

    def _split_features(self, features: torch.Tensor):
        slot_features = features[:, : self.slot_flat_dim].reshape(
            features.shape[0],
            self.num_slots,
            self.slot_feature_dim,
        )
        extra = features[:, self.slot_flat_dim:] if self.extra_dim > 0 else None
        return slot_features, extra

    def _weighted_slots(self, slot_features: torch.Tensor, gate_logits: torch.Tensor):
        weights = torch.sigmoid(gate_logits)
        pooled = torch.einsum("bsf,s->bf", slot_features, weights)
        return pooled / weights.sum().clamp_min(1e-6)

    def _role_inputs(self, features: torch.Tensor):
        slot_features, extra = self._split_features(features)
        nuisance_input = self._weighted_slots(slot_features, self.nuisance_gate_logits)
        effect_input = self._weighted_slots(slot_features, self.effect_gate_logits)
        if extra is not None:
            nuisance_input = torch.cat([nuisance_input, extra], dim=-1)
            effect_input = torch.cat([effect_input, extra], dim=-1)
        return nuisance_input, effect_input

    def get_representation(self, features: torch.Tensor):
        nuisance_input, _ = self._role_inputs(features)
        return self.nuisance_rep(nuisance_input)

    def propensity_from_representation(self, phi: torch.Tensor):
        hidden = self.nuisance_fc(phi)
        return self.propensity_fc(hidden)

    def forward(self, features: torch.Tensor):
        nuisance_input, effect_input = self._role_inputs(features)
        nuisance_phi = self.nuisance_rep(nuisance_input)
        effect_phi = self.effect_rep(effect_input)

        nuisance_hidden = self.nuisance_fc(nuisance_phi)
        t_logit = self.propensity_fc(nuisance_hidden)
        m_logit = self.outcome_fc(nuisance_hidden)

        effect_hidden = self.effect_fc(effect_phi)
        tau = self.effect_out(effect_hidden)
        phi = torch.cat([nuisance_phi, effect_phi], dim=-1)
        return m_logit, tau, t_logit, phi

    def forward_with_activations(self, features: torch.Tensor):
        nuisance_input, effect_input = self._role_inputs(features)
        nuisance_phi = self.nuisance_rep(nuisance_input)
        effect_phi = self.effect_rep(effect_input)

        nuisance_hidden = self.nuisance_fc(nuisance_phi)
        t_logit = self.propensity_fc(nuisance_hidden)
        m_logit = self.outcome_fc(nuisance_hidden)

        effect_hidden = self.effect_fc(effect_phi)
        tau = self.effect_out(effect_hidden)
        phi = torch.cat([nuisance_phi, effect_phi], dim=-1)
        return m_logit, tau, t_logit, phi, nuisance_hidden, effect_hidden

    def compute_regularization_losses(self):
        if self.gate_l1_weight <= 0:
            device = self.nuisance_gate_logits.device
            return {"slot_gate_l1_loss": torch.tensor(0.0, device=device)}
        nuisance = torch.sigmoid(self.nuisance_gate_logits)
        effect = torch.sigmoid(self.effect_gate_logits)
        return {
            "slot_gate_l1_loss": self.gate_l1_weight * (nuisance.mean() + effect.mean())
        }

    def get_gate_values(self):
        return {
            "nuisance": torch.sigmoid(self.nuisance_gate_logits).detach(),
            "effect": torch.sigmoid(self.effect_gate_logits).detach(),
        }
