# cdt/models/outcome_heads.py
"""Lightweight outcome heads for oracle mode - takes phi directly, no representation layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutcomeHeadsOnly(nn.Module):
    """
    Outcome heads that take phi representation directly.

    Use this for oracle mode where phi is already extracted from the generator.
    This matches the old_cdt architecture where outcome heads were simple:
    - phi -> hidden -> hidden -> output (2 hidden layers, not 5)

    Architecture mirrors old_cdt/model.py DragonNetInternal outcome heads.
    """

    def __init__(self, phi_dim, hidden_outcome_dim=100):
        super().__init__()

        # Propensity head (single layer like old version)
        self.propensity_fc1 = nn.Linear(phi_dim, 1)

        # Y0 outcome head (2 hidden layers like old version)
        self.outcome0_fc1 = nn.Linear(phi_dim, hidden_outcome_dim)
        self.outcome0_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome0_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Y1 outcome head (2 hidden layers like old version)
        self.outcome1_fc1 = nn.Linear(phi_dim, hidden_outcome_dim)
        self.outcome1_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome1_fc3 = nn.Linear(hidden_outcome_dim, 1)

    def forward(self, phi):
        """
        Args:
            phi: Pre-extracted representation from generator (batch, phi_dim)

        Returns:
            y0_logit, y1_logit, t_logit, phi (pass-through for compatibility)
        """
        # Propensity (single layer)
        t_logit = self.propensity_fc1(phi)

        # Y0 outcome (2 hidden layers: ReLU -> ELU -> Linear)
        y0 = F.relu(self.outcome0_fc1(phi))
        y0 = F.elu(self.outcome0_fc2(y0))
        y0_logit = self.outcome0_fc3(y0)

        # Y1 outcome (2 hidden layers: ReLU -> ELU -> Linear)
        y1 = F.relu(self.outcome1_fc1(phi))
        y1 = F.elu(self.outcome1_fc2(y1))
        y1_logit = self.outcome1_fc3(y1)

        return y0_logit, y1_logit, t_logit, phi


class UpliftHeadsOnly(nn.Module):
    """
    Uplift parametrization (y0, tau) that takes phi representation directly.

    Use this for oracle mode with uplift modeling.
    """

    def __init__(self, phi_dim, hidden_outcome_dim=100):
        super().__init__()

        # Propensity head (single layer)
        self.propensity_fc1 = nn.Linear(phi_dim, 1)

        # Baseline Y0 head (2 hidden layers)
        self.baseline_fc1 = nn.Linear(phi_dim, hidden_outcome_dim)
        self.baseline_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.baseline_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Treatment effect Tau head (2 hidden layers)
        self.effect_fc1 = nn.Linear(phi_dim, hidden_outcome_dim)
        self.effect_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.effect_fc3 = nn.Linear(hidden_outcome_dim, 1)

    def forward(self, phi):
        """
        Args:
            phi: Pre-extracted representation from generator (batch, phi_dim)

        Returns:
            y0_logit, tau_logit, t_logit, phi (pass-through for compatibility)
        """
        # Propensity
        t_logit = self.propensity_fc1(phi)

        # Baseline Y0
        y0 = F.relu(self.baseline_fc1(phi))
        y0 = F.elu(self.baseline_fc2(y0))
        y0_logit = self.baseline_fc3(y0)

        # Treatment effect Tau
        tau = F.relu(self.effect_fc1(phi))
        tau = F.elu(self.effect_fc2(tau))
        tau_logit = self.effect_fc3(tau)

        return y0_logit, tau_logit, t_logit, phi
