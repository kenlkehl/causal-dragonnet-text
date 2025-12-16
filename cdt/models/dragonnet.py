# cdt/models/dragonnet.py
"""DragonNet architecture for causal inference from confounders."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class DragonNet(nn.Module):
    """Binary treatment DragonNet. Can be initialized from pretrained multi-treatment model."""
    
    def __init__(self, input_dim, representation_dim=200, hidden_outcome_dim=100):
        super().__init__()
        # Shared representation layers (can be loaded from pretrained)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc3 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc4 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc5 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc6 = nn.Linear(representation_dim, representation_dim)

        # Binary treatment propensity head (always randomly initialized)
        self.propensity_fc1 = nn.Linear(representation_dim, representation_dim)
        self.propensity_fc2 = nn.Linear(representation_dim, representation_dim)
        self.propensity_fc3 = nn.Linear(representation_dim, representation_dim)
        self.propensity_fc4 = nn.Linear(representation_dim, 1)
        

        # Binary outcome heads (always randomly initialized)
        self.outcome0_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.outcome0_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome0_fc2a = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome0_fc2b = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome0_fc2c = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome0_fc3 = nn.Linear(hidden_outcome_dim, 1)

        self.outcome1_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.outcome1_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome1_fc2a = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome1_fc2b = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome1_fc2c = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome1_fc3 = nn.Linear(hidden_outcome_dim, 1)

        
    def forward(self, x_representation):
        phi = F.relu(self.representation_fc1(x_representation))
        phi = F.relu(self.representation_fc2(phi))
        phi = F.relu(self.representation_fc3(phi))
        phi = F.relu(self.representation_fc4(phi))
        phi = F.relu(self.representation_fc5(phi))
        phi = F.elu(self.representation_fc6(phi))

        t = F.relu(self.propensity_fc1(phi))
        t = F.relu(self.propensity_fc2(t))
        t = F.relu(self.propensity_fc3(t))
        t_logit = self.propensity_fc4(t)

        y0 = F.relu(self.outcome0_fc1(phi))
        y0 = F.relu(self.outcome0_fc2(y0))
        y0 = F.relu(self.outcome0_fc2a(y0))
        y0 = F.relu(self.outcome0_fc2b(y0))
        y0 = F.elu(self.outcome0_fc2c(y0))
        y0_logit = self.outcome0_fc3(y0)

        y1 = F.relu(self.outcome1_fc1(phi))
        y1 = F.relu(self.outcome1_fc2(y1))
        y1 = F.relu(self.outcome1_fc2a(y1))
        y1 = F.relu(self.outcome1_fc2b(y1))
        y1 = F.elu(self.outcome1_fc2c(y1))
        y1_logit = self.outcome1_fc3(y1)

        return y0_logit, y1_logit, t_logit, phi

    def load_pretrained_representation(self, pretrained_state_dict):
        """
        Load pretrained representation layers (fc1-fc6) if dimensions match.
        
        Args:
            pretrained_state_dict: State dict from pretrained model (can be full checkpoint or just representation layers)
        
        Returns:
            bool: True if loaded successfully, False if dimension mismatch
        """
        # Handle both full checkpoint and direct state dict

        # Handle full checkpoint format
        if 'dragonnet' in pretrained_state_dict:
            state_dict = pretrained_state_dict['dragonnet']
        elif 'dragonnet_representation' in pretrained_state_dict:
            state_dict = pretrained_state_dict['dragonnet_representation']
        elif 'representation_fc1.weight' in pretrained_state_dict:
            state_dict = pretrained_state_dict
        elif 'representation_fc1' in pretrained_state_dict and isinstance(pretrained_state_dict['representation_fc1'], dict):
            # Nested dict format (legacy)
            state_dict = {}
            for key in ['representation_fc1', 'representation_fc2', 'representation_fc3',
                       'representation_fc4', 'representation_fc5', 'representation_fc6']:
                if key in pretrained_state_dict:
                    for param_name, param_value in pretrained_state_dict[key].items():
                        state_dict[f'{key}.{param_name}'] = param_value
        else:
            logger.warning("Cannot parse pretrained state dict format")
            return False
        
        # Check if input dimensions match
        pretrained_fc1_weight_shape = state_dict['representation_fc1.weight'].shape
        current_fc1_weight_shape = self.representation_fc1.weight.shape
        
        if pretrained_fc1_weight_shape != current_fc1_weight_shape:
            logger.warning("Cannot load pretrained representation - dimension mismatch!")
            logger.warning(f"  Pretrained input dim: {pretrained_fc1_weight_shape[1]}")
            logger.warning(f"  Current input dim: {current_fc1_weight_shape[1]}")
            logger.warning(f"  This usually means different numbers of confounders between pretrain and current model.")
            logger.warning(f"  Skipping pretrained representation weights. Model will use random initialization.")
            return False
        
        # Dimensions match - load all layers
        try:
            # Create state dict for just the representation layers
            rep_state_dict = {}
            for key in ['representation_fc1', 'representation_fc2', 'representation_fc3',
                       'representation_fc4', 'representation_fc5', 'representation_fc6']:
                for param_name in ['weight', 'bias']:
                    full_key = f'{key}.{param_name}'
                    if full_key in state_dict:
                        rep_state_dict[full_key] = state_dict[full_key]
            
            # Load with strict=False to allow missing keys (outcome/propensity heads)
            self.load_state_dict(rep_state_dict, strict=False)
            logger.info("Successfully loaded pretrained representation layers (fc1-fc6)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            return False