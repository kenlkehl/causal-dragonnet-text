# cdt/models/components.py
"""Aggregator modules for pooling chunk embeddings into confounder representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfounderAggregator(nn.Module):
    """
    Pool feature maps (B, C_total, L) -> (B, C_total * features_per_agg_output)
    
    Modes: 'max', 'lsep', 'gem', 'topk', 'attn', 'stats', 'noisyor'
    """
    
    def __init__(
        self,
        mode: str = 'attn',
        temperature: float = 0.5,
        topk: int = 3,
        per_confounder_params: bool = True
    ):
        """
        Initialize aggregator.
        
        Args:
            mode: Aggregation mode
            temperature: Temperature parameter for soft pooling
            topk: Number of top chunks for topk mode
            per_confounder_params: Use separate parameters per confounder
        """
        super().__init__()
        
        valid_modes = {'max', 'lsep', 'gem', 'topk', 'attn', 'stats', 'noisyor'}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
        
        self.mode = mode
        self.topk = topk
        self.per_confounder_params = per_confounder_params
        self._initialized = False
        
        if not per_confounder_params:
            self.tau = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.tau = None
    
    def _lazy_init(self, num_confounders: int, device: torch.device):
        """Initialize parameters based on input dimensions."""
        if self.mode == 'gem':
            self.raw_p = nn.Parameter(torch.zeros(1, device=device))
            self.features_per_conf = 1
        elif self.mode == 'stats':
            self.features_per_conf = 2
        else:
            self.features_per_conf = 1
        
        if self.per_confounder_params:
            if self.mode in {'lsep', 'noisyor', 'attn'}:  # Add modes that need it
                self.log_tau = nn.Parameter(torch.zeros(num_confounders, 1, device=device))
            else:
                self.log_tau = None  # Explicitly set to None for other modes
       
        self._initialized = True
    
    def forward(
        self,
        feature_maps: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool feature maps.
        
        Args:
            feature_maps: (batch, confounders, sequence_length)
            mask: (batch, 1, sequence_length) boolean mask
        
        Returns:
            Pooled features: (batch, confounders * features_per_conf)
        """
        batch_size, num_confounders, seq_len = feature_maps.shape
        device = feature_maps.device
        
        if not self._initialized:
            self._lazy_init(num_confounders, device)
        
        fm = feature_maps.clone()
        fill_value = float('-inf') if self.mode in {'max', 'lsep', 'topk', 'attn'} else 0.0
        fm.masked_fill_(mask, fill_value)
        
        if self.mode == 'max':
            pooled = torch.amax(fm, dim=2)
        
        elif self.mode == 'lsep':
            tau = self._get_tau(num_confounders, device)
            z = (fm / tau).logsumexp(dim=2)
            pooled = tau.squeeze(-1) * z
        
        elif self.mode == 'gem':
            p = 1.0 + F.softplus(self.raw_p.squeeze())
            x = F.relu(fm) + 1e-6
            pooled = torch.pow(torch.mean(torch.pow(x, p), dim=2), 1.0 / p)
        
        elif self.mode == 'topk':
            k = min(self.topk, seq_len)
            vals, _ = torch.topk(fm, k=k, dim=2)
            pooled = torch.mean(vals, dim=2)
        
        elif self.mode == 'attn':
            tau = self._get_tau(num_confounders, device)
            scores = fm / tau
            scores.masked_fill_(mask, float('-inf'))
            weights = torch.softmax(scores, dim=2)
            pooled = torch.sum(
                weights * feature_maps.clamp_min(-50).clamp_max(50),
                dim=2
            )
        
        elif self.mode == 'stats':
            valid = (~mask).float()
            denom = valid.sum(dim=2).clamp_min(1.0)
            mean = (fm * valid).sum(dim=2) / denom
            
            x = feature_maps.clone()
            x.masked_fill_(mask, 0.0)
            mean_for_var = (x * valid).sum(dim=2) / denom
            var = ((x - mean_for_var.unsqueeze(-1))**2 * valid).sum(dim=2) / denom
            std = torch.sqrt(var + 1e-6)
            pooled = torch.cat([mean, std], dim=1)
        
        elif self.mode == 'noisyor':
            tau = self._get_tau(num_confounders, device)
            probs = torch.sigmoid(fm / tau).clamp(1e-6, 1 - 1e-6)
            probs.masked_fill_(mask, 0.0)
            log1m = torch.log1p(-probs)
            log_prod = torch.sum(log1m, dim=2)
            any_prob = 1.0 - torch.exp(log_prod)
            pooled = torch.log(any_prob) - torch.log1p(-any_prob)
        
        return pooled
    
    def _get_tau(self, num_confounders: int, device: torch.device) -> torch.Tensor:
        """Get temperature parameter."""
        if self.per_confounder_params:
            if self.log_tau is not None:
                return torch.exp(self.log_tau).view(1, num_confounders, 1)
            else:
                return torch.tensor(0.5, device=device).view(1, 1, 1)
