"""
MRM: Multi-Resolution Mixing

Implements the MRM module from TimeMixer++ (Equation 11).
Aggregates multi-resolution representations using amplitude-weighted sum.

Key Design Choices:
- Amplitudes from FFT on coarsest scale weight each period's contribution
- Supports both global (batch-shared) and per-sample amplitude weighting
- Softmax normalization ensures proper weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal


class MRM(nn.Module):
    """
    Multi-Resolution Mixing module.
    
    For each scale m, aggregates representations from K periods using
    amplitude-weighted sum:
        x_m = Σ_k softmax(A)[k] * z_m^(k)
    
    Args:
        weight_mode: 'global' for batch-shared weights, 'per_sample' for per-batch weights
    """
    
    def __init__(self, weight_mode: Literal['global', 'per_sample'] = 'per_sample'):
        super().__init__()
        self.weight_mode = weight_mode
    
    def forward(
        self,
        z_per_period: List[List[torch.Tensor]],
        amplitudes: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Aggregate multi-resolution representations.
        
        Args:
            z_per_period: List of K period-wise results
                         Each element is List of M+1 tensors [z_0^(k), ..., z_M^(k)]
                         Each z_m^(k) has shape (B, L_m, d_model)
            amplitudes: Amplitude weights, shape (B, K) for per_sample or (K,) for global
            
        Returns:
            x_list: List of M+1 aggregated representations
                   Each x_m has shape (B, L_m, d_model)
        """
        K = len(z_per_period)
        
        if K == 0:
            raise ValueError("z_per_period cannot be empty")
        
        num_scales = len(z_per_period[0])
        
        # Compute normalized weights
        if self.weight_mode == 'global':
            # Average amplitudes across batch, then softmax
            if amplitudes.dim() > 1:
                amp_mean = amplitudes.mean(dim=0)  # (K,)
            else:
                amp_mean = amplitudes
            weights = F.softmax(amp_mean, dim=0)  # (K,)
        else:
            # Per-sample softmax
            # amplitudes: (B, K)
            weights = F.softmax(amplitudes, dim=1)  # (B, K)
        
        # Aggregate for each scale
        x_list = []
        
        for m in range(num_scales):
            # Collect z_m^(k) for all k
            z_m_list = [z_per_period[k][m] for k in range(K)]  # K tensors, each (B, L_m, d)
            
            # Stack: (K, B, L_m, d)
            z_m_stack = torch.stack(z_m_list, dim=0)
            
            if self.weight_mode == 'global':
                # weights: (K,) -> broadcast to (K, 1, 1, 1)
                w = weights.view(K, 1, 1, 1)
                x_m = (w * z_m_stack).sum(dim=0)  # (B, L_m, d)
            else:
                # weights: (B, K) -> (K, B, 1, 1) after transpose
                w = weights.t().unsqueeze(-1).unsqueeze(-1)  # (K, B, 1, 1)
                x_m = (w * z_m_stack).sum(dim=0)  # (B, L_m, d)
            
            x_list.append(x_m)
        
        return x_list

