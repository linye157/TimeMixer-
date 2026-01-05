"""
MCM: Multi-Scale Mixing

Implements the MCM module from TimeMixer++ (Equations 8-9-10).
Performs scale-wise mixing of seasonal (bottom-up) and trend (top-down) components.

Key Design Choices:
- For a fixed period, seasonal images are mixed bottom-up (fine to coarse)
- Trend images are mixed top-down (coarse to fine)
- 2D convolutions use stride=(1,2) on the W dimension since temporal length
  varies across scales through W (columns), not H (rows = period)
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .layers import ConvDown2d, TransConvUp2d, match_shape


class MCM(nn.Module):
    """
    Multi-Scale Mixing module.
    
    For a single period k, mixes:
    - Seasonal images bottom-up: s_m += ConvDown(s_{m-1})
    - Trend images top-down: t_m += TransConvUp(t_{m+1})
    
    Then combines: z_m = s_m + t_m
    
    Args:
        d_model: Model dimension
        kernel_size: Kernel size for 2D convolutions
        use_two_stride_layers: Whether to use stride in both conv layers
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        use_two_stride_layers: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        
        # We'll create conv layers dynamically based on number of scales
        # For now, we use a ModuleList and create layers lazily
        self.kernel_size = kernel_size
        self.use_two_stride_layers = use_two_stride_layers
        
        # Will be initialized on first forward
        self.down_convs = None
        self.up_convs = None
        self._num_scales = None
    
    def _init_convs(self, num_scales: int, device: torch.device):
        """
        Initialize convolution layers for the given number of scales.
        
        Args:
            num_scales: Number of scales (M + 1)
            device: Device to create layers on
        """
        if self._num_scales == num_scales:
            return
        
        self._num_scales = num_scales
        M = num_scales - 1
        
        # Bottom-up: need M ConvDown layers (m=1..M)
        self.down_convs = nn.ModuleList([
            ConvDown2d(self.d_model, self.kernel_size, self.use_two_stride_layers)
            for _ in range(M)
        ]).to(device)
        
        # Top-down: need M TransConvUp layers (m=M-1..0)
        self.up_convs = nn.ModuleList([
            TransConvUp2d(self.d_model, self.kernel_size, self.use_two_stride_layers)
            for _ in range(M)
        ]).to(device)
    
    def forward(
        self,
        seasonal_images: List[torch.Tensor],
        trend_images: List[torch.Tensor],
        original_lengths: List[int],
        period: int
    ) -> List[torch.Tensor]:
        """
        Perform multi-scale mixing for a single period.
        
        Args:
            seasonal_images: List of M+1 seasonal images [s_0, s_1, ..., s_M]
                            Each s_m has shape (B, d_model, H=period, W_m)
            trend_images: List of M+1 trend images [t_0, t_1, ..., t_M]
                         Each t_m has shape (B, d_model, H=period, W_m)
            original_lengths: List of original sequence lengths [L_0, L_1, ..., L_M]
            period: The period length (p_k)
            
        Returns:
            z_list: List of M+1 mixed representations in 1D
                   Each z_m has shape (B, L_m, d_model)
        """
        num_scales = len(seasonal_images)
        M = num_scales - 1
        
        device = seasonal_images[0].device
        self._init_convs(num_scales, device)
        
        # Make copies to avoid modifying inputs
        s_list = [s.clone() for s in seasonal_images]
        t_list = [t.clone() for t in trend_images]
        
        # ============================================
        # Bottom-up mixing for seasonal (m = 1 to M)
        # s_m = s_m + ConvDown(s_{m-1})
        # ============================================
        for m in range(1, num_scales):
            s_prev = s_list[m - 1]  # (B, d, H, W_{m-1})
            s_curr = s_list[m]      # (B, d, H, W_m)
            
            # Downsample s_prev
            s_down = self.down_convs[m - 1](s_prev)  # (B, d, H, W')
            
            # Match shape to s_curr
            target_H = s_curr.shape[2]
            target_W = s_curr.shape[3]
            s_down = match_shape(s_down, target_H, target_W)
            
            # Add
            s_list[m] = s_curr + s_down
        
        # ============================================
        # Top-down mixing for trend (m = M-1 to 0)
        # t_m = t_m + TransConvUp(t_{m+1})
        # ============================================
        for m in range(M - 1, -1, -1):
            t_next = t_list[m + 1]  # (B, d, H, W_{m+1})
            t_curr = t_list[m]      # (B, d, H, W_m)
            
            # Upsample t_next
            t_up = self.up_convs[M - 1 - m](t_next)  # (B, d, H, W')
            
            # Match shape to t_curr
            target_H = t_curr.shape[2]
            target_W = t_curr.shape[3]
            t_up = match_shape(t_up, target_H, target_W)
            
            # Add
            t_list[m] = t_curr + t_up
        
        # ============================================
        # Combine: z_m = s_m + t_m
        # Then reshape 2D -> 1D and truncate to original length
        # ============================================
        z_list = []
        
        for m in range(num_scales):
            z_2d = s_list[m] + t_list[m]  # (B, d_model, H, W)
            
            # Reshape to 1D: (B, d, H, W) -> (B, H*W, d)
            B, d_model, H, W = z_2d.shape
            z_1d = z_2d.permute(0, 2, 3, 1)  # (B, H, W, d)
            z_1d = z_1d.reshape(B, H * W, d_model)  # (B, H*W, d)
            
            # Truncate to original length
            L_m = original_lengths[m]
            z_1d = z_1d[:, :L_m, :]  # (B, L_m, d)
            
            z_list.append(z_1d)
        
        return z_list

