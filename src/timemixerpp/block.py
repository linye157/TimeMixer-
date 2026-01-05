"""
MixerBlock: Core building block of TimeMixer++

Combines MRTI, TID, MCM, and MRM into a single block with residual connections.
Each block takes multi-scale representations and outputs transformed multi-scale representations.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .mrti import MRTI, TimeImageInfo
from .tid import TID
from .mcm import MCM
from .mrm import MRM
from .config import TimeMixerPPConfig


class MixerBlock(nn.Module):
    """
    MixerBlock: X^{l+1} = LayerNorm(X^l + MixerBlock_core(X^l))
    
    The core block performs:
    1. MRTI: Convert multi-scale 1D to multi-resolution 2D images
    2. TID: Decompose each image into seasonal and trend
    3. MCM: Mix seasonal (bottom-up) and trend (top-down) across scales
    4. MRM: Aggregate across periods using amplitude weights
    
    Layer normalization is applied per-scale for stable training.
    This is a practical engineering choice that works well for the
    "across scales" normalization described in the paper.
    
    Args:
        config: TimeMixerPPConfig with all hyperparameters
    """
    
    def __init__(self, config: TimeMixerPPConfig):
        super().__init__()
        self.config = config
        
        # MRTI: Multi-Resolution Time Imaging
        self.mrti = MRTI(
            top_k=config.top_k,
            base_len_for_period=config.base_len_for_period,
            min_period=config.min_period,
            max_period_factor=1.0
        )
        
        # TID: Time Image Decomposition
        self.tid = TID(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        
        # MCM: Multi-Scale Mixing
        self.mcm = MCM(
            d_model=config.d_model,
            kernel_size=config.mcm_kernel_size,
            use_two_stride_layers=config.mcm_use_two_stride_layers
        )
        
        # MRM: Multi-Resolution Mixing
        self.mrm = MRM(weight_mode=config.weight_mode)
        
        # Layer norms for each scale (will be initialized dynamically)
        self._layer_norms = None
        self._num_scales = None
    
    def _init_layer_norms(self, num_scales: int, device: torch.device):
        """Initialize layer norms for each scale."""
        if self._num_scales == num_scales:
            return
        
        self._num_scales = num_scales
        self._layer_norms = nn.ModuleList([
            nn.LayerNorm(self.config.d_model)
            for _ in range(num_scales)
        ]).to(device)
    
    def forward(
        self,
        multi_scale_x: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Forward pass through the MixerBlock.
        
        Args:
            multi_scale_x: List of M+1 tensors [x_0, x_1, ..., x_M]
                          Each x_m has shape (B, L_m, d_model)
                          
        Returns:
            output: List of M+1 tensors with same shapes as input
        """
        num_scales = len(multi_scale_x)
        device = multi_scale_x[0].device
        
        # Initialize layer norms if needed
        self._init_layer_norms(num_scales, device)
        
        # Store residuals
        residuals = multi_scale_x
        
        # ============================================
        # Step 1: MRTI - Convert to time images
        # ============================================
        time_images, periods, amplitudes = self.mrti(multi_scale_x)
        # time_images: List[TimeImageInfo], one per period
        # Each TimeImageInfo contains images for all scales
        
        # ============================================
        # Step 2 & 3: TID + MCM for each period
        # ============================================
        z_per_period = []  # Will hold K lists, each with M+1 tensors
        
        for time_image_info in time_images:
            # time_image_info.images: List of M+1 images
            # Each image: (B, d_model, H=period, W=f_m_k)
            
            # TID: Decompose into seasonal and trend
            seasonal_images, trend_images = self.tid(time_image_info.images)
            
            # MCM: Multi-scale mixing
            z_list = self.mcm(
                seasonal_images,
                trend_images,
                time_image_info.original_lengths,
                time_image_info.period
            )
            # z_list: List of M+1 tensors, each (B, L_m, d_model)
            
            z_per_period.append(z_list)
        
        # ============================================
        # Step 4: MRM - Aggregate across periods
        # ============================================
        x_out = self.mrm(z_per_period, amplitudes)
        # x_out: List of M+1 tensors, each (B, L_m, d_model)
        
        # ============================================
        # Residual connection + LayerNorm
        # ============================================
        output = []
        for m in range(num_scales):
            # Residual
            out_m = residuals[m] + x_out[m]
            # LayerNorm per scale
            out_m = self._layer_norms[m](out_m)
            output.append(out_m)
        
        return output

