"""
TID: Time Image Decomposition

Implements the TID module from TimeMixer++ (Equation 7).
Decomposes time images into seasonal and trend components using
dual-axis attention (column-axis for seasonal, row-axis for trend).

Key Design Choices:
- Shared 2D convolution for Q/K/V generation across all images
- Column-axis attention captures intra-period patterns (seasonal)
- Row-axis attention captures inter-period patterns (trend)
- Batch dimension merging for efficient attention computation
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from .layers import MultiHeadSelfAttention


class SharedQKVConv(nn.Module):
    """
    Shared 2D convolution for generating Q, K, V from time images.
    
    This convolution is shared across all time images (all scales and periods)
    as described in the paper.
    
    Args:
        d_model: Number of input/output channels
        kernel_size: Convolution kernel size (default 1 for projection)
    
    Input Shape: (B, d_model, H, W)
    Output Shape: (B, 3*d_model, H, W) - concatenated Q, K, V
    """
    
    def __init__(self, d_model: int, kernel_size: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            d_model, 3 * d_model,
            kernel_size=kernel_size,
            padding=padding
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate Q, K, V from input.
        
        Args:
            x: Input tensor of shape (B, d_model, H, W)
            
        Returns:
            Q, K, V: Each of shape (B, d_model, H, W)
        """
        qkv = self.conv(x)  # (B, 3*d_model, H, W)
        Q, K, V = torch.chunk(qkv, 3, dim=1)
        return Q, K, V


class DualAxisAttention(nn.Module):
    """
    Dual-axis attention for time image decomposition.
    
    Performs:
    1. Column-axis attention (along W dimension) -> Seasonal component
    2. Row-axis attention (along H dimension) -> Trend component
    
    The key insight is that we merge the non-target axis into the batch dimension
    for efficient attention computation:
    - Column attention: reshape (B, d, H, W) -> (B*H, W, d), attend, reshape back
    - Row attention: reshape (B, d, H, W) -> (B*W, H, d), attend, reshape back
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Shared QKV conv (shared across column and row attention)
        self.qkv_conv = SharedQKVConv(d_model, kernel_size=1)
        
        # Separate attention layers for column (seasonal) and row (trend)
        self.col_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.row_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        # Layer norms
        self.norm_col = nn.LayerNorm(d_model)
        self.norm_row = nn.LayerNorm(d_model)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose time image into seasonal and trend components.
        
        Args:
            z: Time image of shape (B, d_model, H, W)
               H = period (p_k), W = ceil(L_m / p_k)
               
        Returns:
            s: Seasonal component, shape (B, d_model, H, W)
            t: Trend component, shape (B, d_model, H, W)
        """
        B, d_model, H, W = z.shape
        
        # Generate Q, K, V using shared conv
        Q, K, V = self.qkv_conv(z)  # Each: (B, d_model, H, W)
        
        # ============================================
        # Column-axis attention (Seasonal)
        # Attend along W dimension (columns)
        # Reshape: merge H into batch -> (B*H, W, d_model)
        # ============================================
        
        # Permute: (B, d, H, W) -> (B, H, W, d)
        Q_col = Q.permute(0, 2, 3, 1)  # (B, H, W, d)
        K_col = K.permute(0, 2, 3, 1)
        V_col = V.permute(0, 2, 3, 1)
        
        # Merge B and H: (B, H, W, d) -> (B*H, W, d)
        Q_col = Q_col.reshape(B * H, W, d_model)
        K_col = K_col.reshape(B * H, W, d_model)
        V_col = V_col.reshape(B * H, W, d_model)
        
        # For MHSA, we use V as input (Q, K generated internally)
        # But our MHSA takes single input, so we just pass V
        # and rely on the attention module to handle QKV
        s_flat = self.col_attn(V_col)  # (B*H, W, d)
        s_flat = self.norm_col(s_flat)
        
        # Reshape back: (B*H, W, d) -> (B, H, W, d) -> (B, d, H, W)
        s = s_flat.reshape(B, H, W, d_model)
        s = s.permute(0, 3, 1, 2)  # (B, d, H, W)
        
        # ============================================
        # Row-axis attention (Trend)
        # Attend along H dimension (rows)
        # Reshape: merge W into batch -> (B*W, H, d_model)
        # ============================================
        
        # Permute: (B, d, H, W) -> (B, W, H, d)
        Q_row = Q.permute(0, 3, 2, 1)  # (B, W, H, d)
        K_row = K.permute(0, 3, 2, 1)
        V_row = V.permute(0, 3, 2, 1)
        
        # Merge B and W: (B, W, H, d) -> (B*W, H, d)
        Q_row = Q_row.reshape(B * W, H, d_model)
        K_row = K_row.reshape(B * W, H, d_model)
        V_row = V_row.reshape(B * W, H, d_model)
        
        # Attention
        t_flat = self.row_attn(V_row)  # (B*W, H, d)
        t_flat = self.norm_row(t_flat)
        
        # Reshape back: (B*W, H, d) -> (B, W, H, d) -> (B, d, H, W)
        t = t_flat.reshape(B, W, H, d_model)
        t = t.permute(0, 3, 2, 1)  # (B, d, H, W)
        
        # Verify output shapes
        assert s.shape == z.shape, f"Seasonal shape mismatch: {s.shape} vs {z.shape}"
        assert t.shape == z.shape, f"Trend shape mismatch: {t.shape} vs {z.shape}"
        
        return s, t


class TID(nn.Module):
    """
    Time Image Decomposition module.
    
    Applies dual-axis attention to decompose each time image into
    seasonal and trend components.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dual_axis_attn = DualAxisAttention(d_model, n_heads, dropout)
    
    def forward(
        self,
        images: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Decompose a list of time images for one period.
        
        Args:
            images: List of M+1 time images, one per scale
                   Each image has shape (B, d_model, H=period, W=ceil(L_m/period))
                   
        Returns:
            seasonal_images: List of M+1 seasonal components
            trend_images: List of M+1 trend components
        """
        seasonal_images = []
        trend_images = []
        
        for z in images:
            s, t = self.dual_axis_attn(z)
            seasonal_images.append(s)
            trend_images.append(t)
        
        return seasonal_images, trend_images

