"""
Common layers and utility functions for TimeMixer++.

Includes:
- Multi-Head Self-Attention (MHSA)
- Downsampling and Upsampling Conv blocks
- Shape matching utilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with batch_first support.
    
    Implements scaled dot-product attention with multiple heads.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    
    Input Shape: (B, L, d_model)
    Output Shape: (B, L, d_model)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (B, L, d_model)
        """
        B, L, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.W_qkv(x)  # (B, L, 3*d_model)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, L, d_head)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: (B, n_heads, L, d_head)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, n_heads, L, L)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, V)  # (B, n_heads, L, d_head)
        out = out.permute(0, 2, 1, 3).reshape(B, L, self.d_model)  # (B, L, d_model)
        out = self.W_out(out)
        
        return out


class DownsampleConv1d(nn.Module):
    """
    1D Downsampling convolution for multi-scale time series generation.
    
    Reduces sequence length by factor of 2 using strided convolution.
    
    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        kernel_size: Convolution kernel size
        padding: Convolution padding
        groups: Convolution groups
        use_norm: Whether to apply layer normalization
    
    Input Shape: (B, L, C)
    Output Shape: (B, L//2, C)
    """
    
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
        use_norm: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            c_in, c_out,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            groups=groups
        )
        self.norm = nn.LayerNorm(c_out) if use_norm else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, C)
            
        Returns:
            Downsampled tensor of shape (B, L//2, C)
        """
        # Conv1d expects (B, C, L)
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.conv(x)  # (B, C, L//2)
        x = x.transpose(1, 2)  # (B, L//2, C)
        x = self.norm(x)
        return x


class ConvDown2d(nn.Module):
    """
    2D Downsampling convolution block for MCM (Multi-Scale Mixing).
    
    Uses stride=(1, 2) to downsample along the W dimension (columns) while
    keeping H (rows) unchanged. This is because in time images, the temporal
    length varies with scale through the W dimension.
    
    Args:
        d_model: Number of channels
        kernel_size: Convolution kernel size
        use_two_stride_layers: If True, both layers use stride=(1,2); otherwise only first
    
    Input Shape: (B, d_model, H, W)
    Output Shape: (B, d_model, H, W//2) approximately
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        use_two_stride_layers: bool = False
    ):
        super().__init__()
        padding = kernel_size // 2
        
        # First layer with stride
        self.conv1 = nn.Conv2d(
            d_model, d_model,
            kernel_size=kernel_size,
            stride=(1, 2),
            padding=padding
        )
        
        # Second layer: stride depends on config
        stride2 = (1, 2) if use_two_stride_layers else (1, 1)
        self.conv2 = nn.Conv2d(
            d_model, d_model,
            kernel_size=kernel_size,
            stride=stride2,
            padding=padding
        )
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, d_model, H, W)
            
        Returns:
            Downsampled tensor of shape (B, d_model, H, W')
        """
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x


class TransConvUp2d(nn.Module):
    """
    2D Transposed convolution block for MCM (Multi-Scale Mixing).
    
    Uses stride=(1, 2) to upsample along the W dimension while keeping H unchanged.
    
    Args:
        d_model: Number of channels
        kernel_size: Convolution kernel size
        use_two_stride_layers: If True, both layers use stride=(1,2); otherwise only first
    
    Input Shape: (B, d_model, H, W)
    Output Shape: (B, d_model, H, W*2) approximately
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        use_two_stride_layers: bool = False
    ):
        super().__init__()
        padding = kernel_size // 2
        output_padding = (0, 1)  # For stride=(1,2)
        
        # First layer with stride
        self.conv1 = nn.ConvTranspose2d(
            d_model, d_model,
            kernel_size=kernel_size,
            stride=(1, 2),
            padding=padding,
            output_padding=output_padding
        )
        
        # Second layer: stride depends on config
        if use_two_stride_layers:
            self.conv2 = nn.ConvTranspose2d(
                d_model, d_model,
                kernel_size=kernel_size,
                stride=(1, 2),
                padding=padding,
                output_padding=output_padding
            )
        else:
            self.conv2 = nn.Conv2d(
                d_model, d_model,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding
            )
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, d_model, H, W)
            
        Returns:
            Upsampled tensor of shape (B, d_model, H, W')
        """
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x


def match_shape(
    source: torch.Tensor,
    target_H: int,
    target_W: int
) -> torch.Tensor:
    """
    Match the spatial dimensions of source tensor to target dimensions.
    
    Uses center-crop if source is larger, or zero-pad if source is smaller.
    This avoids interpolation which could introduce bias.
    
    Args:
        source: Input tensor of shape (B, C, H, W)
        target_H: Target height
        target_W: Target width
        
    Returns:
        Tensor of shape (B, C, target_H, target_W)
        
    Raises:
        AssertionError: If H dimensions don't match (should be same period)
    """
    B, C, H, W = source.shape
    
    # H should match (same period), assert this
    assert H == target_H, f"Height mismatch: source H={H}, target H={target_H}. This is a bug."
    
    # Handle W dimension
    if W == target_W:
        return source
    elif W > target_W:
        # Center crop
        start = (W - target_W) // 2
        return source[:, :, :, start:start + target_W]
    else:
        # Zero pad
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left
        return F.pad(source, (pad_left, pad_right, 0, 0))


class AttentionPool(nn.Module):
    """
    Attention-based pooling for time dimension.
    
    Learns attention weights over time steps and computes weighted sum.
    
    Args:
        d_model: Model dimension
    
    Input Shape: (B, L, d_model)
    Output Shape: (B, d_model)
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            
        Returns:
            Pooled tensor of shape (B, d_model)
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (B, L, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, L, 1)
        
        # Weighted sum
        out = (x * attn_weights).sum(dim=1)  # (B, d_model)
        return out

