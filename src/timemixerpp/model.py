"""
TimeMixer++ Model

Main model classes:
- TimeMixerPPEncoder: Core encoder with multi-scale processing and MixerBlocks
- TimeMixerPPForBinaryCls: Binary classification head on top of encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any

from .config import TimeMixerPPConfig
from .block import MixerBlock
from .layers import DownsampleConv1d, MultiHeadSelfAttention, AttentionPool


class MultiScaleGenerator(nn.Module):
    """
    Generates multi-scale time series by recursive downsampling.
    
    Given x_0, produces {x_0, x_1, ..., x_M} where x_m = Conv(x_{m-1}, stride=2).
    
    Dynamic M is computed based on min_fft_len to ensure the coarsest scale
    has enough points for meaningful FFT computation.
    
    Args:
        config: TimeMixerPPConfig
    """
    
    def __init__(self, config: TimeMixerPPConfig):
        super().__init__()
        self.config = config
        self.M = config.compute_dynamic_M()
        
        # Create downsampling convolutions
        self.down_convs = nn.ModuleList([
            DownsampleConv1d(
                c_in=config.d_model,
                c_out=config.d_model,
                kernel_size=config.down_kernel_size,
                padding=config.down_padding,
                groups=config.down_groups,
                use_norm=config.use_down_norm
            )
            for _ in range(self.M)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multi-scale representations.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            
        Returns:
            multi_scale_x: List of M+1 tensors [x_0, x_1, ..., x_M]
                          x_m has shape (B, L_m, d_model) where L_m = L // 2^m
        """
        multi_scale_x = [x]  # x_0
        
        current = x
        for m in range(self.M):
            current = self.down_convs[m](current)
            multi_scale_x.append(current)
        
        return multi_scale_x


class InputProjection(nn.Module):
    """
    Projects input from c_in channels to d_model dimensions.
    
    For each scale, applies: Linear(c_in -> d_model) + Dropout
    
    Optionally includes channel attention for C > 1 (variate-wise self-attention
    at coarsest scale for channel mixing).
    
    Args:
        config: TimeMixerPPConfig
    """
    
    def __init__(self, config: TimeMixerPPConfig):
        super().__init__()
        self.config = config
        
        # Embedding projection
        self.embed = nn.Sequential(
            nn.Linear(config.c_in, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Channel attention (optional, for C > 1)
        self.enable_channel_attn = config.enable_channel_attn and config.c_in > 1
        if self.enable_channel_attn:
            self.channel_attn = MultiHeadSelfAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to d_model dimensions.
        
        Args:
            x: Input tensor of shape (B, L, c_in)
            
        Returns:
            Projected tensor of shape (B, L, d_model)
        """
        x = self.embed(x)  # (B, L, d_model)
        return x


class OutputHead(nn.Module):
    """
    Output head for multi-scale representations.
    
    For each scale m:
    1. Pool along time dimension (mean or attention)
    2. Linear projection to output dimension
    
    Then ensemble across scales (mean or learned weights).
    
    Args:
        config: TimeMixerPPConfig
        output_dim: Output dimension (1 for binary classification)
    """
    
    def __init__(self, config: TimeMixerPPConfig, output_dim: int = 1):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        
        # Pooling
        self.pool_type = config.pool_type
        if self.pool_type == 'attention':
            self.pool = AttentionPool(config.d_model)
        
        # Linear head (shared across scales or per-scale)
        self.head = nn.Linear(config.d_model, output_dim)
        
        # Ensemble weights (learned if ensemble_type == 'learned')
        self.ensemble_type = config.ensemble_type
        self._ensemble_weights = None
        self._num_scales = None
    
    def _init_ensemble_weights(self, num_scales: int, device: torch.device):
        """Initialize ensemble weights for learned ensemble."""
        if self._num_scales == num_scales:
            return
        
        self._num_scales = num_scales
        if self.ensemble_type == 'learned':
            self._ensemble_weights = nn.Parameter(
                torch.ones(num_scales, device=device)
            )
    
    def forward(self, multi_scale_x: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute output from multi-scale representations.
        
        Args:
            multi_scale_x: List of M+1 tensors, each (B, L_m, d_model)
            
        Returns:
            logits: Output tensor of shape (B, output_dim)
        """
        num_scales = len(multi_scale_x)
        device = multi_scale_x[0].device
        
        self._init_ensemble_weights(num_scales, device)
        
        # Compute logits for each scale
        logits_list = []
        
        for x_m in multi_scale_x:
            # Pool
            if self.pool_type == 'mean':
                pooled = x_m.mean(dim=1)  # (B, d_model)
            else:
                pooled = self.pool(x_m)  # (B, d_model)
            
            # Head
            logit = self.head(pooled)  # (B, output_dim)
            logits_list.append(logit)
        
        # Ensemble
        logits_stack = torch.stack(logits_list, dim=0)  # (M+1, B, output_dim)
        
        if self.ensemble_type == 'mean':
            logits = logits_stack.mean(dim=0)  # (B, output_dim)
        else:
            # Learned weights with softmax
            weights = F.softmax(self._ensemble_weights, dim=0)  # (M+1,)
            weights = weights.view(-1, 1, 1)  # (M+1, 1, 1)
            logits = (weights * logits_stack).sum(dim=0)  # (B, output_dim)
        
        return logits


class TimeMixerPPEncoder(nn.Module):
    """
    TimeMixer++ Encoder.
    
    Architecture:
    1. Input projection (c_in -> d_model)
    2. Multi-scale generation (recursive downsampling)
    3. L stacked MixerBlocks
    
    Args:
        config: TimeMixerPPConfig
    """
    
    def __init__(self, config: TimeMixerPPConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = InputProjection(config)
        
        # Multi-scale generator
        self.multi_scale_gen = MultiScaleGenerator(config)
        
        # Stacked MixerBlocks
        self.blocks = nn.ModuleList([
            MixerBlock(config)
            for _ in range(config.n_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        return_multi_scale: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (B, L, c_in) or (B, L)
            return_multi_scale: If True, also return multi-scale features
            
        Returns:
            If return_multi_scale=False:
                x_0: Finest scale output, shape (B, L, d_model)
            If return_multi_scale=True:
                x_0: Finest scale output, shape (B, L, d_model)
                multi_scale_x: List of all scale outputs
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L) -> (B, L, 1)
        
        # Input projection
        x = self.input_proj(x)  # (B, L, d_model)
        
        # Generate multi-scale representations
        multi_scale_x = self.multi_scale_gen(x)
        
        # Apply MixerBlocks
        for block in self.blocks:
            multi_scale_x = block(multi_scale_x)
        
        if return_multi_scale:
            return multi_scale_x[0], multi_scale_x
        else:
            return multi_scale_x[0]


class TimeMixerPPForBinaryCls(nn.Module):
    """
    TimeMixer++ for Binary Classification.
    
    Combines TimeMixerPPEncoder with output head for binary classification.
    
    Args:
        config: TimeMixerPPConfig
        
    Input:
        x: (B, 48) or (B, 48, 1) temperature sequence
        
    Output:
        logits: (B, 1) raw logits
        probs: (B, 1) sigmoid probabilities
        features (optional): List of multi-scale features
    """
    
    def __init__(self, config: TimeMixerPPConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = TimeMixerPPEncoder(config)
        
        # Output head
        self.head = OutputHead(config, output_dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass for binary classification.
        
        Args:
            x: Input tensor of shape (B, 48) or (B, 48, 1)
            return_features: If True, include multi-scale features in output
            
        Returns:
            Dict with keys:
                - 'logits': (B, 1) raw logits
                - 'probs': (B, 1) probabilities (sigmoid of logits)
                - 'features': (optional) List of multi-scale features
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L) -> (B, L, 1)
        
        # Encode
        if return_features:
            _, multi_scale_x = self.encoder(x, return_multi_scale=True)
        else:
            multi_scale_x = self.encoder.multi_scale_gen(
                self.encoder.input_proj(x)
            )
            for block in self.encoder.blocks:
                multi_scale_x = block(multi_scale_x)
        
        # Output head
        logits = self.head(multi_scale_x)  # (B, 1)
        probs = torch.sigmoid(logits)  # (B, 1)
        
        result = {
            'logits': logits,
            'probs': probs
        }
        
        if return_features:
            result['features'] = multi_scale_x
        
        return result
    
    def get_multi_scale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Explicitly output multi-scale multi-resolution time series features.
        
        This is the method requested in the specification for obtaining
        the final XL multi-scale representation list.
        
        Args:
            x: Input tensor of shape (B, 48) or (B, 48, 1)
            
        Returns:
            features: List of M+1 tensors [x_L_0, ..., x_L_M]
                     Each x_L_m has shape (B, L_m, d_model)
        """
        output = self.forward(x, return_features=True)
        return output['features']

