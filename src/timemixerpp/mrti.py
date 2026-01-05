"""
MRTI: Multi-Resolution Time Imaging

Implements the MRTI module from TimeMixer++ (Equations 5-6).
Converts multi-scale 1D time series to multi-resolution 2D time images
using FFT-based period detection.

Key Features:
- FFT-based top-K frequency selection on coarsest scale
- Period calculation with deduplication
- 1D to 2D reshape with padding for each scale and period
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class TimeImageInfo:
    """
    Information about time images for a single period.
    
    Attributes:
        period: The period length (p_k)
        amplitude: The amplitude weight for this period
        images: List of 2D time images for each scale, shape (B, d_model, H=p_k, W=f_m_k)
        original_lengths: Original sequence lengths for each scale before padding
    """
    period: int
    amplitude: torch.Tensor  # (B,) or scalar
    images: List[torch.Tensor]  # M+1 images, one per scale
    original_lengths: List[int]  # Original L_m values


class MRTI(nn.Module):
    """
    Multi-Resolution Time Imaging module.
    
    Converts multi-scale 1D representations to multi-resolution 2D time images
    using FFT-based period detection on the coarsest scale.
    
    Args:
        top_k: Number of top frequencies to select
        base_len_for_period: 'coarsest' or 'original' - base for period calculation
        min_period: Minimum period length
        max_period_factor: Maximum period as factor of L_0 (original length)
    
    Why dynamic K/period is needed for short sequences:
        For T=48, the coarsest scale might have L_M=8 or 12.
        FFT on such short sequences has limited frequency resolution.
        We automatically truncate K to available meaningful frequencies
        and clamp periods to valid ranges.
    """
    
    def __init__(
        self,
        top_k: int = 3,
        base_len_for_period: str = 'coarsest',
        min_period: int = 2,
        max_period_factor: float = 1.0
    ):
        super().__init__()
        self.top_k = top_k
        self.base_len_for_period = base_len_for_period
        self.min_period = min_period
        self.max_period_factor = max_period_factor
    
    def compute_periods_from_fft(
        self,
        x_coarsest: torch.Tensor,
        L_0: int
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Compute top-K periods from FFT on coarsest scale.
        
        Args:
            x_coarsest: Coarsest scale representation, shape (B, L_M, d_model)
            L_0: Original sequence length
            
        Returns:
            periods: List of unique period lengths (sorted descending by amplitude)
            amplitudes: Tensor of amplitudes, shape (B, K_eff) or (K_eff,)
        """
        B, L_M, d_model = x_coarsest.shape
        
        # Aggregate over d_model dimension (mean)
        # This gives a single time series per batch sample
        x_agg = x_coarsest.mean(dim=-1)  # (B, L_M)
        
        # Compute FFT
        fft_out = torch.fft.rfft(x_agg, dim=-1)  # (B, L_M//2 + 1)
        amplitudes = torch.abs(fft_out)  # (B, num_freqs)
        
        # Ignore DC component (index 0)
        amplitudes_no_dc = amplitudes[:, 1:]  # (B, num_freqs - 1)
        num_freqs = amplitudes_no_dc.shape[1]
        
        if num_freqs == 0:
            # Edge case: very short sequence
            return [self.min_period], torch.ones(B, 1, device=x_coarsest.device)
        
        # Determine K (truncate if not enough frequencies)
        K = min(self.top_k, num_freqs)
        
        # Get top-K frequencies (indices are 1-based due to DC removal)
        # Average amplitudes across batch for stable period selection
        mean_amplitudes = amplitudes_no_dc.mean(dim=0)  # (num_freqs,)
        top_k_values, top_k_indices = torch.topk(mean_amplitudes, K)
        
        # Convert indices to frequency indices (add 1 for DC offset)
        freq_indices = top_k_indices + 1  # (K,)
        
        # Calculate periods
        # p_k = base_len / f_k
        if self.base_len_for_period == 'coarsest':
            base_len = L_M
        else:
            base_len = L_0
        
        max_period = int(L_0 * self.max_period_factor)
        
        # Compute periods and deduplicate
        period_to_amplitude = {}
        for i in range(K):
            f_idx = freq_indices[i].item()
            if f_idx == 0:
                continue
            
            # Period = base_len / frequency_index
            p = int(round(base_len / f_idx))
            p = max(self.min_period, min(p, max_period))
            
            amp = top_k_values[i].item()
            
            # Keep the one with higher amplitude if duplicate
            if p not in period_to_amplitude or amp > period_to_amplitude[p]:
                period_to_amplitude[p] = amp
        
        # Sort by amplitude (descending)
        sorted_periods = sorted(period_to_amplitude.items(), key=lambda x: -x[1])
        
        if len(sorted_periods) == 0:
            # Fallback
            return [self.min_period], torch.ones(B, 1, device=x_coarsest.device)
        
        periods = [p for p, _ in sorted_periods]
        amp_values = [a for _, a in sorted_periods]
        
        # Create amplitude tensor
        # For per_sample mode, we need per-batch amplitudes
        # Use the indices to get batch-specific amplitudes
        K_eff = len(periods)
        batch_amplitudes = torch.zeros(B, K_eff, device=x_coarsest.device)
        
        for k, period in enumerate(periods):
            # Find which frequency index gave this period
            for i in range(K):
                f_idx = freq_indices[i].item()
                if f_idx == 0:
                    continue
                p = int(round(base_len / f_idx))
                p = max(self.min_period, min(p, max_period))
                if p == period:
                    # Get batch-specific amplitude for this frequency
                    batch_amplitudes[:, k] = amplitudes_no_dc[:, freq_indices[i] - 1]
                    break
        
        return periods, batch_amplitudes
    
    def reshape_1d_to_2d(
        self,
        x: torch.Tensor,
        period: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Reshape 1D time series to 2D time image.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            period: Period length (becomes height H)
            
        Returns:
            image: 2D time image, shape (B, d_model, H=period, W=ceil(L/period))
            original_length: Original sequence length L
        """
        B, L, d_model = x.shape
        
        # Calculate padded length
        W = math.ceil(L / period)
        pad_len = period * W
        
        # Pad if necessary
        if pad_len > L:
            padding = torch.zeros(B, pad_len - L, d_model, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)  # (B, pad_len, d_model)
        else:
            x_padded = x
        
        # Reshape: (B, period * W, d_model) -> (B, period, W, d_model)
        x_2d = x_padded.reshape(B, period, W, d_model)
        
        # Permute to channel-first: (B, d_model, H=period, W)
        x_2d = x_2d.permute(0, 3, 1, 2)
        
        return x_2d, L
    
    def reshape_2d_to_1d(
        self,
        image: torch.Tensor,
        original_length: int
    ) -> torch.Tensor:
        """
        Reshape 2D time image back to 1D time series.
        
        Args:
            image: 2D time image, shape (B, d_model, H=period, W)
            original_length: Original sequence length to truncate to
            
        Returns:
            x: 1D time series, shape (B, original_length, d_model)
        """
        B, d_model, H, W = image.shape
        
        # Permute: (B, d_model, H, W) -> (B, H, W, d_model)
        x = image.permute(0, 2, 3, 1)
        
        # Reshape: (B, H, W, d_model) -> (B, H*W, d_model)
        x = x.reshape(B, H * W, d_model)
        
        # Truncate to original length
        x = x[:, :original_length, :]
        
        return x
    
    def forward(
        self,
        multi_scale_x: List[torch.Tensor]
    ) -> Tuple[List[TimeImageInfo], List[int], torch.Tensor]:
        """
        Forward pass: convert multi-scale 1D to multi-resolution 2D images.
        
        Args:
            multi_scale_x: List of tensors [x_0, x_1, ..., x_M]
                          Each x_m has shape (B, L_m, d_model)
                          
        Returns:
            time_images: List of TimeImageInfo, one per period
            periods: List of period lengths
            amplitudes: Amplitude weights, shape (B, K_eff)
        """
        M = len(multi_scale_x) - 1
        x_coarsest = multi_scale_x[-1]  # x_M
        L_0 = multi_scale_x[0].shape[1]  # Original sequence length
        
        # Step 1: Compute periods from FFT on coarsest scale
        periods, amplitudes = self.compute_periods_from_fft(x_coarsest, L_0)
        
        assert len(periods) > 0, "No valid periods found. Check input sequence."
        
        # Step 2: For each period, create 2D images for all scales
        time_images = []
        
        for k, period in enumerate(periods):
            images = []
            original_lengths = []
            
            for m, x_m in enumerate(multi_scale_x):
                image, orig_len = self.reshape_1d_to_2d(x_m, period)
                images.append(image)
                original_lengths.append(orig_len)
            
            time_image_info = TimeImageInfo(
                period=period,
                amplitude=amplitudes[:, k] if amplitudes.dim() > 1 else amplitudes[k],
                images=images,
                original_lengths=original_lengths
            )
            time_images.append(time_image_info)
        
        return time_images, periods, amplitudes

