"""
Configuration dataclass for TimeMixer++ model.

All hyperparameters are centralized here with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class TimeMixerPPConfig:
    """
    Configuration for TimeMixer++ model.
    
    Attributes:
        seq_len: Input sequence length (default: 48 for temperature data)
        c_in: Number of input channels (default: 1 for univariate)
        d_model: Hidden dimension size
        n_layers: Number of MixerBlock layers
        
        # Multi-scale parameters
        min_fft_len: Minimum length for FFT computation (determines max M)
        max_scales_upper_bound: Upper bound for number of scales
        allow_fixed_M: Whether to allow fixed M specification
        fixed_M: Fixed number of scales if allow_fixed_M is True
        
        # Downsampling conv parameters
        down_kernel_size: Kernel size for downsampling Conv1d
        down_padding: Padding for downsampling Conv1d
        down_groups: Groups for downsampling Conv1d
        use_down_norm: Whether to use normalization after downsampling
        
        # MRTI parameters
        top_k: Number of top frequencies to select
        base_len_for_period: Base length for period calculation ('coarsest' or 'original')
        min_period: Minimum period length
        
        # TID parameters
        n_heads: Number of attention heads
        tid_conv_kernel: Kernel size for TID Conv2d
        
        # MCM parameters
        mcm_kernel_size: Kernel size for MCM Conv2d
        mcm_use_two_stride_layers: Whether to use stride in both conv layers
        
        # MRM parameters
        weight_mode: Amplitude weight mode ('global' or 'per_sample')
        
        # General parameters
        dropout: Dropout rate
        activation: Activation function name
        
        # Channel attention (for C > 1)
        enable_channel_attn: Enable channel attention at coarsest scale
        
        # Output head parameters
        pool_type: Pooling type for output head ('mean' or 'attention')
        ensemble_type: Ensemble type for multi-scale outputs ('mean' or 'learned')
        
        # Training parameters
        pos_weight: Positive class weight for BCEWithLogitsLoss
    """
    # Input dimensions
    seq_len: int = 48
    c_in: int = 1
    d_model: int = 64
    n_layers: int = 2
    
    # Multi-scale parameters
    min_fft_len: int = 8
    max_scales_upper_bound: int = 6
    allow_fixed_M: bool = False
    fixed_M: Optional[int] = None
    
    # Downsampling conv parameters
    down_kernel_size: int = 3
    down_padding: int = 1
    down_groups: int = 1
    use_down_norm: bool = False
    
    # MRTI parameters
    top_k: int = 3
    base_len_for_period: Literal['coarsest', 'original'] = 'coarsest'
    min_period: int = 2
    
    # TID parameters
    n_heads: int = 4
    tid_conv_kernel: int = 1
    
    # MCM parameters
    mcm_kernel_size: int = 3
    mcm_use_two_stride_layers: bool = False
    
    # MRM parameters
    weight_mode: Literal['global', 'per_sample'] = 'per_sample'
    
    # General parameters
    dropout: float = 0.1
    activation: str = 'gelu'
    
    # Channel attention (for C > 1)
    enable_channel_attn: bool = False
    
    # Output head parameters
    pool_type: Literal['mean', 'attention'] = 'mean'
    ensemble_type: Literal['mean', 'learned'] = 'learned'
    
    # Training parameters
    pos_weight: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.seq_len > 0, "seq_len must be positive"
        assert self.c_in > 0, "c_in must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.min_fft_len >= 4, "min_fft_len must be at least 4 for meaningful FFT"
        assert self.top_k > 0, "top_k must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.min_period >= 2, "min_period must be at least 2"
        
        if self.allow_fixed_M:
            assert self.fixed_M is not None, "fixed_M must be specified when allow_fixed_M is True"
            assert self.fixed_M > 0, "fixed_M must be positive"
    
    def compute_dynamic_M(self) -> int:
        """
        Compute the dynamic number of scales M based on sequence length and min_fft_len.
        
        For short sequences (like T=48), we need to ensure the coarsest scale
        has at least min_fft_len points for meaningful FFT computation.
        
        Returns:
            M: Number of scales (including the original scale at index 0)
        """
        if self.allow_fixed_M and self.fixed_M is not None:
            # Validate that fixed_M doesn't violate min_fft_len constraint
            coarsest_len = self.seq_len // (2 ** self.fixed_M)
            if coarsest_len < self.min_fft_len:
                import logging
                logging.warning(
                    f"fixed_M={self.fixed_M} results in coarsest_len={coarsest_len} < min_fft_len={self.min_fft_len}. "
                    f"Adjusting M dynamically."
                )
            else:
                return self.fixed_M
        
        # Compute maximum M such that L_M >= min_fft_len
        # L_M = seq_len / 2^M >= min_fft_len
        # 2^M <= seq_len / min_fft_len
        # M <= log2(seq_len / min_fft_len)
        import math
        max_M_for_fft = int(math.log2(self.seq_len / self.min_fft_len))
        M = min(self.max_scales_upper_bound, max_M_for_fft)
        M = max(M, 1)  # At least 1 scale (M=0 means only original, M=1 means one downsampling)
        
        return M
    
    def get_scale_lengths(self) -> list:
        """
        Get the sequence lengths at each scale.
        
        Returns:
            List of lengths [L_0, L_1, ..., L_M] where L_0 = seq_len
        """
        M = self.compute_dynamic_M()
        lengths = [self.seq_len // (2 ** m) for m in range(M + 1)]
        return lengths

