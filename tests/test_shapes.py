#!/usr/bin/env python
"""
Unit tests for TimeMixer++ implementation.

Tests cover:
1. Shape consistency across modules
2. Dynamic M computation
3. Period deduplication
4. Pad/reshape/restore consistency
5. Edge cases with short sequences
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import torch
import numpy as np

from timemixerpp import (
    TimeMixerPPConfig,
    TimeMixerPPEncoder,
    TimeMixerPPForBinaryCls,
    MixerBlock,
    MRTI,
    TID,
    MCM,
    MRM
)
from timemixerpp.layers import (
    MultiHeadSelfAttention,
    DownsampleConv1d,
    ConvDown2d,
    TransConvUp2d,
    match_shape
)


class TestConfig:
    """Test configuration and dynamic M computation."""
    
    def test_dynamic_m_computation(self):
        """Test that dynamic M is computed correctly based on min_fft_len."""
        # T=48, min_fft_len=8 -> max M such that 48/2^M >= 8
        # 48/2^2 = 12 >= 8, 48/2^3 = 6 < 8 -> M = 2
        config = TimeMixerPPConfig(seq_len=48, min_fft_len=8)
        M = config.compute_dynamic_M()
        assert M == 2, f"Expected M=2, got M={M}"
        
        # Verify coarsest scale length
        lengths = config.get_scale_lengths()
        assert lengths[-1] >= config.min_fft_len, \
            f"Coarsest scale length {lengths[-1]} < min_fft_len {config.min_fft_len}"
    
    def test_dynamic_m_with_upper_bound(self):
        """Test that upper bound is respected."""
        config = TimeMixerPPConfig(
            seq_len=48,
            min_fft_len=4,
            max_scales_upper_bound=2
        )
        M = config.compute_dynamic_M()
        assert M <= config.max_scales_upper_bound
    
    def test_scale_lengths(self):
        """Test scale length computation."""
        config = TimeMixerPPConfig(seq_len=48)
        lengths = config.get_scale_lengths()
        
        # Should be halved at each scale
        for i in range(1, len(lengths)):
            assert lengths[i] == lengths[i-1] // 2


class TestLayers:
    """Test individual layer components."""
    
    def test_mhsa_shape(self):
        """Test multi-head self-attention output shape."""
        B, L, d_model = 2, 24, 64
        n_heads = 4
        
        mhsa = MultiHeadSelfAttention(d_model, n_heads)
        x = torch.randn(B, L, d_model)
        
        out = mhsa(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_downsample_conv1d(self):
        """Test 1D downsampling convolution."""
        B, L, C = 2, 48, 64
        
        conv = DownsampleConv1d(C, C)
        x = torch.randn(B, L, C)
        
        out = conv(x)
        expected_L = L // 2
        assert out.shape == (B, expected_L, C), \
            f"Expected (B, {expected_L}, C), got {out.shape}"
    
    def test_conv_down_2d(self):
        """Test 2D downsampling for MCM."""
        B, d_model, H, W = 2, 64, 6, 12
        
        conv = ConvDown2d(d_model)
        x = torch.randn(B, d_model, H, W)
        
        out = conv(x)
        # H should stay same, W should be reduced
        assert out.shape[2] == H, "H dimension should not change"
        assert out.shape[3] < W, "W dimension should decrease"
    
    def test_trans_conv_up_2d(self):
        """Test 2D upsampling for MCM."""
        B, d_model, H, W = 2, 64, 6, 6
        
        conv = TransConvUp2d(d_model)
        x = torch.randn(B, d_model, H, W)
        
        out = conv(x)
        # H should stay same, W should increase
        assert out.shape[2] == H, "H dimension should not change"
        assert out.shape[3] > W, "W dimension should increase"
    
    def test_match_shape_crop(self):
        """Test shape matching with cropping."""
        source = torch.randn(2, 64, 6, 12)
        target_W = 10
        
        matched = match_shape(source, 6, target_W)
        assert matched.shape == (2, 64, 6, target_W)
    
    def test_match_shape_pad(self):
        """Test shape matching with padding."""
        source = torch.randn(2, 64, 6, 8)
        target_W = 10
        
        matched = match_shape(source, 6, target_W)
        assert matched.shape == (2, 64, 6, target_W)


class TestMRTI:
    """Test Multi-Resolution Time Imaging module."""
    
    def test_period_computation(self):
        """Test that periods are computed and deduplicated."""
        mrti = MRTI(top_k=5, min_period=2)
        
        B, L_M, d_model = 2, 12, 64
        x_coarsest = torch.randn(B, L_M, d_model)
        
        periods, amplitudes = mrti.compute_periods_from_fft(x_coarsest, L_0=48)
        
        # Check that periods are unique
        assert len(periods) == len(set(periods)), "Periods should be unique"
        
        # Check that periods are valid
        for p in periods:
            assert p >= mrti.min_period, f"Period {p} < min_period {mrti.min_period}"
    
    def test_reshape_1d_to_2d_and_back(self):
        """Test that 1D -> 2D -> 1D preserves data."""
        mrti = MRTI()
        
        B, L, d_model = 2, 24, 64
        period = 6
        
        x = torch.randn(B, L, d_model)
        
        # 1D to 2D
        image, orig_len = mrti.reshape_1d_to_2d(x, period)
        assert orig_len == L
        
        # 2D back to 1D
        x_restored = mrti.reshape_2d_to_1d(image, orig_len)
        
        # Should be equal (within numerical precision)
        assert torch.allclose(x, x_restored, atol=1e-6), \
            "1D -> 2D -> 1D should preserve data"
    
    def test_forward_output_structure(self):
        """Test MRTI forward pass output structure."""
        mrti = MRTI(top_k=3)
        
        # Multi-scale input
        multi_scale_x = [
            torch.randn(2, 48, 64),  # x_0
            torch.randn(2, 24, 64),  # x_1
            torch.randn(2, 12, 64),  # x_2 (coarsest)
        ]
        
        time_images, periods, amplitudes = mrti(multi_scale_x)
        
        # Check that we have K_eff time images
        K_eff = len(periods)
        assert len(time_images) == K_eff
        
        # Each time image should have images for all scales
        for ti in time_images:
            assert len(ti.images) == 3  # M+1 = 3


class TestTID:
    """Test Time Image Decomposition module."""
    
    def test_dual_axis_attention_shapes(self):
        """Test that TID preserves shapes."""
        d_model, n_heads = 64, 4
        tid = TID(d_model, n_heads)
        
        # Time images for one period
        images = [
            torch.randn(2, d_model, 6, 8),
            torch.randn(2, d_model, 6, 4),
            torch.randn(2, d_model, 6, 2),
        ]
        
        seasonal, trend = tid(images)
        
        # Shapes should be preserved
        for s, t, z in zip(seasonal, trend, images):
            assert s.shape == z.shape, f"Seasonal shape {s.shape} != input shape {z.shape}"
            assert t.shape == z.shape, f"Trend shape {t.shape} != input shape {z.shape}"


class TestMCM:
    """Test Multi-Scale Mixing module."""
    
    def test_mcm_output_shapes(self):
        """Test MCM preserves original sequence lengths."""
        d_model = 64
        mcm = MCM(d_model)
        
        period = 6
        original_lengths = [48, 24, 12]
        
        # Create seasonal and trend images
        seasonal_images = [
            torch.randn(2, d_model, period, 8),
            torch.randn(2, d_model, period, 4),
            torch.randn(2, d_model, period, 2),
        ]
        trend_images = [
            torch.randn(2, d_model, period, 8),
            torch.randn(2, d_model, period, 4),
            torch.randn(2, d_model, period, 2),
        ]
        
        z_list = mcm(seasonal_images, trend_images, original_lengths, period)
        
        # Check output shapes match original lengths
        for m, z in enumerate(z_list):
            assert z.shape[1] == original_lengths[m], \
                f"Scale {m}: expected L={original_lengths[m]}, got {z.shape[1]}"


class TestMRM:
    """Test Multi-Resolution Mixing module."""
    
    def test_mrm_global_weights(self):
        """Test MRM with global weight mode."""
        mrm = MRM(weight_mode='global')
        
        # K=2 periods, M+1=3 scales
        z_per_period = [
            [torch.randn(2, 48, 64), torch.randn(2, 24, 64), torch.randn(2, 12, 64)],
            [torch.randn(2, 48, 64), torch.randn(2, 24, 64), torch.randn(2, 12, 64)],
        ]
        amplitudes = torch.tensor([[1.0, 0.5], [1.2, 0.6]])  # (B=2, K=2)
        
        x_list = mrm(z_per_period, amplitudes)
        
        # Check output shapes
        assert len(x_list) == 3
        assert x_list[0].shape == (2, 48, 64)
        assert x_list[1].shape == (2, 24, 64)
        assert x_list[2].shape == (2, 12, 64)
    
    def test_mrm_per_sample_weights(self):
        """Test MRM with per-sample weight mode."""
        mrm = MRM(weight_mode='per_sample')
        
        z_per_period = [
            [torch.randn(2, 48, 64), torch.randn(2, 24, 64)],
            [torch.randn(2, 48, 64), torch.randn(2, 24, 64)],
        ]
        amplitudes = torch.tensor([[1.0, 0.5], [1.2, 0.6]])
        
        x_list = mrm(z_per_period, amplitudes)
        
        assert len(x_list) == 2
        assert x_list[0].shape == (2, 48, 64)


class TestMixerBlock:
    """Test MixerBlock integration."""
    
    def test_mixer_block_residual(self):
        """Test that MixerBlock output has same shapes as input."""
        config = TimeMixerPPConfig(
            seq_len=48,
            d_model=64,
            n_heads=4,
            top_k=2
        )
        
        block = MixerBlock(config)
        
        # Multi-scale input
        multi_scale_x = [
            torch.randn(2, 48, 64),
            torch.randn(2, 24, 64),
            torch.randn(2, 12, 64),
        ]
        
        output = block(multi_scale_x)
        
        # Check shapes preserved
        for inp, out in zip(multi_scale_x, output):
            assert inp.shape == out.shape


class TestFullModel:
    """Test full model forward pass."""
    
    def test_encoder_forward(self):
        """Test encoder with both input shapes."""
        config = TimeMixerPPConfig(seq_len=48, d_model=32, n_layers=1)
        encoder = TimeMixerPPEncoder(config)
        
        # Test (B, 48) input
        x_2d = torch.randn(2, 48)
        out = encoder(x_2d)
        assert out.shape == (2, 48, 32)
        
        # Test (B, 48, 1) input
        x_3d = torch.randn(2, 48, 1)
        out = encoder(x_3d)
        assert out.shape == (2, 48, 32)
    
    def test_binary_cls_forward(self):
        """Test binary classification model."""
        config = TimeMixerPPConfig(seq_len=48, d_model=32, n_layers=1, top_k=2)
        model = TimeMixerPPForBinaryCls(config)
        
        x = torch.randn(2, 48)
        output = model(x)
        
        assert 'logits' in output
        assert 'probs' in output
        assert output['logits'].shape == (2, 1)
        assert output['probs'].shape == (2, 1)
        
        # Probs should be in [0, 1]
        assert (output['probs'] >= 0).all() and (output['probs'] <= 1).all()
    
    def test_get_multi_scale_features(self):
        """Test feature extraction method."""
        config = TimeMixerPPConfig(seq_len=48, d_model=32, n_layers=1)
        model = TimeMixerPPForBinaryCls(config)
        
        x = torch.randn(2, 48)
        features = model.get_multi_scale_features(x)
        
        # Should have M+1 feature tensors
        M = config.compute_dynamic_M()
        assert len(features) == M + 1
        
        # Check shapes
        lengths = config.get_scale_lengths()
        for m, (feat, L) in enumerate(zip(features, lengths)):
            assert feat.shape == (2, L, 32), \
                f"Scale {m}: expected (2, {L}, 32), got {feat.shape}"
    
    def test_backward_pass(self):
        """Test that gradients flow correctly."""
        config = TimeMixerPPConfig(seq_len=48, d_model=32, n_layers=1, top_k=2)
        model = TimeMixerPPForBinaryCls(config)
        
        x = torch.randn(2, 48, requires_grad=True)
        y = torch.tensor([[1.0], [0.0]])
        
        output = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output['logits'], y
        )
        
        loss.backward()
        
        # Check that some gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters())
        assert has_grad, "No gradients computed"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_sample_batch(self):
        """Test with batch size 1."""
        config = TimeMixerPPConfig(seq_len=48, d_model=32, n_layers=1)
        model = TimeMixerPPForBinaryCls(config)
        
        x = torch.randn(1, 48)
        output = model(x)
        
        assert output['logits'].shape == (1, 1)
    
    def test_k_eff_truncation(self):
        """Test that K is properly truncated for short sequences."""
        mrti = MRTI(top_k=10, min_period=2)  # top_k > available freqs
        
        # Very short coarsest scale
        x_coarsest = torch.randn(2, 8, 64)  # L_M = 8, only 4 positive freqs
        
        periods, amplitudes = mrti.compute_periods_from_fft(x_coarsest, L_0=48)
        
        # Should have at most 4 periods (from 4 positive frequencies)
        assert len(periods) <= 4
    
    def test_different_top_k_values(self):
        """Test model with different K values."""
        for k in [1, 2, 3, 5]:
            config = TimeMixerPPConfig(seq_len=48, d_model=32, n_layers=1, top_k=k)
            model = TimeMixerPPForBinaryCls(config)
            
            x = torch.randn(2, 48)
            output = model(x)
            
            assert output['logits'].shape == (2, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

