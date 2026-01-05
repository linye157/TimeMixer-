#!/usr/bin/env python
"""
Inference script for TimeMixer++ binary classification model.

Usage:
    python scripts/infer.py --checkpoint checkpoints/best_model.pt --input data.csv --output predictions.csv

For testing with random data:
    python scripts/infer.py --checkpoint checkpoints/best_model.pt --use_random_input --n_samples 10
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import torch

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.data import load_file_strict
from timemixerpp.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with TimeMixer++')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Input arguments
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input data file (.xlsx, .csv, or .npy)')
    parser.add_argument('--use_random_input', action='store_true',
                        help='Use random input for testing')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of random samples')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to save predictions')
    parser.add_argument('--output_features', action='store_true',
                        help='Also output multi-scale features')
    parser.add_argument('--features_output', type=str, default='features.npz',
                        help='Path to save features')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    
    return parser.parse_args()


def load_input_data(input_path: str) -> np.ndarray:
    """
    Load input data from various formats.
    
    Args:
        input_path: Path to input file
        
    Returns:
        X: Input array, shape (n, 48)
    """
    if input_path.endswith('.npy'):
        X = np.load(input_path)
    elif input_path.endswith('.xlsx') or input_path.endswith('.xls') or input_path.endswith('.csv'):
        _, X, _ = load_file_strict(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")
    
    return X.astype(np.float32)


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("=" * 60)
    logger.info("TimeMixer++ Inference")
    logger.info("=" * 60)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Reconstruct config
    if 'config' in checkpoint:
        config = TimeMixerPPConfig(**checkpoint['config'])
    else:
        logger.warning("No config in checkpoint, using defaults")
        config = TimeMixerPPConfig()
    
    # Create model
    model = TimeMixerPPForBinaryCls(config).to(device)
    
    # Initialize dynamic layers by doing a dummy forward pass
    # This is needed because some layers (LayerNorms, MCM convs) are lazily initialized
    with torch.no_grad():
        dummy_input = torch.randn(1, config.seq_len, device=device)
        _ = model(dummy_input)
    
    # Now load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load normalization stats
    normalizer_mean = checkpoint.get('normalizer_mean', None)
    normalizer_std = checkpoint.get('normalizer_std', None)
    
    # Load input data
    if args.use_random_input:
        logger.info(f"Generating {args.n_samples} random samples")
        X = np.random.randn(args.n_samples, 48).astype(np.float32)
    else:
        if args.input is None:
            raise ValueError("Must provide --input or use --use_random_input")
        X = load_input_data(args.input)
    
    logger.info(f"Input shape: {X.shape}")
    
    # Normalize
    if normalizer_mean is not None and normalizer_std is not None:
        logger.info("Applying normalization from checkpoint")
        X = (X - normalizer_mean) / normalizer_std
    
    # Inference
    all_probs = []
    all_features = []
    
    n_samples = len(X)
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size
    
    logger.info(f"Running inference on {n_samples} samples...")
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, n_samples)
            
            batch_x = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=device)
            
            if args.output_features:
                output = model(batch_x, return_features=True)
                features = output['features']
                # Store features as list of numpy arrays
                batch_features = [f.cpu().numpy() for f in features]
                all_features.append(batch_features)
            else:
                output = model(batch_x)
            
            probs = output['probs'].cpu().numpy()
            all_probs.append(probs)
    
    # Combine results
    probs = np.concatenate(all_probs, axis=0).squeeze()  # (n,)
    predictions = (probs >= args.threshold).astype(int)
    
    # Create output DataFrame
    results = pd.DataFrame({
        'sample_id': np.arange(n_samples),
        'probability': probs,
        'prediction': predictions
    })
    
    # Save predictions
    results.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to: {args.output}")
    
    # Save features if requested
    if args.output_features and all_features:
        # Combine features across batches
        num_scales = len(all_features[0])
        combined_features = {}
        
        for m in range(num_scales):
            scale_features = [batch[m] for batch in all_features]
            combined_features[f'scale_{m}'] = np.concatenate(scale_features, axis=0)
        
        np.savez(args.features_output, **combined_features)
        logger.info(f"Features saved to: {args.features_output}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Inference Summary:")
    logger.info(f"  Total samples: {n_samples}")
    logger.info(f"  Predicted positive: {predictions.sum()} ({100*predictions.mean():.1f}%)")
    logger.info(f"  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    logger.info(f"  Mean probability: {probs.mean():.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

