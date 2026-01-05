#!/usr/bin/env python
"""
Extract multi-scale features from TimeMixer++ model.

This script loads a trained model and extracts the multi-scale features
(output before the classification head) for a given dataset.

Usage:
    # Extract features from a dataset
    python scripts/extract_features.py --checkpoint checkpoints/best_model.pt --data_path TDdata/TrainData.csv --output features/train_features.npz

    # View saved features
    python scripts/extract_features.py --view features/train_features.npz
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import logging
import numpy as np
import torch

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.data import load_file_strict
from timemixerpp.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract multi-scale features from TimeMixer++')
    
    # Mode: extract or view
    parser.add_argument('--view', type=str, default=None,
                        help='Path to saved features file to view (skip extraction)')
    
    # Extraction arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file (.xlsx or .csv)')
    parser.add_argument('--output', type=str, default='features/extracted_features.npz',
                        help='Path to save extracted features')
    
    # Other arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--save_labels', action='store_true',
                        help='Also save labels in the output file')
    
    return parser.parse_args()


def view_features(feature_path: str):
    """
    View saved multi-scale features.
    
    Args:
        feature_path: Path to .npz file containing features
    """
    logger.info(f"Loading features from: {feature_path}")
    
    data = np.load(feature_path, allow_pickle=True)
    
    print("\n" + "=" * 60)
    print(" Multi-Scale Features Summary")
    print("=" * 60)
    
    # List all arrays in the file
    print(f"\nKeys in file: {list(data.keys())}")
    
    # Show shape and statistics for each scale
    scale_keys = [k for k in data.keys() if k.startswith('scale_')]
    scale_keys.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"\nNumber of scales: {len(scale_keys)}")
    print("-" * 60)
    
    for key in scale_keys:
        arr = data[key]
        print(f"\n{key}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Min:   {arr.min():.6f}")
        print(f"  Max:   {arr.max():.6f}")
        print(f"  Mean:  {arr.mean():.6f}")
        print(f"  Std:   {arr.std():.6f}")
        
        # Show first sample's first few values
        if arr.ndim >= 2:
            print(f"  First sample, first 5 time steps, first 3 dims:")
            sample = arr[0, :5, :3] if arr.ndim == 3 else arr[0, :5]
            print(f"  {sample}")
    
    # Show labels if present
    if 'labels' in data:
        labels = data['labels']
        print(f"\nLabels:")
        print(f"  Shape: {labels.shape}")
        print(f"  Unique values: {np.unique(labels)[:10]}...")
        print(f"  Positive ratio: {labels.mean():.4f}")
    
    # Show metadata if present
    if 'config' in data:
        print(f"\nModel config: {data['config']}")
    
    print("\n" + "=" * 60)


def extract_features(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    normalizer_mean: np.ndarray = None,
    normalizer_std: np.ndarray = None
) -> dict:
    """
    Extract multi-scale features from the model.
    
    Args:
        model: TimeMixerPPForBinaryCls model
        X: Input data, shape (n, 48)
        batch_size: Batch size
        device: Device
        normalizer_mean: Normalization mean
        normalizer_std: Normalization std
        
    Returns:
        Dictionary with scale features
    """
    model.eval()
    
    # Normalize if stats provided
    if normalizer_mean is not None and normalizer_std is not None:
        X = (X - normalizer_mean) / normalizer_std
    
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Collect features for each scale
    all_features = {}
    
    logger.info(f"Extracting features from {n_samples} samples...")
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_x = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=device)
            
            # Get multi-scale features
            features = model.get_multi_scale_features(batch_x)
            
            # Store features by scale
            for m, feat in enumerate(features):
                key = f'scale_{m}'
                feat_np = feat.cpu().numpy()
                
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(feat_np)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {end_idx}/{n_samples} samples")
    
    # Concatenate all batches
    for key in all_features:
        all_features[key] = np.concatenate(all_features[key], axis=0)
    
    return all_features


def main():
    args = parse_args()
    setup_logging()
    
    # View mode
    if args.view:
        view_features(args.view)
        return
    
    # Extraction mode
    if args.checkpoint is None:
        raise ValueError("Must provide --checkpoint for feature extraction")
    if args.data_path is None:
        raise ValueError("Must provide --data_path for feature extraction")
    
    logger.info("=" * 60)
    logger.info("TimeMixer++ Feature Extraction")
    logger.info("=" * 60)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Reconstruct config
    if 'config' in checkpoint:
        config = TimeMixerPPConfig(**checkpoint['config'])
    else:
        config = TimeMixerPPConfig()
    
    # Create model
    model = TimeMixerPPForBinaryCls(config).to(device)
    
    # Initialize dynamic layers
    with torch.no_grad():
        dummy_input = torch.randn(1, config.seq_len, device=device)
        _ = model(dummy_input)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded. M={config.compute_dynamic_M()}, scales={config.get_scale_lengths()}")
    
    # Load normalization stats
    normalizer_mean = checkpoint.get('normalizer_mean', None)
    normalizer_std = checkpoint.get('normalizer_std', None)
    
    # Load data
    logger.info(f"Loading data: {args.data_path}")
    _, X, y = load_file_strict(args.data_path)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Extract features
    features = extract_features(
        model, X, args.batch_size, device,
        normalizer_mean, normalizer_std
    )
    
    # Add labels and config if requested
    if args.save_labels:
        features['labels'] = y
    features['config'] = np.array(str(checkpoint.get('config', {})))
    
    # Save features
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    np.savez(args.output, **features)
    
    logger.info(f"Features saved to: {args.output}")
    
    # Show summary
    print("\n" + "=" * 60)
    print(" Extracted Features Summary")
    print("=" * 60)
    for key, arr in features.items():
        if isinstance(arr, np.ndarray) and arr.dtype != object:
            print(f"  {key}: {arr.shape}")
    print("=" * 60)
    
    # Optionally view
    print(f"\nTo view features: python scripts/extract_features.py --view {args.output}")


if __name__ == '__main__':
    main()

