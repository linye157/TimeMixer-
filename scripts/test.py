#!/usr/bin/env python
"""
Test/Evaluate script for TimeMixer++ binary classification model.

Evaluates a trained model on a test dataset and computes metrics.

Usage:
    python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/TestData.csv
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
from timemixerpp.utils import setup_logging, compute_metrics

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Test/Evaluate TimeMixer++ on a test set')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test data file (.xlsx or .csv)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (optional)')
    parser.add_argument('--output_features', action='store_true',
                        help='Also output multi-scale features')
    parser.add_argument('--features_output', type=str, default='test_features.npz',
                        help='Path to save features')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for predictions')
    parser.add_argument('--label_threshold', type=float, default=None,
                        help='Threshold for converting labels to binary (default: same as --threshold)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("=" * 60)
    logger.info("TimeMixer++ Test/Evaluation")
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
    
    # Load test data
    logger.info(f"Loading test data: {args.test_path}")
    _, X_test, y_test = load_file_strict(args.test_path)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    logger.info(f"Test positive samples: {y_test.sum():.0f} ({100*y_test.mean():.1f}%)")
    
    # Normalize
    if normalizer_mean is not None and normalizer_std is not None:
        logger.info("Applying normalization from checkpoint")
        X_test = (X_test - normalizer_mean) / normalizer_std
    else:
        logger.warning("No normalization stats in checkpoint, using raw data")
    
    # Evaluation
    all_probs = []
    all_features = []
    
    n_samples = len(X_test)
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size
    
    logger.info(f"Evaluating on {n_samples} samples...")
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, n_samples)
            
            batch_x = torch.tensor(X_test[start_idx:end_idx], dtype=torch.float32, device=device)
            
            if args.output_features:
                output = model(batch_x, return_features=True)
                features = output['features']
                batch_features = [f.cpu().numpy() for f in features]
                all_features.append(batch_features)
            else:
                output = model(batch_x)
            
            probs = output['probs'].cpu().numpy()
            all_probs.append(probs)
    
    # Combine results
    probs = np.concatenate(all_probs, axis=0).squeeze()  # (n,)
    predictions = (probs >= args.threshold).astype(int)
    
    # Compute metrics
    # Both predictions and labels are thresholded for computing classification metrics
    label_threshold = args.label_threshold if args.label_threshold is not None else args.threshold
    metrics = compute_metrics(y_test, probs, threshold=args.threshold, label_threshold=label_threshold)
    
    # Print detailed results
    logger.info("=" * 60)
    logger.info("Test Results:")
    logger.info("=" * 60)
    logger.info(f"  Samples:         {n_samples}")
    logger.info(f"  Pred Threshold:  {args.threshold}")
    logger.info(f"  Label Threshold: {label_threshold}")
    logger.info("-" * 40)
    logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:  {metrics['precision']:.4f}")
    logger.info(f"  Recall:     {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:   {metrics['f1']:.4f}")
    logger.info(f"  AUROC:      {metrics['auroc']:.4f}")
    logger.info("-" * 40)
    logger.info(f"  误报率 FPR: {metrics['fpr']:.4f}  (FP/(FP+TN))")
    logger.info(f"  漏报率 FNR: {metrics['fnr']:.4f}  (FN/(TP+FN))")
    logger.info("-" * 40)
    logger.info("Confusion Matrix:")
    logger.info(f"  TP: {metrics['tp']:4d}  |  FP: {metrics['fp']:4d}")
    logger.info(f"  FN: {metrics['fn']:4d}  |  TN: {metrics['tn']:4d}")
    logger.info("=" * 60)
    
    # Save predictions if requested
    if args.output:
        results = pd.DataFrame({
            'sample_id': np.arange(n_samples),
            'true_label': y_test.astype(int),
            'probability': probs,
            'prediction': predictions,
            'correct': (predictions == y_test).astype(int)
        })
        results.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to: {args.output}")
    
    # Save features if requested
    if args.output_features and all_features:
        num_scales = len(all_features[0])
        combined_features = {}
        
        for m in range(num_scales):
            scale_features = [batch[m] for batch in all_features]
            combined_features[f'scale_{m}'] = np.concatenate(scale_features, axis=0)
        
        np.savez(args.features_output, **combined_features)
        logger.info(f"Features saved to: {args.features_output}")
    
    # Return metrics for programmatic use
    return metrics


if __name__ == '__main__':
    main()

