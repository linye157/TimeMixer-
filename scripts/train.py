#!/usr/bin/env python
"""
Training script for TimeMixer++ binary classification model.

Usage:
    python scripts/train.py --data_path TDdata/TrainData.csv --epochs 50 --batch_size 32

For a minimal test with random data:
    python scripts/train.py --use_random_data --epochs 2
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.data import load_file_strict, create_dataloaders, TemperatureDataset
from timemixerpp.utils import (
    set_seed, compute_metrics, save_checkpoint, 
    EarlyStopping, setup_logging, AverageMeter
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train TimeMixer++ for binary classification')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to training data (.xlsx or .csv)')
    parser.add_argument('--use_random_data', action='store_true',
                        help='Use random synthetic data for testing')
    parser.add_argument('--n_samples', type=int, default=128,
                        help='Number of samples for random data')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of MixerBlock layers')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Top-K frequencies for MRTI')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--pos_weight', type=float, default=None,
                        help='Positive class weight for BCEWithLogitsLoss')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path')
    
    return parser.parse_args()


def generate_random_data(n_samples: int = 128, seq_len: int = 48) -> tuple:
    """
    Generate random synthetic data for testing.
    
    Args:
        n_samples: Number of samples
        seq_len: Sequence length
        
    Returns:
        X: Features, shape (n_samples, seq_len)
        y: Binary labels, shape (n_samples,)
    """
    # Generate temperature-like data
    X = np.random.randn(n_samples, seq_len).astype(np.float32)
    
    # Add some temporal patterns
    for i in range(n_samples):
        # Add trend
        X[i] += np.linspace(0, 0.5, seq_len) * np.random.randn()
        # Add periodicity
        X[i] += 0.3 * np.sin(2 * np.pi * np.arange(seq_len) / 12)
    
    # Generate labels based on some pattern in the data
    # (High variance in latter half -> higher accident probability)
    late_variance = X[:, seq_len//2:].var(axis=1)
    threshold = np.median(late_variance)
    y = (late_variance > threshold).astype(np.float32)
    
    # Add some noise to labels
    noise_mask = np.random.rand(n_samples) < 0.1
    y[noise_mask] = 1 - y[noise_mask]
    
    return X, y


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).unsqueeze(-1)
        
        optimizer.zero_grad()
        
        output = model(batch_x)
        logits = output['logits']
        
        loss = criterion(logits, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        loss_meter.update(loss.item(), batch_x.size(0))
    
    return loss_meter.avg


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate model."""
    model.eval()
    loss_meter = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            
            output = model(batch_x)
            logits = output['logits']
            probs = output['probs']
            
            loss = criterion(logits, batch_y)
            loss_meter.update(loss.item(), batch_x.size(0))
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0).squeeze()
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = loss_meter.avg
    
    return loss_meter.avg, metrics


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("TimeMixer++ Training")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load or generate data
    if args.use_random_data:
        logger.info(f"Generating random data with {args.n_samples} samples")
        X, y = generate_random_data(args.n_samples)
    else:
        if args.data_path is None:
            raise ValueError("Must provide --data_path or use --use_random_data")
        _, X, y = load_file_strict(args.data_path)
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Positive samples: {y.sum():.0f} ({100*y.mean():.1f}%)")
    
    # Create dataloaders
    train_loader, val_loader, stats = create_dataloaders(
        X, y,
        batch_size=args.batch_size,
        val_split=args.val_split,
        normalize=True,
        seed=args.seed
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create or load model
    start_epoch = 0
    best_f1 = 0.0
    
    if args.resume:
        # Resume from checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Reconstruct config from checkpoint
        if 'config' in checkpoint:
            config = TimeMixerPPConfig(**checkpoint['config'])
            logger.info("Loaded config from checkpoint")
        else:
            config = TimeMixerPPConfig(
                seq_len=48, c_in=1, d_model=args.d_model,
                n_layers=args.n_layers, n_heads=args.n_heads,
                top_k=args.top_k, dropout=args.dropout,
                pos_weight=args.pos_weight
            )
        
        model = TimeMixerPPForBinaryCls(config).to(device)
        
        # Initialize dynamic layers by doing a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, config.seq_len, device=device)
            _ = model(dummy_input)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'metrics' in checkpoint and 'f1' in checkpoint['metrics']:
            best_f1 = checkpoint['metrics']['f1']
        
        logger.info(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
    else:
        # Create new model
        config = TimeMixerPPConfig(
            seq_len=48,
            c_in=1,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            top_k=args.top_k,
            dropout=args.dropout,
            pos_weight=args.pos_weight
        )
        model = TimeMixerPPForBinaryCls(config).to(device)
    
    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Dynamic M (scales): {config.compute_dynamic_M()}")
    logger.info(f"Scale lengths: {config.get_scale_lengths()}")
    
    # Loss function
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Load optimizer state if resuming
    if args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state from checkpoint")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
            f"AUROC: {metrics['auroc']:.4f}"
        )
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            save_checkpoint(
                model, optimizer, epoch, metrics, save_path,
                config=config.__dict__,
                normalizer_stats=stats
            )
            logger.info(f"New best model! F1: {best_f1:.4f}")
        
        # Early stopping
        if early_stopping(metrics['f1']):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model.pt')
    save_checkpoint(
        model, optimizer, epoch, metrics, final_path,
        config=config.__dict__,
        normalizer_stats=stats
    )
    
    logger.info("=" * 60)
    logger.info(f"Training complete! Best F1: {best_f1:.4f}")
    logger.info(f"Models saved to: {args.save_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

