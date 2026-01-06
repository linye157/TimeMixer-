#!/usr/bin/env python
"""
训练序列级 embedding 编码器（SupCon + 可选 BCE）。

使用已提取的多尺度特征训练 TemporalConvEmbedder，
输出可用于向量检索的 L2 归一化 embedding。

升级为概率预测模式：
- 评估使用三尺度融合后的概率 p = w0*p0 + w1*p1 + w2*p2
- 以 NLL/Brier 作为 checkpoint 选择标准
- 支持 float 标签（0~1 软标签）

Usage:
   # 训练（概率预测模式）
python scripts/train_embedding.py --npz_path features/alldata_features_no_tid.npz --out_dir runs/emb_exp1 --epochs 20 --batch_size 256 --use_bce true --fusion_mode fixed --best_metric nll

"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from timemixerpp.metric_encoder import TemporalConvEmbedder, MultiScaleEmbedder
from timemixerpp.losses import MultiScaleSupConLoss
from timemixerpp.data import (
    NPZMultiScaleDataset, create_splits, load_splits,
    create_multiscale_dataloaders
)
from timemixerpp.utils import set_seed, setup_logging, AverageMeter

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train sequence-level embedding encoder with SupCon',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to NPZ file with multi-scale features')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--splits_path', type=str, default=None,
                        help='Path to existing splits.json (optional)')
    parser.add_argument('--split_ratio', type=str, default='0.7,0.15,0.15',
                        help='Train/val/test split ratio (comma-separated)')
    
    # Model arguments
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for conv layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Loss arguments
    parser.add_argument('--tau', type=float, default=0.07,
                        help='Temperature for SupCon loss')
    parser.add_argument('--use_bce', type=str, default='false',
                        help='Whether to use BCE loss (true/false)')
    parser.add_argument('--lambda_bce', type=float, default=0.5,
                        help='Weight for BCE loss')
    parser.add_argument('--scale_weights', type=str, default='0.5,0.3,0.2',
                        help='Weights for each scale in loss (comma-separated)')
    
    # Fusion arguments (for evaluation)
    parser.add_argument('--fusion_mode', type=str, default='fixed',
                        choices=['fixed', 'learned'],
                        help='Fusion mode for probability aggregation: fixed weights or learned from model')
    parser.add_argument('--w0', type=float, default=0.5,
                        help='Weight for scale 0 probability (fixed mode)')
    parser.add_argument('--w1', type=float, default=0.3,
                        help='Weight for scale 1 probability (fixed mode)')
    parser.add_argument('--w2', type=float, default=0.2,
                        help='Weight for scale 2 probability (fixed mode)')
    
    # Checkpoint selection criterion
    parser.add_argument('--best_metric', type=str, default='nll',
                        choices=['nll', 'brier'],
                        help='Metric for selecting best checkpoint (lower is better)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--balanced_sampling', type=str, default='false',
                        help='Whether to use balanced sampling (true/false)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers')
    
    return parser.parse_args()


def str_to_bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ('true', '1', 'yes', 'on')


def get_fusion_weights(
    model: nn.Module,
    fusion_mode: str,
    fixed_weights: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Get fusion weights based on mode.
    
    Args:
        model: MultiScaleEmbedder model
        fusion_mode: 'fixed' or 'learned'
        fixed_weights: (w0, w1, w2) for fixed mode
        
    Returns:
        Normalized (w0, w1, w2)
    """
    if fusion_mode == 'learned' and hasattr(model, 'fusion_logits'):
        with torch.no_grad():
            weights = F.softmax(model.fusion_logits, dim=0).cpu().numpy().tolist()
        return tuple(weights)
    else:
        # Normalize fixed weights
        total = sum(fixed_weights)
        return tuple(w / total for w in fixed_weights)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    fusion_mode: str = 'fixed',
    fixed_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on a dataset using fused probability.
    
    三尺度融合评估：
    - p0 = sigmoid(logits0), p1 = sigmoid(logits1), p2 = sigmoid(logits2)
    - p = w0*p0 + w1*p1 + w2*p2
    
    Metrics:
    - nll: Binary Cross Entropy (lower is better)
    - brier: Brier Score = mean((p - labels)^2) (lower is better)
    - mae: Mean Absolute Error = mean(|p - labels|) (lower is better)
    - accuracy, f1, auroc: Reference metrics based on threshold
    
    Returns metrics including p_mean for debugging.
    """
    model.eval()
    
    # Get fusion weights
    w0, w1, w2 = get_fusion_weights(model, fusion_mode, fixed_weights)
    
    all_probs = []  # Fused probabilities
    all_p0, all_p1, all_p2 = [], [], []  # Per-scale probabilities
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            x0, x1, x2, labels, _ = batch
            x0, x1, x2 = x0.to(device), x1.to(device), x2.to(device)
            
            out = model(x0, x1, x2)
            
            # Check that logits are available (requires --use_bce true)
            if 'logits0' not in out or 'logits1' not in out or 'logits2' not in out:
                raise ValueError(
                    "Model must output logits0/logits1/logits2 for probability evaluation. "
                    "Please train with --use_bce true"
                )
            
            # Per-scale probabilities: sigmoid(logits)
            p0 = torch.sigmoid(out['logits0']).squeeze(-1)  # (B,)
            p1 = torch.sigmoid(out['logits1']).squeeze(-1)  # (B,)
            p2 = torch.sigmoid(out['logits2']).squeeze(-1)  # (B,)
            
            # Fused probability: p = w0*p0 + w1*p1 + w2*p2
            p_fused = w0 * p0 + w1 * p1 + w2 * p2  # (B,)
            
            all_probs.append(p_fused.cpu().numpy())
            all_p0.append(p0.cpu().numpy())
            all_p1.append(p1.cpu().numpy())
            all_p2.append(p2.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_p0 = np.concatenate(all_p0)
    all_p1 = np.concatenate(all_p1)
    all_p2 = np.concatenate(all_p2)
    all_labels = np.concatenate(all_labels)
    
    # Clamp probabilities for numerical stability
    eps = 1e-7
    all_probs_clamped = np.clip(all_probs, eps, 1 - eps)
    
    # === Probability Metrics ===
    # NLL: -mean(labels * log(p) + (1-labels) * log(1-p))
    nll = -np.mean(
        all_labels * np.log(all_probs_clamped) + 
        (1 - all_labels) * np.log(1 - all_probs_clamped)
    )
    
    # Brier Score: mean((p - labels)^2)
    brier = np.mean((all_probs - all_labels) ** 2)
    
    # MAE: mean(|p - labels|)
    mae = np.mean(np.abs(all_probs - all_labels))
    
    # === Reference Binary Metrics (threshold-based) ===
    preds = (all_probs >= threshold).astype(int)
    binary_labels = (all_labels >= threshold).astype(int)
    
    accuracy = accuracy_score(binary_labels, preds)
    f1 = f1_score(binary_labels, preds, zero_division=0)
    
    # AUROC needs at least 2 classes
    if len(np.unique(binary_labels)) > 1:
        auroc = roc_auc_score(binary_labels, all_probs)
    else:
        auroc = 0.0
    
    metrics = {
        # Probability metrics (primary)
        'nll': float(nll),
        'brier': float(brier),
        'mae': float(mae),
        # Binary metrics (reference)
        'accuracy': float(accuracy),
        'f1': float(f1),
        'auroc': float(auroc),
        # Debugging info
        'p_mean': float(np.mean(all_probs)),
        'p_std': float(np.std(all_probs)),
        'p0_mean': float(np.mean(all_p0)),
        'p1_mean': float(np.mean(all_p1)),
        'p2_mean': float(np.mean(all_p2)),
        # Fusion weights used
        'w0': w0,
        'w1': w1,
        'w2': w2,
    }
    
    return metrics


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    supcon_meter = AverageMeter()
    bce_meter = AverageMeter()
    
    for batch_idx, batch in enumerate(loader):
        x0, x1, x2, labels, _ = batch
        x0, x1, x2, labels = x0.to(device), x1.to(device), x2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        out = model(x0, x1, x2)
        
        # Compute loss
        loss_dict = criterion(
            e0=out['e0'],
            e1=out['e1'],
            e2=out['e2'],
            labels=labels,
            logits0=out.get('logits0'),
            logits1=out.get('logits1'),
            logits2=out.get('logits2')
        )
        
        loss = loss_dict['total']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update meters
        batch_size = x0.size(0)
        loss_meter.update(loss.item(), batch_size)
        supcon_meter.update(loss_dict['supcon'].item(), batch_size)
        if 'bce' in loss_dict:
            bce_meter.update(loss_dict['bce'].item(), batch_size)
    
    return {
        'loss': loss_meter.avg,
        'supcon': supcon_meter.avg,
        'bce': bce_meter.avg if bce_meter.count > 0 else 0.0
    }


def format_prob(p: float, decimals: int = 1) -> str:
    """Format probability with specified decimal places for display."""
    return f"{round(p, decimals):.{decimals}f}"


def main():
    args = parse_args()
    
    # Parse boolean arguments
    use_bce = str_to_bool(args.use_bce)
    balanced_sampling = str_to_bool(args.balanced_sampling)
    
    # Parse list arguments
    split_ratio = tuple(float(x) for x in args.split_ratio.split(','))
    scale_weights = tuple(float(x) for x in args.scale_weights.split(','))
    fixed_weights = (args.w0, args.w1, args.w2)
    
    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, 'train.log'))
    
    logger.info("=" * 60)
    logger.info("Embedding Encoder Training (SupCon + Probability Prediction)")
    logger.info("=" * 60)
    
    # Validate: use_bce must be true for probability evaluation
    if not use_bce:
        logger.warning(
            "WARNING: --use_bce is false. Model will not have classification heads. "
            "Probability-based evaluation requires --use_bce true. "
            "Will use SupCon-only training."
        )
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Data
    logger.info(f"Loading data from: {args.npz_path}")
    
    # Handle splits path
    splits_path = args.splits_path
    if splits_path is None:
        splits_path = os.path.join(args.out_dir, 'splits.json')
    
    train_loader, val_loader, test_loader, splits = create_multiscale_dataloaders(
        npz_path=args.npz_path,
        batch_size=args.batch_size,
        splits_path=splits_path,
        split_ratio=split_ratio,
        balanced_sampling=balanced_sampling,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    logger.info("Creating model...")
    model = MultiScaleEmbedder(
        input_dim=64,  # Feature dimension from TimeMixer++
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        use_classification_head=use_bce
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Loss
    criterion = MultiScaleSupConLoss(
        temperature=args.tau,
        scale_weights=scale_weights,
        use_bce=use_bce,
        lambda_bce=args.lambda_bce
    )
    
    logger.info(f"Loss: SupCon (tau={args.tau}) + BCE (use={use_bce}, lambda={args.lambda_bce})")
    logger.info(f"Scale weights (loss): {scale_weights}")
    logger.info(f"Fusion mode (eval): {args.fusion_mode}")
    logger.info(f"Fusion weights (eval): w0={args.w0}, w1={args.w1}, w2={args.w2}")
    logger.info(f"Best checkpoint metric: {args.best_metric} (lower is better)")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Training loop
    best_metric_value = float('inf')  # Lower is better for nll/brier
    best_epoch = 0
    history = []
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        if use_bce:
            val_metrics = evaluate(
                model, val_loader, device,
                fusion_mode=args.fusion_mode,
                fixed_weights=fixed_weights
            )
        else:
            # Fallback for SupCon-only: no probability metrics
            val_metrics = {
                'nll': float('inf'),
                'brier': float('inf'),
                'mae': float('inf'),
                'accuracy': 0.0,
                'f1': 0.0,
                'auroc': 0.0,
                'p_mean': 0.0,
                'p_std': 0.0,
                'w0': args.w0, 'w1': args.w1, 'w2': args.w2,
            }
        
        # Update scheduler
        scheduler.step()
        
        # Log with formatted probabilities
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {train_metrics['loss']:.4f} (SupCon: {train_metrics['supcon']:.4f}, BCE: {train_metrics['bce']:.4f}) - "
            f"Val NLL: {val_metrics['nll']:.4f}, Brier: {val_metrics['brier']:.4f}, MAE: {val_metrics['mae']:.4f} - "
            f"p̄={format_prob(val_metrics['p_mean'])} - "
            f"(Ref: Acc={val_metrics['accuracy']:.3f}, F1={val_metrics['f1']:.3f}, AUROC={val_metrics['auroc']:.3f})"
        )
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_supcon': train_metrics['supcon'],
            'train_bce': train_metrics['bce'],
            'val_nll': val_metrics['nll'],
            'val_brier': val_metrics['brier'],
            'val_mae': val_metrics['mae'],
            'val_p_mean': val_metrics['p_mean'],
            'val_accuracy': val_metrics.get('accuracy', 0),
            'val_f1': val_metrics.get('f1', 0),
            'val_auroc': val_metrics.get('auroc', 0)
        })
        
        # Save best model (lower is better for nll/brier)
        current_metric = val_metrics[args.best_metric]
        if current_metric < best_metric_value:
            best_metric_value = current_metric
            best_epoch = epoch + 1
            
            # Get current fusion weights (for learned mode)
            current_weights = get_fusion_weights(model, args.fusion_mode, fixed_weights)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': {
                    'input_dim': 64,
                    'hidden_dim': args.hidden_dim,
                    'emb_dim': args.emb_dim,
                    'dropout': args.dropout,
                    'use_classification_head': use_bce,
                    'tau': args.tau,
                    'scale_weights': scale_weights,
                    'use_bce': use_bce,
                    'lambda_bce': args.lambda_bce,
                    'fusion_mode': args.fusion_mode,
                    'fusion_weights': current_weights,
                },
                'fusion_logits': model.fusion_logits.detach().cpu().numpy().tolist(),
            }
            
            torch.save(checkpoint, os.path.join(args.out_dir, 'checkpoint.pt'))
            logger.info(f"  -> New best model! {args.best_metric}: {best_metric_value:.4f}")
    
    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on test set...")
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.out_dir, 'checkpoint.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if use_bce:
        test_metrics = evaluate(
            model, test_loader, device,
            fusion_mode=args.fusion_mode,
            fixed_weights=fixed_weights
        )
        
        logger.info(f"Test Results:")
        logger.info(f"  NLL: {test_metrics['nll']:.4f}")
        logger.info(f"  Brier: {test_metrics['brier']:.4f}")
        logger.info(f"  MAE: {test_metrics['mae']:.4f}")
        logger.info(f"  Mean Prob: {format_prob(test_metrics['p_mean'])} (raw: {test_metrics['p_mean']:.4f})")
        logger.info(f"  Per-scale means: p0={format_prob(test_metrics['p0_mean'])}, p1={format_prob(test_metrics['p1_mean'])}, p2={format_prob(test_metrics['p2_mean'])}")
        logger.info(f"  Fusion weights: w0={test_metrics['w0']:.3f}, w1={test_metrics['w1']:.3f}, w2={test_metrics['w2']:.3f}")
        logger.info(f"  Reference metrics: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUROC={test_metrics['auroc']:.4f}")
    else:
        test_metrics = {'nll': float('inf'), 'brier': float('inf'), 'mae': float('inf')}
        logger.info("Test evaluation skipped (SupCon-only mode)")
    
    # Save metrics
    metrics_output = {
        'best_epoch': best_epoch,
        f'best_val_{args.best_metric}': best_metric_value,
        'test_nll': test_metrics.get('nll', float('inf')),
        'test_brier': test_metrics.get('brier', float('inf')),
        'test_mae': test_metrics.get('mae', float('inf')),
        'test_accuracy': test_metrics.get('accuracy', 0),
        'test_f1': test_metrics.get('f1', 0),
        'test_auroc': test_metrics.get('auroc', 0),
        'test_p_mean': test_metrics.get('p_mean', 0),
        'history': history,
        'config': {
            'npz_path': args.npz_path,
            'emb_dim': args.emb_dim,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'tau': args.tau,
            'use_bce': use_bce,
            'lambda_bce': args.lambda_bce,
            'scale_weights': scale_weights,
            'fusion_mode': args.fusion_mode,
            'fusion_weights': fixed_weights,
            'best_metric': args.best_metric,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'balanced_sampling': balanced_sampling,
            'seed': args.seed,
        }
    }
    
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    logger.info(f"\nMetrics saved to: {os.path.join(args.out_dir, 'metrics.json')}")
    logger.info(f"Checkpoint saved to: {os.path.join(args.out_dir, 'checkpoint.pt')}")
    logger.info(f"Splits saved to: {splits_path}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best model at epoch {best_epoch} with val {args.best_metric}: {best_metric_value:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
