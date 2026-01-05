"""
Utility functions for TimeMixer++ training and evaluation.

Includes:
- Seed setting for reproducibility
- Metric computation
- Checkpoint saving/loading
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    label_threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Both predictions and labels are thresholded to binary values for computing
    classification metrics (Accuracy, Precision, Recall, F1).
    
    Args:
        y_true: Ground truth labels, shape (n,). Can be 0-1 floats (probabilities)
        y_pred: Predicted probabilities, shape (n,). Values in [0, 1]
        y_prob: (Optional) Probabilities for AUROC, defaults to y_pred
        threshold: Threshold for converting predictions to binary (default: 0.5)
        label_threshold: Threshold for converting labels to binary.
                        If None, uses the same value as threshold.
                        
    Returns:
        Dictionary of metrics
        
    Note:
        - Model output remains as probabilities (0-1 floats)
        - For metric computation, both y_true and y_pred are thresholded:
          - y_true >= label_threshold -> 1 (positive/accident)
          - y_pred >= threshold -> 1 (predicted positive)
        - AUROC uses raw probability values (no thresholding)
    """
    if y_prob is None:
        y_prob = y_pred
    
    if label_threshold is None:
        label_threshold = threshold
    
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred >= threshold).astype(float)
    
    # Convert labels to binary using label_threshold
    # This handles cases where labels are 0-1 floats (probabilities)
    y_true_binary = (y_true >= label_threshold).astype(float)
    
    # Accuracy
    accuracy = (y_pred_binary == y_true_binary).mean()
    
    # Confusion matrix components
    tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
    tn = ((y_pred_binary == 0) & (y_true_binary == 0)).sum()
    fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
    fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 误报率 (False Positive Rate) = FP / (FP + TN)
    # 即：实际为负类但被预测为正类的比例
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # 漏报率 (False Negative Rate / Miss Rate) = FN / (TP + FN) = 1 - Recall
    # 即：实际为正类但被预测为负类的比例
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fpr': float(fpr),  # 误报率
        'fnr': float(fnr),  # 漏报率
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    # AUROC uses raw probability values (no thresholding)
    # Note: For proper AUROC, y_true should ideally be binary
    # If y_true is continuous, we threshold it for AUROC computation too
    try:
        auroc = compute_auroc(y_true_binary, y_prob)
        metrics['auroc'] = float(auroc)
    except Exception as e:
        logger.warning(f"Could not compute AUROC: {e}")
        metrics['auroc'] = 0.0
    
    return metrics


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve.
    
    Simple implementation using the Wilcoxon-Mann-Whitney statistic.
    
    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores/probabilities
        
    Returns:
        AUROC value
    """
    # Get positive and negative samples
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5  # Random baseline
    
    # Count pairs where positive > negative
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    
    # Use vectorized comparison
    comparisons = (pos_scores[:, None] > neg_scores[None, :]).sum()
    ties = (pos_scores[:, None] == neg_scores[None, :]).sum()
    
    auroc = (comparisons + 0.5 * ties) / (n_pos * n_neg)
    
    return auroc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    config: Optional[Any] = None,
    normalizer_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Metrics dictionary
        path: Save path
        config: Optional config to save
        normalizer_stats: Optional (mean, std) tuple for normalization
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    if normalizer_stats is not None:
        checkpoint['normalizer_mean'] = normalizer_stats[0]
        checkpoint['normalizer_std'] = normalizer_stats[1]
    
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
    init_dynamic_layers: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load to
        init_dynamic_layers: If True, initialize dynamic layers before loading
                            (required for models with lazy initialization)
        
    Returns:
        Checkpoint dictionary (epoch, metrics, etc.)
    
    Note:
        Some layers (LayerNorms in MixerBlock, convs in MCM, ensemble weights)
        are lazily initialized on first forward pass. This function handles
        that by optionally doing a dummy forward pass before loading state dict.
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Initialize dynamic layers if needed
    if init_dynamic_layers:
        # Get config from checkpoint to determine seq_len
        config = checkpoint.get('config', {})
        seq_len = config.get('seq_len', 48)
        
        # Do dummy forward pass to initialize dynamic layers
        with torch.no_grad():
            dummy_input = torch.randn(1, seq_len, device=device)
            try:
                _ = model(dummy_input)
            except Exception as e:
                logger.warning(f"Dummy forward pass failed: {e}. Trying to load anyway.")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model weights loaded from {path}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state loaded")
    
    return checkpoint


class EarlyStopping:
    """
    Early stopping utility.
    
    Stops training when monitored metric stops improving.
    
    Args:
        patience: Number of epochs to wait
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max'
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional file to log to
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

