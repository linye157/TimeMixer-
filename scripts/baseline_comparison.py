#!/usr/bin/env python
"""
Baseline model comparison for time series binary classification.

Compares TimeMixer++ with baseline models:
- LSTM
- BiLSTM
- LSTM-Transformer
- CNN-BiLSTM
- Transformer
- MLP

Usage:
    # 使用默认划分（从训练集分出30%作为测试集）
    python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --epochs 50
    
    # 指定独立测试集
    python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --test_path TDdata/TestData.csv --epochs 50
    
    # 使用全部训练数据作为测试集（test_split=0）
    python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --test_split 0 --epochs 50
    
    # 只对比特定模型
    python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --models lstm bilstm transformer
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import logging
import os
import json
from typing import Dict, Type, Any, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.data import load_file_strict, create_dataloaders
from timemixerpp.utils import set_seed, compute_metrics, setup_logging, AverageMeter

logger = logging.getLogger(__name__)


# ============================================
# Baseline Models
# ============================================

class LSTMClassifier(nn.Module):
    """LSTM-based classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        logits = self.fc(out)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, (h_n, _) = self.lstm(x)
        # Concatenate forward and backward hidden states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_concat = torch.cat([h_forward, h_backward], dim=-1)
        out = self.dropout(h_concat)
        logits = self.fc(out)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class LSTMTransformerClassifier(nn.Module):
    """LSTM + Transformer hybrid classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, 
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        logits = self.fc(x)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class CNNBiLSTMClassifier(nn.Module):
    """CNN + BiLSTM hybrid classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # CNN: (B, L, 1) -> (B, 1, L) -> (B, hidden, L) -> (B, hidden, L//2)
        x = x.transpose(1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        # LSTM: (B, hidden, L//2) -> (B, L//2, hidden)
        x = x.transpose(1, 2)
        out, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_concat = torch.cat([h_forward, h_backward], dim=-1)
        out = self.dropout(h_concat)
        logits = self.fc(out)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class TransformerClassifier(nn.Module):
    """Pure Transformer classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.embed(x) + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class MLPClassifier(nn.Module):
    """Simple MLP classifier (flattened input)."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = seq_len
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        logits = self.mlp(x)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


class GRUClassifier(nn.Module):
    """GRU-based classifier."""
    
    def __init__(self, seq_len: int = 48, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        logits = self.fc(out)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}


# ============================================
# Model Registry (Extensible)
# ============================================

@dataclass
class ModelConfig:
    """Configuration for a baseline model."""
    model_class: Type[nn.Module]
    default_kwargs: Dict[str, Any]
    description: str


# Registry of available models - can be extended by adding new entries
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    'lstm': ModelConfig(
        model_class=LSTMClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        description='LSTM classifier'
    ),
    'bilstm': ModelConfig(
        model_class=BiLSTMClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        description='Bidirectional LSTM classifier'
    ),
    'lstm_transformer': ModelConfig(
        model_class=LSTMTransformerClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'n_heads': 4, 'dropout': 0.1},
        description='LSTM + Transformer hybrid'
    ),
    'cnn_bilstm': ModelConfig(
        model_class=CNNBiLSTMClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        description='CNN + BiLSTM hybrid'
    ),
    'transformer': ModelConfig(
        model_class=TransformerClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'n_heads': 4, 'dropout': 0.1},
        description='Pure Transformer classifier'
    ),
    'mlp': ModelConfig(
        model_class=MLPClassifier,
        default_kwargs={'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.1},
        description='Multi-layer Perceptron'
    ),
    'gru': ModelConfig(
        model_class=GRUClassifier,
        default_kwargs={'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        description='GRU classifier'
    ),
}


def register_model(name: str, model_class: Type[nn.Module], default_kwargs: Dict, description: str):
    """
    Register a new model to the registry.
    
    Example:
        register_model('my_model', MyModelClass, {'hidden_dim': 64}, 'My custom model')
    """
    MODEL_REGISTRY[name] = ModelConfig(model_class, default_kwargs, description)


def list_models() -> List[str]:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


def create_model(name: str, seq_len: int = 48, **kwargs) -> nn.Module:
    """Create a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list_models()}")
    
    config = MODEL_REGISTRY[name]
    model_kwargs = {**config.default_kwargs, 'seq_len': seq_len, **kwargs}
    return config.model_class(**model_kwargs)


# ============================================
# Training and Evaluation
# ============================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    model_name: str
) -> nn.Module:
    """Train a model and return the trained model."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = AverageMeter()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output['logits'], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss.update(loss.item(), batch_x.size(0))
        
        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                all_preds.append(output['probs'].cpu().numpy())
                all_labels.append(batch_y.numpy())
        
        all_preds = np.concatenate(all_preds).squeeze()
        all_labels = np.concatenate(all_labels).squeeze()
        
        metrics = compute_metrics(all_labels, all_preds)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  [{model_name}] Epoch {epoch+1}/{epochs}: Loss={train_loss.avg:.4f}, F1={metrics['f1']:.4f}")
    
    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate a model on test set and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            all_preds.append(output['probs'].cpu().numpy())
            all_labels.append(batch_y.numpy())
    
    all_preds = np.concatenate(all_preds).squeeze()
    all_labels = np.concatenate(all_labels).squeeze()
    
    return compute_metrics(all_labels, all_preds)


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline model comparison')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data file')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to test data file (optional). If not provided, split from training data')
    parser.add_argument('--test_split', type=float, default=0.3,
                        help='Test split ratio if test_path not provided (default: 0.3). '
                             'If 0, use all training data as test set')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'Models to compare. Available: {list_models()}. Default: all')
    parser.add_argument('--include_timemixer', action='store_true',
                        help='Include TimeMixer++ in comparison')
    
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for all models')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split from training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--output', type=str, default='results/baseline_comparison.json',
                        help='Path to save results')
    
    return parser.parse_args()


def prepare_data(args, device):
    """
    Prepare training, validation, and test data loaders.
    
    Returns:
        train_loader, val_loader, test_loader, normalizer_stats
    """
    from torch.utils.data import TensorDataset
    
    # Load training data
    logger.info(f"Loading training data: {args.data_path}")
    _, X_train, y_train = load_file_strict(args.data_path)
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Handle test data
    if args.test_path:
        # Use provided test set
        logger.info(f"Loading test data: {args.test_path}")
        _, X_test, y_test = load_file_strict(args.test_path)
        logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    elif args.test_split == 0:
        # Use all training data as test set
        logger.info("Using all training data as test set (test_split=0)")
        X_test, y_test = X_train.copy(), y_train.copy()
    else:
        # Split from training data
        logger.info(f"Splitting {args.test_split*100:.0f}% of training data as test set")
        n_samples = len(X_train)
        n_test = int(n_samples * args.test_split)
        
        np.random.seed(args.seed)
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_test, y_test = X_train[test_indices], y_train[test_indices]
        X_train, y_train = X_train[train_indices], y_train[train_indices]
        logger.info(f"After split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Normalize training data
    mean = X_train.mean()
    std = X_train.std() + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Create train/val split
    n_train = len(X_train_norm)
    n_val = int(n_train * args.val_split)
    
    np.random.seed(args.seed + 1)
    indices = np.random.permutation(n_train)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train_final = X_train_norm[train_indices]
    y_train_final = y_train[train_indices]
    X_val = X_train_norm[val_indices]
    y_val = y_train[val_indices]
    
    logger.info(f"Final split - Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_final, dtype=torch.float32),
        torch.tensor(y_train_final, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, (mean, std)


def main():
    args = parse_args()
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("Baseline Model Comparison")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader, normalizer_stats = prepare_data(args, device)
    
    # Select models
    model_names = args.models if args.models else list_models()
    logger.info(f"Models to compare: {model_names}")
    
    # Results
    results = {}
    
    # Train and evaluate each baseline model
    for name in model_names:
        if name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {name}, skipping")
            continue
        
        logger.info(f"\nTraining {name}...")
        set_seed(args.seed)  # Reset seed for fair comparison
        
        model = create_model(name, seq_len=48, hidden_dim=args.hidden_dim)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {n_params:,}")
        
        # Train model
        model = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=device, model_name=name
        )
        
        # Evaluate on test set
        logger.info(f"  Evaluating on test set...")
        test_metrics = evaluate_model(model, test_loader, device)
        
        results[name] = {
            'params': n_params,
            'description': MODEL_REGISTRY[name].description,
            **test_metrics
        }
        logger.info(f"  Test F1: {test_metrics['f1']:.4f}, FPR: {test_metrics['fpr']:.4f}, FNR: {test_metrics['fnr']:.4f}")
    
    # Include TimeMixer++ if requested
    if args.include_timemixer:
        logger.info("\nTraining TimeMixer++...")
        set_seed(args.seed)
        
        config = TimeMixerPPConfig(
            seq_len=48, c_in=1, d_model=args.hidden_dim,
            n_layers=2, n_heads=4, top_k=3, dropout=0.1
        )
        model = TimeMixerPPForBinaryCls(config)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {n_params:,}")
        
        # Train model
        model = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=device, model_name='TimeMixer++'
        )
        
        # Evaluate on test set
        logger.info(f"  Evaluating on test set...")
        test_metrics = evaluate_model(model, test_loader, device)
        
        results['timemixer++'] = {
            'params': n_params,
            'description': 'TimeMixer++ (ours)',
            **test_metrics
        }
        logger.info(f"  Test F1: {test_metrics['f1']:.4f}, FPR: {test_metrics['fpr']:.4f}, FNR: {test_metrics['fnr']:.4f}")
    
    # Print results
    print("\n" + "=" * 90)
    print(" Test Set Results")
    print("=" * 90)
    print(f"{'Model':<20} {'Params':>10} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUROC':>8} {'FPR':>8} {'FNR':>8}")
    print("-" * 90)
    
    for name, res in sorted(results.items(), key=lambda x: -x[1].get('f1', 0)):
        print(f"{name:<20} {res['params']:>10,} {res['accuracy']:>8.4f} {res['precision']:>8.4f} "
              f"{res['recall']:>8.4f} {res['f1']:>8.4f} {res['auroc']:>8.4f} {res['fpr']:>8.4f} {res['fnr']:>8.4f}")
    
    print("=" * 90)
    
    # Print metric explanation
    print("\n指标说明:")
    print("  FPR (误报率) = FP / (FP + TN) - 实际为负类但被预测为正类的比例")
    print("  FNR (漏报率) = FN / (TP + FN) - 实际为正类但被预测为负类的比例 (= 1 - Recall)")
    
    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

