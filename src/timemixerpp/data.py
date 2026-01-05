"""
Data utilities for TimeMixer++ training.

Includes:
- Dataset class for temperature/accident data
- Data normalization utilities
- Data loading functions
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def load_file_strict(file_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load data file with strict format handling.
    
    Args:
        file_path: Path to data file (.xlsx or .csv)
        
    Returns:
        data: Original DataFrame
        X: Feature array, shape (n, 48)
        y: Label array, shape (n,)
        
    File format specifications:
        .xlsx: sheet_name=2, header=0, X=iloc[:, 3:51], y=iloc[:, 51]
        .csv: header=None, X=iloc[:, 0:48], y=iloc[:, 48]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        logger.info(f"Loading Excel file (Sheet3): {file_path}")
        # Read third sheet (index 2)
        data = pd.read_excel(file_path, sheet_name=2, header=0)
        # Extract columns 4-51 (index 3:51) as features, column 52 (index 51) as label
        X = data.iloc[:, 3:51].values
        y = data.iloc[:, 51].values
        
    elif file_path.endswith('.csv'):
        logger.info(f"Loading CSV file (no header): {file_path}")
        # Read CSV without header
        data = pd.read_csv(file_path, header=None)
        # Extract first 48 columns as features, column 49 (index 48) as label
        X = data.iloc[:, 0:48].values
        y = data.iloc[:, 48].values
        
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Ensure correct data types
    X = pd.to_numeric(X.flatten(), errors='coerce').reshape(X.shape).astype(float)
    y = pd.to_numeric(y, errors='coerce').astype(float)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    logger.info(f"Loaded {len(X)} samples, X shape: {X.shape}, y shape: {y.shape}")
    
    return data, X, y


class TemperatureDataset(Dataset):
    """
    Dataset for temperature time series with binary labels.
    
    Args:
        X: Feature array, shape (n, 48) or (n, 48, 1)
        y: Label array, shape (n,)
        normalize: Whether to normalize features
        mean: Pre-computed mean for normalization
        std: Pre-computed std for normalization
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ):
        super().__init__()
        
        # Handle input shape
        if X.ndim == 1:
            X = X.reshape(-1, 48)
        
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        
        # Normalization
        self.normalize = normalize
        if normalize:
            if mean is None:
                self.mean = self.X.mean(axis=0, keepdims=True)
            else:
                self.mean = mean
            if std is None:
                self.std = self.X.std(axis=0, keepdims=True)
                self.std[self.std < 1e-6] = 1.0  # Avoid division by zero
            else:
                self.std = std
            
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
    
    def get_stats(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get normalization statistics."""
        return self.mean, self.std


class Normalizer:
    """
    Normalizer for time series data.
    
    Supports z-score normalization with optional saved statistics.
    """
    
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std
        self.fitted = mean is not None and std is not None
    
    def fit(self, X: np.ndarray) -> 'Normalizer':
        """
        Fit normalizer to data.
        
        Args:
            X: Data array, shape (n, seq_len) or (n, seq_len, c_in)
            
        Returns:
            self
        """
        if X.ndim == 3:
            self.mean = X.mean(axis=(0, 1), keepdims=True)
            self.std = X.std(axis=(0, 1), keepdims=True)
        else:
            self.mean = X.mean(axis=0, keepdims=True)
            self.std = X.std(axis=0, keepdims=True)
        
        self.std[self.std < 1e-6] = 1.0
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data.
        
        Args:
            X: Data array
            
        Returns:
            Normalized data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data.
        
        Args:
            X: Normalized data
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return X * self.std + self.mean
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(X).transform(X)
    
    def save(self, path: str):
        """Save normalizer statistics."""
        np.savez(path, mean=self.mean, std=self.std)
    
    @classmethod
    def load(cls, path: str) -> 'Normalizer':
        """Load normalizer from file."""
        data = np.load(path)
        return cls(mean=data['mean'], std=data['std'])


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    normalize: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Create train and validation dataloaders.
    
    Args:
        X: Feature array, shape (n, 48)
        y: Label array, shape (n,)
        batch_size: Batch size
        val_split: Validation split ratio
        normalize: Whether to normalize
        shuffle: Whether to shuffle training data
        num_workers: Number of dataloader workers
        seed: Random seed for split
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        stats: Normalization statistics (mean, std) or None
    """
    # Split data
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    val_size = int(n * val_split)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Create datasets
    train_dataset = TemperatureDataset(X_train, y_train, normalize=normalize)
    mean, std = train_dataset.get_stats()
    val_dataset = TemperatureDataset(X_val, y_val, normalize=normalize, mean=mean, std=std)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    stats = (mean, std) if normalize else None
    
    return train_loader, val_loader, stats

