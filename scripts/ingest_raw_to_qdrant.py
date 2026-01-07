#!/usr/bin/env python
"""
将原始时间序列数据直接入库到 Qdrant（单 Collection）。

每条数据包含：
- 48维温度向量作为 embedding
- label 值存入 payload

适用于简单的相似样本检索，无需训练 embedding encoder。
注意：此脚本创建单个 Collection，用于 query_raw_qdrant.py 查询。
如需三尺度知识库，请使用 ingest_to_qdrant_3scales.py。

Usage:
    REM 入库 CSV 文件
    python scripts/ingest_raw_to_qdrant.py --data_path TDdata/TrainData.csv --qdrant_url http://localhost:6333 --collection_name raw_temperature_kb --l2_normalize --recreate

    REM 入库 Excel 文件
    python scripts/ingest_raw_to_qdrant.py --data_path TDdata/alldata.xlsx --qdrant_url http://localhost:6333 --collection_name raw_temperature_kb --l2_normalize

    REM 指定 ID 偏移（用于追加数据）
    python scripts/ingest_raw_to_qdrant.py --data_path TDdata/TestData.csv --qdrant_url http://localhost:6333 --collection_name raw_temperature_kb --id_offset 1000
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import logging
from typing import Dict, List, Any, Optional

import numpy as np

from timemixerpp.data import load_file_strict
from timemixerpp.qdrant_utils import (
    get_client, create_or_validate_collection, upsert_points, get_collection_info
)
from timemixerpp.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Ingest raw time series data to Qdrant',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file (.xlsx or .csv)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize vectors (z-score) before ingestion')
    parser.add_argument('--l2_normalize', action='store_true',
                        help='L2 normalize vectors (unit length) for cosine similarity')
    
    # Qdrant arguments
    parser.add_argument('--qdrant_url', type=str, default='http://localhost:6333',
                        help='Qdrant server URL')
    parser.add_argument('--collection_name', type=str, default='raw_temperature_kb',
                        help='Collection name')
    parser.add_argument('--recreate', action='store_true',
                        help='Recreate collection if exists')
    parser.add_argument('--distance', type=str, default='Cosine',
                        choices=['Cosine', 'Euclid', 'Dot'],
                        help='Distance metric for similarity search')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for upserting')
    parser.add_argument('--id_offset', type=int, default=0,
                        help='Offset to add to point IDs (for appending data)')
    
    return parser.parse_args()


def normalize_zscore(X: np.ndarray) -> np.ndarray:
    """Z-score normalization (per feature)."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0  # Avoid division by zero
    return (X - mean) / std


def normalize_l2(X: np.ndarray) -> np.ndarray:
    """L2 normalization (unit length vectors)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0  # Avoid division by zero
    return X / norms


def create_payloads(
    n_samples: int,
    labels: np.ndarray,
    data_path: str,
    id_offset: int
) -> List[Dict[str, Any]]:
    """
    Create payload dictionaries for Qdrant points.
    
    Payload includes:
    - label: binary label (0/1)
    - label_raw: original label value (float)
    - sample_id: original sample index
    - source_file: source data file name
    """
    payloads = []
    source_file = Path(data_path).name
    
    for i in range(n_samples):
        payload = {
            'label': int(labels[i] >= 0.5),  # Binary
            'label_raw': float(labels[i]),   # Original float value
            'sample_id': i + id_offset,
            'source_file': source_file,
        }
        payloads.append(payload)
    
    return payloads


def main():
    args = parse_args()
    
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("Raw Data Qdrant Ingestion")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading data from: {args.data_path}")
    _, X, y = load_file_strict(args.data_path)
    
    n_samples, vector_dim = X.shape
    logger.info(f"Loaded {n_samples} samples, vector dimension: {vector_dim}")
    logger.info(f"Label distribution: positive={np.mean(y >= 0.5):.2%}, negative={np.mean(y < 0.5):.2%}")
    
    # Normalize if requested
    if args.normalize:
        logger.info("Applying z-score normalization...")
        X = normalize_zscore(X)
    
    if args.l2_normalize:
        logger.info("Applying L2 normalization...")
        X = normalize_l2(X)
    
    # Convert to float32
    X = X.astype(np.float32)
    
    # Connect to Qdrant
    logger.info(f"Connecting to Qdrant: {args.qdrant_url}")
    client = get_client(args.qdrant_url)
    
    # Create collection
    logger.info(f"Creating/validating collection: {args.collection_name}")
    create_or_validate_collection(
        client,
        args.collection_name,
        vector_size=vector_dim,
        distance=args.distance,
        recreate=args.recreate
    )
    
    # Prepare data for upserting
    ids = list(range(args.id_offset, args.id_offset + n_samples))
    vectors = X.tolist()
    payloads = create_payloads(n_samples, y, args.data_path, args.id_offset)
    
    # Upsert
    logger.info(f"Upserting {n_samples} points...")
    n_upserted = upsert_points(
        client,
        args.collection_name,
        ids,
        vectors,
        payloads,
        batch_size=args.batch_size
    )
    
    logger.info(f"Upserted {n_upserted} points to '{args.collection_name}'")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Complete!")
    logger.info("=" * 60)
    
    info = get_collection_info(client, args.collection_name)
    logger.info(f"Collection: {args.collection_name}")
    logger.info(f"  Total points: {info['points_count']}")
    
    logger.info(f"Vector dimension: {vector_dim}")
    logger.info(f"Distance metric: {args.distance}")
    logger.info(f"Source file: {Path(args.data_path).name}")
    
    if args.normalize:
        logger.info("Normalization: z-score")
    elif args.l2_normalize:
        logger.info("Normalization: L2")
    else:
        logger.info("Normalization: none")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

