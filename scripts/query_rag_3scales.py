#!/usr/bin/env python
"""
三尺度 RAG 查询脚本。

从 Qdrant 三个 collection 检索相似样本，
可选择仅检索或计算概率融合预测。

新增参数说明：
  --retrieve_only: 仅检索不预测（默认 true）
  --exclude_self:  过滤掉查询样本本身（默认 true）
  --min_results:   额外请求的结果数，确保过滤后有足够 top_k（默认 10）

Usage:
    # 仅检索（默认行为）
    python scripts/query_rag_3scales.py --npz_path features/alldata_features_no_tid.npz --ckpt_path runs/emb_exp1/checkpoint.pt --qdrant_url http://localhost:6333 --collection_prefix accident_kb_no_tid --query_index 123 --top_k 10


    # 检索 + 概率融合预测
    # 查询（融合预测模式，输出保留1位小数）
python scripts/query_rag_3scales.py --npz_path features/alldata_features_no_tid.npz --ckpt_path runs/emb_exp1/checkpoint.pt --qdrant_url http://localhost:6333 --collection_prefix accident_kb_no_tid --query_index 123 --retrieve_only false --gamma 10

"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

from timemixerpp.metric_encoder import MultiScaleEmbedder
from timemixerpp.qdrant_utils import get_client, search_similar
from timemixerpp.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Query 3-scale RAG system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to NPZ file with multi-scale features')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to trained embedding encoder checkpoint')
    parser.add_argument('--query_index', type=int, required=True,
                        help='Index of the query sample in NPZ')
    
    # Qdrant arguments
    parser.add_argument('--qdrant_url', type=str, default='http://localhost:6333',
                        help='Qdrant server URL')
    parser.add_argument('--collection_prefix', type=str, default='accident_kb',
                        help='Prefix for collection names')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of similar samples to retrieve')
    
    # Retrieval mode arguments
    parser.add_argument('--retrieve_only', type=str, default='true',
                        help='Only retrieve, do not compute probabilities/fusion (true/false)')
    parser.add_argument('--exclude_self', type=str, default='true',
                        help='Exclude the query sample itself from results (true/false)')
    parser.add_argument('--min_results', type=int, default=10,
                        help='Extra results to request to ensure enough after filtering')
    
    # Fusion arguments (only used when retrieve_only=false)
    parser.add_argument('--gamma', type=float, default=10.0,
                        help='Gamma for similarity weighting: w_i = exp(gamma * score_i)')
    parser.add_argument('--fusion_mode', type=str, default='fixed',
                        choices=['fixed', 'learned'],
                        help='Fusion mode: fixed weights or learned from checkpoint')
    parser.add_argument('--w0', type=float, default=0.5,
                        help='Weight for scale 0 (fixed mode)')
    parser.add_argument('--w1', type=float, default=0.3,
                        help='Weight for scale 1 (fixed mode)')
    parser.add_argument('--w2', type=float, default=0.2,
                        help='Weight for scale 2 (fixed mode)')
    
    # Output arguments
    parser.add_argument('--json_output', type=str, default='false',
                        help='Output JSON format (true/false)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save JSON output (optional)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    
    return parser.parse_args()


def str_to_bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ('true', '1', 'yes', 'on')


def filter_results(
    results: List[Dict[str, Any]],
    query_index: int,
    exclude_self: bool,
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Filter search results: optionally exclude self, then truncate to top_k.
    
    Args:
        results: Raw search results
        query_index: Query sample index
        exclude_self: Whether to exclude the query sample itself
        top_k: Number of results to keep after filtering
        
    Returns:
        Filtered and truncated results
    """
    if not exclude_self:
        return results[:top_k]
    
    filtered = []
    for r in results:
        # Get sample_id from payload, fallback to point id
        sample_id = r.get('payload', {}).get('sample_id', r.get('id'))
        if sample_id != query_index:
            filtered.append(r)
    
    # Check if we have enough results
    if len(filtered) < top_k:
        logger.warning(f"After filtering self, only {len(filtered)} results remain (requested {top_k})")
    
    return filtered[:top_k]


def compute_scale_probability(
    search_results: List[Dict[str, Any]],
    gamma: float
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute probability from search results using similarity-weighted voting.
    
    p_m = Σ w_i * label_i / Σ w_i
    where w_i = exp(gamma * score_i)
    """
    if not search_results:
        return 0.5, []
    
    weighted_results = []
    total_weight = 0.0
    weighted_label_sum = 0.0
    
    for result in search_results:
        score = result['score']
        label = result.get('payload', {}).get('label', 0)
        
        weight = np.exp(gamma * score)
        total_weight += weight
        weighted_label_sum += weight * label
        
        weighted_results.append({
            **result,
            'weight': weight
        })
    
    probability = weighted_label_sum / (total_weight + 1e-12)
    
    # Normalize weights for display
    for r in weighted_results:
        r['weight_normalized'] = r['weight'] / (total_weight + 1e-12)
    
    return probability, weighted_results


def format_retrieve_only_display(
    query_info: Dict[str, Any],
    scale_results: List[List[Dict[str, Any]]],
    top_k: int
) -> str:
    """Format retrieval-only results for console display."""
    lines = []
    
    lines.append("=" * 70)
    lines.append(" 三尺度 RAG 检索结果")
    lines.append("=" * 70)
    
    # Query info
    lines.append(f"\n查询样本:")
    lines.append(f"  Index: {query_info['index']}")
    lines.append(f"  真实标签: {query_info['label']:.1f}")
    
    # Scale results
    scale_names = ['Scale 0 (48 时间步)', 'Scale 1 (24 时间步)', 'Scale 2 (12 时间步)']
    
    for scale_idx, results in enumerate(scale_results):
        lines.append(f"\n{'-' * 70}")
        lines.append(f" {scale_names[scale_idx]}")
        lines.append(f"{'-' * 70}")
        lines.append(f"{'Rank':<6}{'ID':<10}{'Label':<8}{'Label_raw':<12}{'Score':<12}")
        lines.append("-" * 48)
        
        for i, r in enumerate(results[:top_k]):
            payload = r.get('payload', {})
            label = payload.get('label', 'N/A')
            label_raw = payload.get('label_raw', label)
            # Format label_raw with 1 decimal
            if isinstance(label_raw, (int, float)):
                label_raw_str = f"{label_raw:.1f}"
            else:
                label_raw_str = str(label_raw)
            
            lines.append(
                f"{i+1:<6}"
                f"{r['id']:<10}"
                f"{label:<8}"
                f"{label_raw_str:<12}"
                f"{r['score']:<12.4f}"
            )
        
        if len(results) > top_k:
            lines.append(f"  ... 还有 {len(results) - top_k} 个结果")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


def format_full_display(
    query_info: Dict[str, Any],
    scale_results: List[Tuple[float, List[Dict]]],
    fusion_result: Dict[str, Any]
) -> str:
    """Format full results (with fusion) for console display."""
    lines = []
    
    lines.append("=" * 70)
    lines.append(" 三尺度 RAG 查询结果")
    lines.append("=" * 70)
    
    # Query info
    lines.append(f"\n查询样本:")
    lines.append(f"  Index: {query_info['index']}")
    lines.append(f"  真实标签: {query_info['label']:.4f} ({'正类' if query_info['label'] >= 0.5 else '负类'})")
    
    # Scale results
    scale_names = ['Scale 0 (48 时间步)', 'Scale 1 (24 时间步)', 'Scale 2 (12 时间步)']
    
    for scale_idx, (prob, results) in enumerate(scale_results):
        lines.append(f"\n{'-' * 70}")
        lines.append(f" {scale_names[scale_idx]} - 概率: {prob:.4f}")
        lines.append(f"{'-' * 70}")
        lines.append(f"{'Rank':<6}{'ID':<10}{'Label':<8}{'Score':<12}{'Weight':<10}")
        lines.append("-" * 46)
        
        for i, r in enumerate(results[:5]):  # Show top 5
            payload = r.get('payload', {})
            label = payload.get('label', 'N/A')
            weight_norm = r.get('weight_normalized', 0)
            lines.append(
                f"{i+1:<6}"
                f"{r['id']:<10}"
                f"{label:<8}"
                f"{r['score']:<12.4f}"
                f"{weight_norm:<10.4f}"
            )
        
        if len(results) > 5:
            lines.append(f"  ... 还有 {len(results) - 5} 个结果")
    
    # Fusion result
    lines.append(f"\n{'=' * 70}")
    lines.append(" 融合结果")
    lines.append("=" * 70)
    lines.append(f"  融合模式: {fusion_result['mode']}")
    lines.append(f"  尺度权重: w0={fusion_result['weights'][0]:.3f}, w1={fusion_result['weights'][1]:.3f}, w2={fusion_result['weights'][2]:.3f}")
    
    # Format probabilities with both raw and 1-decimal display
    p0, p1, p2 = scale_results[0][0], scale_results[1][0], scale_results[2][0]
    fused_prob = fusion_result['probability']
    lines.append(f"  各尺度概率: p0={p0:.4f} ({p0:.1f}), p1={p1:.4f} ({p1:.1f}), p2={p2:.4f} ({p2:.1f})")
    lines.append(f"  融合概率: {fused_prob:.4f} (展示: {fused_prob:.1f})")
    lines.append(f"  预测: {'正类 (事故风险高)' if fused_prob >= 0.5 else '负类 (事故风险低)'}")
    
    # Comparison with true label
    pred = 1 if fusion_result['probability'] >= 0.5 else 0
    true = 1 if query_info['label'] >= 0.5 else 0
    correct = "✓ 正确" if pred == true else "✗ 错误"
    lines.append(f"  与真实标签对比: {correct}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    args = parse_args()
    
    json_output = str_to_bool(args.json_output)
    retrieve_only = str_to_bool(args.retrieve_only)
    exclude_self = str_to_bool(args.exclude_self)
    
    if not json_output:
        setup_logging()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if not json_output:
        logger.info(f"Using device: {device}")
        logger.info(f"Mode: {'retrieve_only' if retrieve_only else 'retrieve+fusion'}")
        logger.info(f"Exclude self: {exclude_self}")
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = MultiScaleEmbedder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        emb_dim=config['emb_dim'],
        dropout=config.get('dropout', 0.1),
        use_classification_head=config.get('use_classification_head', False)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load query data from NPZ
    data = np.load(args.npz_path, allow_pickle=True)
    
    query_idx = args.query_index
    x0 = torch.tensor(data['scale_0'][query_idx:query_idx+1], dtype=torch.float32, device=device)
    x1 = torch.tensor(data['scale_1'][query_idx:query_idx+1], dtype=torch.float32, device=device)
    x2 = torch.tensor(data['scale_2'][query_idx:query_idx+1], dtype=torch.float32, device=device)
    query_label = float(data['labels'][query_idx])
    
    # Get query embeddings
    with torch.no_grad():
        out = model(x0, x1, x2, return_attention=True)
        e0 = out['e0'][0].cpu().numpy().tolist()
        e1 = out['e1'][0].cpu().numpy().tolist()
        e2 = out['e2'][0].cpu().numpy().tolist()
    
    query_info = {
        'index': query_idx,
        'label': query_label
    }
    
    # Connect to Qdrant
    client = get_client(args.qdrant_url)
    
    # Collection names
    collection_names = [
        f"{args.collection_prefix}_scale0",
        f"{args.collection_prefix}_scale1",
        f"{args.collection_prefix}_scale2"
    ]
    
    # Calculate actual limit to request (extra for self-exclusion filtering)
    request_limit = args.top_k + args.min_results if exclude_self else args.top_k
    
    # Query each scale
    query_embeddings = [e0, e1, e2]
    raw_results_per_scale = []
    
    for scale_idx, (coll_name, query_emb) in enumerate(zip(collection_names, query_embeddings)):
        # Search with extra results
        results = search_similar(
            client, coll_name, query_emb,
            top_k=request_limit,
            with_payload=True
        )
        
        # Filter: exclude self and truncate to top_k
        filtered_results = filter_results(results, query_idx, exclude_self, args.top_k)
        raw_results_per_scale.append(filtered_results)
    
    # Output based on mode
    if retrieve_only:
        # Retrieve-only mode: just output the search results
        if json_output:
            output = {
                'query_index': query_idx,
                'true_label': round(query_label, 1),
                'results': {
                    'scale0': [
                        {
                            'rank': i + 1,
                            'id': r['id'],
                            'label': r.get('payload', {}).get('label', None),
                            'label_raw': round(r.get('payload', {}).get('label_raw', r.get('payload', {}).get('label', 0)), 1),
                            'score': round(r['score'], 4),
                            'sample_id': r.get('payload', {}).get('sample_id', r['id'])
                        }
                        for i, r in enumerate(raw_results_per_scale[0])
                    ],
                    'scale1': [
                        {
                            'rank': i + 1,
                            'id': r['id'],
                            'label': r.get('payload', {}).get('label', None),
                            'label_raw': round(r.get('payload', {}).get('label_raw', r.get('payload', {}).get('label', 0)), 1),
                            'score': round(r['score'], 4),
                            'sample_id': r.get('payload', {}).get('sample_id', r['id'])
                        }
                        for i, r in enumerate(raw_results_per_scale[1])
                    ],
                    'scale2': [
                        {
                            'rank': i + 1,
                            'id': r['id'],
                            'label': r.get('payload', {}).get('label', None),
                            'label_raw': round(r.get('payload', {}).get('label_raw', r.get('payload', {}).get('label', 0)), 1),
                            'score': round(r['score'], 4),
                            'sample_id': r.get('payload', {}).get('sample_id', r['id'])
                        }
                        for i, r in enumerate(raw_results_per_scale[2])
                    ]
                }
            }
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"Output saved to: {args.output_file}")
            else:
                print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            # Console display
            display = format_retrieve_only_display(query_info, raw_results_per_scale, args.top_k)
            print(display)
    
    else:
        # Full mode: compute probabilities and fusion
        # Determine fusion weights
        if args.fusion_mode == 'learned':
            if 'fusion_logits' in checkpoint:
                fusion_logits = torch.tensor(checkpoint['fusion_logits'])
                weights = torch.softmax(fusion_logits, dim=0).numpy().tolist()
            else:
                raise ValueError(
                    "Learned fusion mode requested but 'fusion_logits' not found in checkpoint. "
                    "Please use --fusion_mode fixed or retrain with a model that saves fusion weights."
                )
        else:
            # Normalize fixed weights
            total = args.w0 + args.w1 + args.w2
            weights = [args.w0 / total, args.w1 / total, args.w2 / total]
        
        # Compute probabilities for each scale
        scale_results = []
        for filtered_results in raw_results_per_scale:
            prob, weighted_results = compute_scale_probability(filtered_results, args.gamma)
            scale_results.append((prob, weighted_results))
        
        # Fuse probabilities
        p0, p1, p2 = scale_results[0][0], scale_results[1][0], scale_results[2][0]
        fused_prob = weights[0] * p0 + weights[1] * p1 + weights[2] * p2
        
        fusion_result = {
            'mode': args.fusion_mode,
            'weights': weights,
            'probability': fused_prob,
            'prediction': 1 if fused_prob >= 0.5 else 0
        }
        
        # Output
        if json_output:
            output = {
                'query': query_info,
                'scale_results': [
                    {
                        'scale': i,
                        'probability': prob,
                        'top_k': [
                            {
                                'rank': j + 1,
                                'id': r['id'],
                                'label': r.get('payload', {}).get('label', None),
                                'score': r['score'],
                                'weight': r.get('weight_normalized', 0),
                                'sample_id': r.get('payload', {}).get('sample_id', r['id']),
                                'attn_top_timesteps': r.get('payload', {}).get('attn_top_timesteps', [])
                            }
                            for j, r in enumerate(weighted_results)
                        ]
                    }
                    for i, (prob, weighted_results) in enumerate(scale_results)
                ],
                'fusion': {
                    **fusion_result,
                    'probability_1d': round(fusion_result['probability'], 1),  # 保留1位小数展示
                },
                'explanation': {
                    'p0': p0,
                    'p0_1d': round(p0, 1),
                    'p1': p1,
                    'p1_1d': round(p1, 1),
                    'p2': p2,
                    'p2_1d': round(p2, 1),
                    'w0': weights[0],
                    'w1': weights[1],
                    'w2': weights[2],
                    'gamma': args.gamma,
                    'formula': 'p = w0*p0 + w1*p1 + w2*p2'
                }
            }
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"Output saved to: {args.output_file}")
            else:
                print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            # Console display
            display = format_full_display(query_info, scale_results, fusion_result)
            print(display)


if __name__ == '__main__':
    main()
