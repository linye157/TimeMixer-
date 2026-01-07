#!/usr/bin/env python
"""
TimeMixer++ y1 + RAG投票 y2 + LLM 输出 y3 + 解释的综合推理脚本。

支持：
- 单条输入（48个逗号分隔的数值）
- 批量输入（xlsx/csv 文件）
- 三尺度知识库检索
- TimeMixer++ 模型预测
- LLM 增强解释

Usage:
    REM 单条输入（inline）
    python scripts/predict_with_ollama_rag.py --input_inline "25.1,25.3,25.5,..." --qdrant_url http://localhost:6333 --collection_prefix raw_temperature_kb --use_y2 true

    REM 批量输入（xlsx 文件）
    python scripts/predict_with_ollama_rag.py --data_path TDdata/alldata.xlsx --qdrant_url http://localhost:6333 --collection_prefix raw_temperature_kb --use_y2 true --llm_mode none --output_dir results

    REM 批量输入（csv 文件）
    python scripts/predict_with_ollama_rag.py --data_path TDdata/TrainData.csv --qdrant_url http://localhost:6333 --collection_prefix raw_temperature_kb --use_y2 true --output_dir results

    REM 启用 LLM
    python scripts/predict_with_ollama_rag.py --data_path TDdata/alldata.xlsx --qdrant_url http://localhost:6333 --collection_prefix raw_temperature_kb --use_y2 true --llm_mode all --provide_y2_to_llm true --ollama_url http://localhost:11434 --ollama_model qwen2.5:7b

    REM 完整模式：y1 + y2 + LLM
    python scripts/predict_with_ollama_rag.py --data_path TDdata/alldata.xlsx --qdrant_url http://localhost:6333 --collection_prefix raw_temperature_kb --use_y1 true --use_y2 true --timemixer_ckpt checkpoints/best_model.pt --llm_mode all
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import json
import logging
import csv
import os
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

from timemixerpp.data import load_file_strict
from timemixerpp.qdrant_utils import get_client, search_similar
from timemixerpp.ollama_client import (
    OllamaClient, build_prediction_prompt,
    validate_llm_response, get_default_response
)
from timemixerpp.evidence_builder import (
    build_evidence_pack, get_valid_sample_ids,
    compute_final_probability, compute_query_stats
)
from timemixerpp.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='TimeMixer++ + RAG + LLM 综合推理',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入源（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_inline', type=str,
                              help='48维向量（逗号分隔）')
    input_group.add_argument('--data_path', type=str,
                              help='批量输入文件路径（xlsx 或 csv）')
    
    # 样本范围（可选，用于批量输入）
    parser.add_argument('--start_idx', type=int, default=0,
                        help='起始样本索引（批量输入时有效）')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='结束样本索引（批量输入时有效，None 表示到末尾）')
    
    # Qdrant 配置
    parser.add_argument('--qdrant_url', type=str, default='http://localhost:6333',
                        help='Qdrant 服务地址')
    parser.add_argument('--collection_prefix', type=str, required=True,
                        help='Collection 名称前缀（会自动添加 _scale0/_scale1/_scale2）')
    parser.add_argument('--top_k', type=int, default=10,
                        help='检索的相似样本数')
    parser.add_argument('--gamma', type=float, default=10.0,
                        help='相似度加权系数')
    parser.add_argument('--fusion_weights', type=str, default='0.5,0.3,0.2',
                        help='三尺度融合权重（逗号分隔）')
    parser.add_argument('--exclude_self', type=str, default='true',
                        help='是否排除自身')
    parser.add_argument('--min_results', type=int, default=10,
                        help='额外请求的结果数（用于过滤后保证足够 top_k）')
    
    # 向量归一化（需与入库时一致）
    parser.add_argument('--l2_normalize', action='store_true',
                        help='对查询向量进行 L2 归一化（需与入库时一致）')
    
    # TimeMixer++ 配置
    parser.add_argument('--timemixer_ckpt', type=str, default=None,
                        help='TimeMixer++ checkpoint 路径（use_y1=true 时需要）')
    
    # 基线开关
    parser.add_argument('--use_y1', type=str, default='false',
                        help='是否计算 TimeMixer++ 预测 y1')
    parser.add_argument('--use_y2', type=str, default='true',
                        help='是否计算 RAG 投票融合 y2')
    
    # 传给 LLM 的开关
    parser.add_argument('--provide_y1_to_llm', type=str, default='false',
                        help='是否将 y1 提供给 LLM')
    parser.add_argument('--provide_y2_to_llm', type=str, default='false',
                        help='是否将 y2 提供给 LLM')
    
    # LLM 配置
    parser.add_argument('--llm_mode', type=str, default='none',
                        choices=['none', 'top', 'uncertain', 'all'],
                        help='LLM 触发模式')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='不确定性阈值（用于 uncertain 模式）')
    parser.add_argument('--delta', type=float, default=0.1,
                        help='概率接近阈值范围（用于 uncertain 模式）')
    parser.add_argument('--user_confirm', type=str, default='true',
                        help='批量模式下是否需要用户确认')
    parser.add_argument('--ollama_url', type=str, default='http://localhost:11434',
                        help='Ollama 服务地址')
    parser.add_argument('--ollama_model', type=str, default='qwen2.5:7b',
                        help='Ollama 模型名称')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM 生成温度')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    parser.add_argument('--final_mode', type=str, default='avg',
                        choices=['y1', 'y2', 'avg'],
                        help='非 LLM 模式下的最终概率计算方式')
    
    return parser.parse_args()


def str_to_bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ('true', '1', 'yes', 'on')


def load_input_data(args) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    加载输入数据。
    
    支持：
    - 单条输入（--input_inline）
    - 批量输入（--data_path，支持 xlsx/csv）
    
    Returns:
        (data, labels) - data shape: (N, 48), labels shape: (N,) or None
    """
    if args.input_inline:
        # 单条输入
        values = [float(x.strip()) for x in args.input_inline.split(',')]
        if len(values) != 48:
            raise ValueError(f"输入向量必须是48维，实际为 {len(values)} 维")
        return np.array([values], dtype=np.float32), None
    
    elif args.data_path:
        # 批量输入：使用 load_file_strict 读取 xlsx/csv
        logger.info(f"从文件加载数据: {args.data_path}")
        _, X, y = load_file_strict(args.data_path)
        
        # 应用样本范围
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else len(X)
        
        if start_idx >= len(X):
            raise ValueError(f"start_idx ({start_idx}) 超出数据范围 (0-{len(X)-1})")
        if end_idx > len(X):
            end_idx = len(X)
        
        X = X[start_idx:end_idx].astype(np.float32)
        y = y[start_idx:end_idx].astype(np.float32)
        
        logger.info(f"加载样本范围: [{start_idx}, {end_idx})，共 {len(X)} 个样本")
        
        return X, y
    
    else:
        raise ValueError("必须提供 --input_inline 或 --data_path")


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """L2 normalize a vector."""
    norm = np.linalg.norm(x)
    if norm > 1e-8:
        return x / norm
    return x


def retrieve_from_3scales(
    client,
    collection_prefix: str,
    query_vector: np.ndarray,
    top_k: int,
    min_results: int,
    exclude_self: bool,
    query_index: Optional[int] = None
) -> Dict[str, List[Dict]]:
    """
    从三尺度 collection 检索相似样本。
    
    Args:
        client: Qdrant 客户端
        collection_prefix: Collection 前缀
        query_vector: 48维查询向量
        top_k: 返回的结果数
        min_results: 额外请求的结果数（用于过滤）
        exclude_self: 是否排除自身
        query_index: 查询样本索引（用于排除自身）
        
    Returns:
        三尺度检索结果 {"scale0": [...], "scale1": [...], "scale2": [...]}
    """
    collection_names = [
        f"{collection_prefix}_scale0",
        f"{collection_prefix}_scale1",
        f"{collection_prefix}_scale2"
    ]
    
    results = {}
    request_limit = top_k + min_results if exclude_self else top_k
    
    for scale_idx, coll_name in enumerate(collection_names):
        scale_key = f"scale{scale_idx}"
        
        try:
            raw_results = search_similar(
                client, coll_name, query_vector.tolist(),
                top_k=request_limit, with_payload=True
            )
            
            # 过滤并处理结果
            filtered = []
            for r in raw_results:
                sample_id = r.get('payload', {}).get('sample_id', r.get('id'))
                
                # 排除自身
                if exclude_self and query_index is not None and sample_id == query_index:
                    continue
                
                filtered.append({
                    'id': r.get('id'),
                    'sample_id': sample_id,
                    'score': r.get('score', 0),
                    'label': r.get('payload', {}).get('label', 0),
                    'label_raw': r.get('payload', {}).get('label_raw', 
                                  r.get('payload', {}).get('label', 0)),
                })
                
                if len(filtered) >= top_k:
                    break
            
            results[scale_key] = filtered
            
        except Exception as e:
            logger.warning(f"检索 {coll_name} 失败: {e}")
            results[scale_key] = []
    
    return results


def compute_rag_probability(
    ref_samples: Dict[str, List[Dict]],
    gamma: float,
    fusion_weights: Tuple[float, float, float]
) -> Tuple[float, float, float, float]:
    """
    计算 RAG 投票概率。
    
    p_m = Σ w_i * label_i / Σ w_i  (w_i = exp(gamma * score_i))
    y2 = w0 * p0 + w1 * p1 + w2 * p2
    
    Returns:
        (y2, p0, p1, p2)
    """
    w0, w1, w2 = fusion_weights
    
    scale_probs = []
    for scale_key in ["scale0", "scale1", "scale2"]:
        samples = ref_samples.get(scale_key, [])
        
        if not samples:
            scale_probs.append(0.5)
            continue
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for s in samples:
            score = s.get('score', 0)
            label = s.get('label_raw', s.get('label', 0))
            w = np.exp(gamma * score)
            
            total_weight += w
            weighted_sum += w * label
        
        if total_weight > 0:
            scale_probs.append(weighted_sum / total_weight)
        else:
            scale_probs.append(0.5)
    
    p0, p1, p2 = scale_probs
    y2 = w0 * p0 + w1 * p1 + w2 * p2
    
    return y2, p0, p1, p2


def should_trigger_llm(
    llm_mode: str,
    y1: Optional[float],
    y2: Optional[float],
    threshold: float,
    delta: float,
    sample_idx: int = 0
) -> bool:
    """判断是否应该触发 LLM。"""
    if llm_mode == 'none':
        return False
    
    if llm_mode == 'all':
        return True
    
    if llm_mode == 'top':
        # 只对第一个样本触发
        return sample_idx == 0
    
    if llm_mode == 'uncertain':
        # 当 y1 或 y2 接近阈值时触发
        prob = y2 if y2 is not None else (y1 if y1 is not None else 0.5)
        return abs(prob - 0.5) < delta or abs(prob - threshold) < delta
    
    return False


def compute_timemixer_prediction(x: np.ndarray, ckpt_path: str) -> float:
    """
    使用 TimeMixer++ 模型计算预测。
    
    Args:
        x: 48维输入向量
        ckpt_path: checkpoint 路径
        
    Returns:
        预测概率 [0, 1]
    """
    from timemixerpp.model import TimeMixerPPForBinaryCls
    from timemixerpp.config import TimeMixerPPConfig
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get('config', {})
    
    # 创建配置和模型
    config = TimeMixerPPConfig(
        seq_len=config_dict.get('seq_len', 48),
        c_in=config_dict.get('c_in', 1),
        d_model=config_dict.get('d_model', 64),
        n_layers=config_dict.get('n_layers', 2),
        n_heads=config_dict.get('n_heads', 4),
        top_k=config_dict.get('top_k', 3),
        dropout=config_dict.get('dropout', 0.1),
    )
    
    model = TimeMixerPPForBinaryCls(config).to(device)
    
    # Dummy forward to initialize lazy layers
    with torch.no_grad():
        dummy = torch.randn(1, config.seq_len).to(device)
        _ = model(dummy)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 归一化（如果有）
    mean = checkpoint.get('normalizer_mean')
    std = checkpoint.get('normalizer_std')
    
    x_input = x.copy()
    if mean is not None and std is not None:
        x_input = (x_input - mean.flatten()) / std.flatten()
    
    # 预测
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(x_tensor)
        prob = output['probs'].item()
    
    return prob


def create_raw_data_loader(X: np.ndarray, start_idx: int = 0):
    """创建原始数据加载器函数。"""
    def loader(sample_id: int) -> Optional[np.ndarray]:
        # sample_id 是全局索引，需要调整为局部索引
        local_idx = sample_id - start_idx
        if 0 <= local_idx < len(X):
            return X[local_idx]
        return None
    return loader


def process_single_sample(
    sample_idx: int,
    global_idx: int,
    x: np.ndarray,
    label: Optional[float],
    args,
    qdrant_client,
    raw_data_loader: Optional[callable],
    ollama_client: Optional[OllamaClient],
    fusion_weights: Tuple[float, float, float],
    use_y1: bool,
    use_y2: bool,
    provide_y1_to_llm: bool,
    provide_y2_to_llm: bool,
    exclude_self: bool,
    l2_normalize: bool
) -> Dict[str, Any]:
    """
    处理单个样本。
    
    Args:
        sample_idx: 批次内索引
        global_idx: 全局索引（用于排除自身）
        x: 48维输入向量
        label: 真实标签（可选）
        ...
    """
    result = {
        "sample_id": global_idx,
    }
    
    if label is not None:
        result["true_label"] = round(float(label), 4)
        result["true_label_display"] = round(float(label), 1)
    
    y1 = None
    y2 = None
    p0, p1, p2 = None, None, None
    ref_samples = {}
    
    # 准备查询向量
    query_vector = x.copy()
    if l2_normalize:
        query_vector = normalize_l2(query_vector)
    
    # Step 1: 从三尺度知识库检索相似样本
    ref_samples = retrieve_from_3scales(
        qdrant_client,
        args.collection_prefix,
        query_vector,
        args.top_k,
        args.min_results,
        exclude_self,
        query_index=global_idx
    )
    
    # 记录检索到的样本数
    result["retrieved_counts"] = {
        "scale0": len(ref_samples.get("scale0", [])),
        "scale1": len(ref_samples.get("scale1", [])),
        "scale2": len(ref_samples.get("scale2", [])),
    }
    
    # Step 2: 计算 y2 (RAG 投票)
    if use_y2:
        y2, p0, p1, p2 = compute_rag_probability(ref_samples, args.gamma, fusion_weights)
        result["y2_rag_vote"] = round(y2, 4)
        result["y2_display"] = round(y2, 1)
        result["p0"] = round(p0, 4)
        result["p1"] = round(p1, 4)
        result["p2"] = round(p2, 4)
    
    # Step 3: 计算 y1 (TimeMixer++)
    if use_y1:
        if args.timemixer_ckpt:
            try:
                y1 = compute_timemixer_prediction(x, args.timemixer_ckpt)
                result["y1_timemixer"] = round(y1, 4)
                result["y1_display"] = round(y1, 1)
            except Exception as e:
                logger.warning(f"TimeMixer++ 预测失败: {e}")
                y1 = None
        else:
            logger.warning("use_y1=true 但未提供 --timemixer_ckpt，跳过 y1 计算")
    
    # Step 4: LLM 预测
    trigger_llm = should_trigger_llm(
        args.llm_mode, y1, y2,
        args.threshold, args.delta, sample_idx
    )
    
    llm_response = None
    if trigger_llm and ollama_client is not None:
        # 构建证据包
        evidence = build_evidence_pack(
            query_x=x,
            ref_samples=ref_samples,
            raw_data_loader=raw_data_loader,
            y1=y1 if provide_y1_to_llm else None,
            y2=y2 if provide_y2_to_llm else None,
            p0=p0, p1=p1, p2=p2
        )
        
        # 构建 Prompt
        prompt = build_prediction_prompt(
            evidence,
            provide_y1=provide_y1_to_llm and y1 is not None,
            provide_y2=provide_y2_to_llm and y2 is not None
        )
        
        # 调用 LLM
        messages = [{"role": "user", "content": prompt}]
        llm_result = ollama_client.chat(messages, json_mode=True)
        
        if llm_result.get("json_valid"):
            valid_ids = get_valid_sample_ids(ref_samples)
            llm_response = validate_llm_response(
                llm_result.get("parsed_json"),
                valid_ids,
                has_y1=y1 is not None,
                has_y2=y2 is not None
            )
            result["y3_llm"] = round(llm_response.get("y3_llm", 0.5), 4)
            result["y3_display"] = round(llm_response.get("y3_llm", 0.5), 1)
            result["llm_explanation"] = llm_response.get("explanation")
            result["llm_uncertainty"] = round(llm_response.get("uncertainty", 0.5), 4)
        else:
            logger.warning(f"LLM 响应无效: {llm_result.get('error', 'unknown')}")
            llm_response = get_default_response(y1 is not None, y2 is not None)
    
    # Step 5: 计算最终概率
    final_prob, details = compute_final_probability(
        y1, y2, llm_response, ref_samples,
        use_llm=(llm_response is not None)
    )
    
    result["final_probability"] = round(final_prob, 4)
    result["final_probability_display"] = round(final_prob, 1)
    result["computation_mode"] = details.get("mode")
    
    if llm_response is not None:
        result["explanation"] = {
            "llm_weights": llm_response.get("alpha"),
            "scale_weights": llm_response.get("beta_scale"),
            "reasoning": llm_response.get("explanation"),
            "uncertainty": llm_response.get("uncertainty")
        }
    
    return result


def main():
    args = parse_args()
    
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("TimeMixer++ + RAG + LLM 综合推理")
    logger.info("=" * 60)
    
    # 解析参数
    use_y1 = str_to_bool(args.use_y1)
    use_y2 = str_to_bool(args.use_y2)
    provide_y1_to_llm = str_to_bool(args.provide_y1_to_llm)
    provide_y2_to_llm = str_to_bool(args.provide_y2_to_llm)
    exclude_self = str_to_bool(args.exclude_self)
    user_confirm = str_to_bool(args.user_confirm)
    
    fusion_weights = tuple(float(x) for x in args.fusion_weights.split(','))
    # 归一化
    total = sum(fusion_weights)
    fusion_weights = tuple(w / total for w in fusion_weights)
    
    logger.info(f"融合权重: w0={fusion_weights[0]:.3f}, w1={fusion_weights[1]:.3f}, w2={fusion_weights[2]:.3f}")
    logger.info(f"use_y1={use_y1}, use_y2={use_y2}")
    logger.info(f"L2 归一化: {args.l2_normalize}")
    
    # 验证参数
    if not use_y1 and not use_y2:
        raise ValueError("必须至少启用 use_y1 或 use_y2 之一")
    
    # 加载输入数据
    logger.info("加载输入数据...")
    X, labels = load_input_data(args)
    n_samples = len(X)
    logger.info(f"加载了 {n_samples} 个样本")
    
    if labels is not None:
        pos_ratio = np.mean(labels >= 0.5)
        logger.info(f"标签分布: 正类 {pos_ratio:.2%}, 负类 {1-pos_ratio:.2%}")
    
    # 计算全局起始索引
    global_start_idx = args.start_idx if args.data_path else 0
    
    # 创建原始数据加载器（用于 LLM 证据构建）
    raw_data_loader = create_raw_data_loader(X, global_start_idx)
    
    # 连接 Qdrant
    logger.info(f"连接 Qdrant: {args.qdrant_url}")
    qdrant_client = get_client(args.qdrant_url)
    
    # 初始化 Ollama 客户端
    ollama_client = None
    if args.llm_mode != 'none':
        logger.info(f"初始化 Ollama 客户端: {args.ollama_url}, 模型: {args.ollama_model}")
        ollama_client = OllamaClient(
            base_url=args.ollama_url,
            model=args.ollama_model,
            temperature=args.temperature
        )
        
        if not ollama_client.check_connection():
            logger.warning("无法连接到 Ollama 服务，LLM 功能将被禁用")
            ollama_client = None
    
    # 批量模式用户确认
    if n_samples > 50 and args.llm_mode in ['top', 'uncertain', 'all'] and user_confirm and ollama_client is not None:
        # 预估触发 LLM 的样本数
        llm_count = 0
        for i in range(n_samples):
            if should_trigger_llm(args.llm_mode, None, None, args.threshold, args.delta, i):
                llm_count += 1
        
        logger.info(f"预计将对 {llm_count}/{n_samples} 个样本调用 LLM")
        try:
            confirm = input("是否继续? (y/n): ").strip().lower()
            if confirm != 'y':
                logger.info("用户取消，将不调用 LLM")
                ollama_client = None
        except EOFError:
            # 非交互模式，继续执行
            pass
    
    # 处理所有样本
    logger.info("开始处理样本...")
    results = []
    
    for i in range(n_samples):
        x = X[i]
        label = labels[i] if labels is not None else None
        global_idx = global_start_idx + i
        
        result = process_single_sample(
            sample_idx=i,
            global_idx=global_idx,
            x=x,
            label=label,
            args=args,
            qdrant_client=qdrant_client,
            raw_data_loader=raw_data_loader,
            ollama_client=ollama_client,
            fusion_weights=fusion_weights,
            use_y1=use_y1,
            use_y2=use_y2,
            provide_y1_to_llm=provide_y1_to_llm,
            provide_y2_to_llm=provide_y2_to_llm,
            exclude_self=exclude_self,
            l2_normalize=args.l2_normalize
        )
        
        results.append(result)
        
        if (i + 1) % 10 == 0 or i == n_samples - 1:
            logger.info(f"已处理 {i + 1}/{n_samples} 个样本")
    
    # 输出结果
    if n_samples == 1:
        # 单条：直接打印 JSON
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    else:
        # 批量：写文件
        os.makedirs(args.output_dir, exist_ok=True)
        
        # JSONL
        jsonl_path = os.path.join(args.output_dir, 'results.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        logger.info(f"JSONL 结果保存到: {jsonl_path}")
        
        # CSV
        csv_path = os.path.join(args.output_dir, 'results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'final_probability', 'display',
                'y1', 'y2', 'true_label', 'mode'
            ])
            for r in results:
                writer.writerow([
                    r.get('sample_id'),
                    r.get('final_probability'),
                    r.get('final_probability_display'),
                    r.get('y1_timemixer', ''),
                    r.get('y2_rag_vote', ''),
                    r.get('true_label', ''),
                    r.get('computation_mode', '')
                ])
        logger.info(f"CSV 结果保存到: {csv_path}")
        
        # 统计信息
        final_probs = [r.get('final_probability', 0.5) for r in results]
        logger.info(f"\n统计信息:")
        logger.info(f"  样本数: {len(results)}")
        logger.info(f"  平均概率: {np.mean(final_probs):.4f}")
        logger.info(f"  预测正类数: {sum(1 for p in final_probs if p >= 0.5)}")
        
        if labels is not None:
            # 计算准确率
            preds = [1 if r.get('final_probability', 0.5) >= 0.5 else 0 for r in results]
            true_labels = [1 if r.get('true_label', 0) >= 0.5 else 0 for r in results]
            accuracy = sum(1 for p, t in zip(preds, true_labels) if p == t) / len(preds)
            logger.info(f"  准确率: {accuracy:.4f}")
    
    logger.info("=" * 60)
    logger.info("处理完成!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
