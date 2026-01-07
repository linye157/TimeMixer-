"""
证据包构建器，用于准备 LLM 推理所需的信息。

包含：
- 查询样本统计特征
- 相似样本信息
- 尺度摘要
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def compute_query_stats(x: np.ndarray) -> Dict[str, float]:
    """
    计算查询样本的统计特征。
    
    Args:
        x: 48维温度序列
        
    Returns:
        统计特征字典
    """
    x = np.asarray(x).flatten()
    
    # 基础统计
    stats = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
    }
    
    # 末端斜率（最后10个点的线性拟合斜率）
    if len(x) >= 10:
        tail = x[-10:]
        t = np.arange(10)
        slope = np.polyfit(t, tail, 1)[0]
        stats["tail_slope"] = float(slope)
    else:
        stats["tail_slope"] = 0.0
    
    # 最大跳变
    diffs = np.abs(np.diff(x))
    max_jump_idx = int(np.argmax(diffs))
    max_jump_mag = float(diffs[max_jump_idx])
    stats["max_jump_idx"] = max_jump_idx
    stats["max_jump_mag"] = max_jump_mag
    
    # 趋势（整体斜率）
    t = np.arange(len(x))
    overall_slope = np.polyfit(t, x, 1)[0]
    stats["overall_slope"] = float(overall_slope)
    
    # 前后半段均值差
    half = len(x) // 2
    stats["half_diff"] = float(np.mean(x[half:]) - np.mean(x[:half]))
    
    return stats


def compute_sample_stats(x: np.ndarray) -> Dict[str, float]:
    """计算样本的简要统计（用于参考样本）。"""
    x = np.asarray(x).flatten()
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def compute_scale_summary(features: np.ndarray) -> Dict[str, Any]:
    """
    计算多尺度特征的摘要（不发送完整矩阵给 LLM）。
    
    Args:
        features: (L, d) 特征矩阵
        
    Returns:
        摘要字典
    """
    # Channel mean 得到时序曲线
    channel_mean = features.mean(axis=1)  # (L,)
    
    summary = {
        "length": len(channel_mean),
        "mean": float(np.mean(channel_mean)),
        "std": float(np.std(channel_mean)),
        "max_idx": int(np.argmax(channel_mean)),
        "min_idx": int(np.argmin(channel_mean)),
    }
    
    # 找异常点（偏离均值超过2倍标准差）
    mean_val = np.mean(channel_mean)
    std_val = np.std(channel_mean)
    if std_val > 1e-6:
        z_scores = np.abs((channel_mean - mean_val) / std_val)
        anomaly_indices = np.where(z_scores > 2)[0].tolist()[:3]  # 最多3个
        summary["anomaly_indices"] = anomaly_indices
    else:
        summary["anomaly_indices"] = []
    
    return summary


def build_evidence_pack(
    query_x: np.ndarray,
    ref_samples: Dict[str, List[Dict[str, Any]]],
    raw_data_loader: Optional[callable] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    p0: Optional[float] = None,
    p1: Optional[float] = None,
    p2: Optional[float] = None,
    scale_features: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    构建完整的证据包。
    
    Args:
        query_x: 48维查询向量
        ref_samples: 三尺度相似样本 {"scale0": [...], "scale1": [...], "scale2": [...]}
        raw_data_loader: 可选的函数，通过 sample_id 加载原始48维数据
        y1: TimeMixer++ 预测
        y2: RAG 投票预测
        p0, p1, p2: 各尺度概率
        scale_features: 可选的多尺度特征 {"S0": (48,64), "S1": (24,64), "S2": (12,64)}
        
    Returns:
        证据包字典
    """
    evidence = {}
    
    # 查询样本统计
    evidence["query_stats"] = compute_query_stats(query_x)
    evidence["query_x48"] = query_x.tolist() if isinstance(query_x, np.ndarray) else list(query_x)
    
    # 基线预测
    if y1 is not None:
        evidence["y1"] = float(y1)
    if y2 is not None:
        evidence["y2"] = float(y2)
    if p0 is not None:
        evidence["p0"] = float(p0)
    if p1 is not None:
        evidence["p1"] = float(p1)
    if p2 is not None:
        evidence["p2"] = float(p2)
    
    # 参考样本（添加统计信息）
    enriched_refs = {}
    for scale_key, samples in ref_samples.items():
        enriched = []
        for sample in samples:
            item = {
                "sample_id": sample.get("sample_id", sample.get("id")),
                "label_raw": sample.get("label_raw", sample.get("label", 0)),
                "score": sample.get("score", 0),
            }
            
            # 如果有原始数据加载器，获取参考样本的48维数据并计算统计
            if raw_data_loader is not None:
                try:
                    ref_x = raw_data_loader(item["sample_id"])
                    if ref_x is not None:
                        item["stats"] = compute_sample_stats(ref_x)
                except Exception:
                    item["stats"] = {}
            
            enriched.append(item)
        
        enriched_refs[scale_key] = enriched
    
    evidence["ref_samples"] = enriched_refs
    
    # 尺度特征摘要
    if scale_features is not None:
        scale_summaries = {}
        for key, feat in scale_features.items():
            if feat is not None:
                scale_summaries[key] = compute_scale_summary(feat)
        if scale_summaries:
            evidence["scale_summaries"] = scale_summaries
    
    return evidence


def get_valid_sample_ids(ref_samples: Dict[str, List[Dict]]) -> Dict[str, List[int]]:
    """从参考样本中提取有效的 sample_id 列表。"""
    valid_ids = {}
    for scale_key, samples in ref_samples.items():
        ids = []
        for s in samples:
            sid = s.get("sample_id", s.get("id"))
            if sid is not None:
                ids.append(int(sid))
        valid_ids[scale_key] = ids
    return valid_ids


def compute_final_probability(
    y1: Optional[float],
    y2: Optional[float],
    llm_response: Optional[Dict[str, Any]],
    ref_samples: Dict[str, List[Dict]],
    use_llm: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    计算最终概率。
    
    Args:
        y1: TimeMixer++ 预测
        y2: RAG 投票预测
        llm_response: LLM 验证后的响应
        ref_samples: 参考样本
        use_llm: 是否使用 LLM 结果
        
    Returns:
        (final_probability, computation_details)
    """
    details = {}
    
    if not use_llm or llm_response is None:
        # 非 LLM 模式：简单融合
        if y1 is not None and y2 is not None:
            final = (y1 + y2) / 2
            details["mode"] = "avg_y1_y2"
        elif y1 is not None:
            final = y1
            details["mode"] = "y1_only"
        elif y2 is not None:
            final = y2
            details["mode"] = "y2_only"
        else:
            final = 0.5
            details["mode"] = "fallback"
        
        details["final"] = float(final)
        return float(final), details
    
    # LLM 模式：根据权重计算
    alpha = llm_response.get("alpha", {})
    beta = llm_response.get("beta_scale", {})
    ref_weights = llm_response.get("ref_weights", {})
    
    a_y1 = alpha.get("a_y1", 0)
    a_y2 = alpha.get("a_y2", 0)
    a_ref = alpha.get("a_ref", 0)
    
    b0 = beta.get("b0", 0.5)
    b1 = beta.get("b1", 0.3)
    b2 = beta.get("b2", 0.2)
    
    # 计算 y_ref
    y_ref = 0.0
    scale_probs = []
    
    for scale_idx, (scale_key, b_weight) in enumerate([
        ("scale0", b0), ("scale1", b1), ("scale2", b2)
    ]):
        scale_refs = ref_samples.get(scale_key, [])
        llm_weights = {
            item["sample_id"]: item["weight"]
            for item in ref_weights.get(scale_key, [])
        }
        
        scale_prob = 0.0
        total_weight = 0.0
        
        for ref in scale_refs:
            sid = ref.get("sample_id", ref.get("id"))
            label = ref.get("label_raw", ref.get("label", 0))
            
            # 使用 LLM 给的权重，如果没有则跳过
            w = llm_weights.get(sid, 0)
            if w > 0:
                scale_prob += w * label
                total_weight += w
        
        if total_weight > 0:
            scale_prob /= total_weight
        else:
            # Fallback: 使用相似度加权
            for ref in scale_refs:
                label = ref.get("label_raw", ref.get("label", 0))
                score = ref.get("score", 0)
                w = np.exp(10 * score)  # gamma=10
                scale_prob += w * label
                total_weight += w
            if total_weight > 0:
                scale_prob /= total_weight
        
        scale_probs.append(scale_prob)
        y_ref += b_weight * scale_prob
    
    details["y_ref"] = float(y_ref)
    details["scale_probs_llm"] = scale_probs
    
    # 最终融合
    y1_val = y1 if y1 is not None else 0
    y2_val = y2 if y2 is not None else 0
    
    final = a_y1 * y1_val + a_y2 * y2_val + a_ref * y_ref
    final = max(0, min(1, final))  # Clamp to [0, 1]
    
    details["mode"] = "llm_fusion"
    details["alpha"] = alpha
    details["beta"] = beta
    details["final"] = float(final)
    
    return float(final), details


