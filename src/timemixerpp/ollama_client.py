"""
Ollama API 客户端，用于与本地 LLM 交互。

支持：
- 结构化 JSON 输出
- 温度控制
- 超时处理
- 响应解析与验证
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama API 客户端。"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        temperature: float = 0.0,
        timeout: int = 120
    ):
        """
        初始化 Ollama 客户端。
        
        Args:
            base_url: Ollama 服务地址
            model: 模型名称
            temperature: 生成温度（0.0 = 确定性输出）
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = True
    ) -> Dict[str, Any]:
        """
        发送聊天请求。
        
        Args:
            messages: 消息列表 [{"role": "user/assistant/system", "content": "..."}]
            json_mode: 是否要求 JSON 输出
            
        Returns:
            响应字典，包含 content 和 parsed_json（如果 json_mode=True）
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        
        if json_mode:
            payload["format"] = "json"
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            content = result.get("message", {}).get("content", "")
            
            response_dict = {
                "content": content,
                "model": result.get("model"),
                "done": result.get("done", True),
                "total_duration": result.get("total_duration"),
            }
            
            # 尝试解析 JSON
            if json_mode and content:
                parsed = self._parse_json(content)
                response_dict["parsed_json"] = parsed
                response_dict["json_valid"] = parsed is not None
            
            return response_dict
            
        except urllib.error.URLError as e:
            logger.error(f"Ollama request failed: {e}")
            return {"error": str(e), "content": "", "json_valid": False}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            return {"error": str(e), "content": "", "json_valid": False}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": str(e), "content": "", "json_valid": False}
    
    def _parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        """
        从响应内容中解析 JSON。
        
        支持：
        - 纯 JSON 字符串
        - Markdown 代码块中的 JSON
        - 混合文本中的 JSON
        """
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 尝试从 Markdown 代码块提取
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试找到 JSON 对象
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def check_connection(self) -> bool:
        """检查与 Ollama 服务的连接。"""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """列出可用模型。"""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return [m["name"] for m in result.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


def build_prediction_prompt(
    evidence: Dict[str, Any],
    provide_y1: bool = False,
    provide_y2: bool = False
) -> str:
    """
    构建预测 Prompt。
    
    Args:
        evidence: 证据包（query_stats, ref_samples, y1, y2 等）
        provide_y1: 是否提供 TimeMixer++ 预测
        provide_y2: 是否提供 RAG 投票预测
        
    Returns:
        Prompt 字符串
    """
    prompt_parts = []
    
    # 系统指令
    prompt_parts.append("""你是一个工业事故风险预测专家。根据提供的温度时序数据和相似样本信息，预测事故发生概率。

输出格式要求（ONLY JSON，不要其他文字）：
{
  "alpha": {"a_y1": 0.0~1.0, "a_y2": 0.0~1.0, "a_ref": 0.0~1.0},
  "beta_scale": {"b0": 0.0~1.0, "b1": 0.0~1.0, "b2": 0.0~1.0},
  "ref_weights": {
    "scale0": [{"sample_id": int, "weight": float}, ...],
    "scale1": [...],
    "scale2": [...]
  },
  "y3_llm": 0.0~1.0,
  "explanation": "引用具体 sample_id 和相似度说明判断依据",
  "uncertainty": 0.0~1.0
}

注意：
- alpha 权重用于融合 y1(TimeMixer预测)、y2(RAG投票)、y_ref(参考样本加权)
- 若未提供 y1，则 a_y1 必须为 0
- 若未提供 y2，则 a_y2 必须为 0
- beta_scale 权重用于加权三个尺度的参考样本
- ref_weights 选择你认为最有参考价值的样本及其权重
- uncertainty 表示你对预测的不确定性""")
    
    # 查询样本统计
    query_stats = evidence.get("query_stats", {})
    prompt_parts.append(f"""
【查询样本统计】
- 均值: {query_stats.get('mean', 0):.4f}
- 标准差: {query_stats.get('std', 0):.4f}
- 最小值: {query_stats.get('min', 0):.4f}
- 最大值: {query_stats.get('max', 0):.4f}
- 末端斜率: {query_stats.get('tail_slope', 0):.4f}
- 最大跳变位置: {query_stats.get('max_jump_idx', 0)}
- 最大跳变幅度: {query_stats.get('max_jump_mag', 0):.4f}""")
    
    # 基线预测（如果提供）
    if provide_y1 and "y1" in evidence:
        prompt_parts.append(f"""
【TimeMixer++ 预测 (y1)】
y1 = {evidence['y1']:.4f}""")
    
    if provide_y2 and "y2" in evidence:
        prompt_parts.append(f"""
【RAG 投票预测 (y2)】
y2 = {evidence['y2']:.4f}
- 尺度0概率: {evidence.get('p0', 0):.4f}
- 尺度1概率: {evidence.get('p1', 0):.4f}
- 尺度2概率: {evidence.get('p2', 0):.4f}""")
    
    # 相似样本
    for scale_idx in range(3):
        scale_key = f"scale{scale_idx}"
        refs = evidence.get("ref_samples", {}).get(scale_key, [])
        if refs:
            scale_name = ["48步", "24步", "12步"][scale_idx]
            prompt_parts.append(f"""
【尺度{scale_idx} ({scale_name}) 相似样本】""")
            for ref in refs[:5]:  # 限制数量
                prompt_parts.append(
                    f"  - ID:{ref['sample_id']}, "
                    f"标签:{ref['label_raw']:.2f}, "
                    f"相似度:{ref['score']:.4f}, "
                    f"统计:(μ={ref.get('stats', {}).get('mean', 0):.2f}, σ={ref.get('stats', {}).get('std', 0):.2f})"
                )
    
    prompt_parts.append("""
请基于以上信息，输出预测 JSON：""")
    
    return "\n".join(prompt_parts)


def validate_llm_response(
    response: Dict[str, Any],
    valid_sample_ids: Dict[str, List[int]],
    has_y1: bool,
    has_y2: bool
) -> Dict[str, Any]:
    """
    验证并修正 LLM 响应。
    
    Args:
        response: LLM 解析出的 JSON
        valid_sample_ids: 每个尺度的有效 sample_id 列表
        has_y1: 是否有 y1
        has_y2: 是否有 y2
        
    Returns:
        修正后的响应
    """
    if response is None:
        return get_default_response(has_y1, has_y2)
    
    result = {}
    
    # 验证 alpha
    alpha = response.get("alpha", {})
    a_y1 = float(alpha.get("a_y1", 0)) if has_y1 else 0.0
    a_y2 = float(alpha.get("a_y2", 0)) if has_y2 else 0.0
    a_ref = float(alpha.get("a_ref", 1.0))
    
    # 确保非负
    a_y1 = max(0, a_y1)
    a_y2 = max(0, a_y2)
    a_ref = max(0, a_ref)
    
    # 归一化
    total = a_y1 + a_y2 + a_ref
    if total > 0:
        a_y1 /= total
        a_y2 /= total
        a_ref /= total
    else:
        a_ref = 1.0
    
    result["alpha"] = {"a_y1": a_y1, "a_y2": a_y2, "a_ref": a_ref}
    
    # 验证 beta_scale
    beta = response.get("beta_scale", {})
    b0 = max(0, float(beta.get("b0", 0.5)))
    b1 = max(0, float(beta.get("b1", 0.3)))
    b2 = max(0, float(beta.get("b2", 0.2)))
    
    total = b0 + b1 + b2
    if total > 0:
        b0 /= total
        b1 /= total
        b2 /= total
    else:
        b0, b1, b2 = 0.5, 0.3, 0.2
    
    result["beta_scale"] = {"b0": b0, "b1": b1, "b2": b2}
    
    # 验证 ref_weights
    ref_weights = {}
    for scale_key in ["scale0", "scale1", "scale2"]:
        valid_ids = set(valid_sample_ids.get(scale_key, []))
        raw_weights = response.get("ref_weights", {}).get(scale_key, [])
        
        cleaned = []
        for item in raw_weights:
            if isinstance(item, dict):
                sid = item.get("sample_id")
                w = item.get("weight", 0)
                if sid in valid_ids and w > 0:
                    cleaned.append({"sample_id": sid, "weight": float(w)})
        
        # 归一化
        total = sum(item["weight"] for item in cleaned)
        if total > 0:
            for item in cleaned:
                item["weight"] /= total
        
        ref_weights[scale_key] = cleaned
    
    result["ref_weights"] = ref_weights
    
    # 其他字段
    result["y3_llm"] = float(response.get("y3_llm", 0.5))
    result["y3_llm"] = max(0, min(1, result["y3_llm"]))
    
    result["explanation"] = response.get("explanation", "")
    result["uncertainty"] = max(0, min(1, float(response.get("uncertainty", 0.5))))
    
    return result


def get_default_response(has_y1: bool, has_y2: bool) -> Dict[str, Any]:
    """获取默认响应（LLM 失败时的 fallback）。"""
    return {
        "alpha": {
            "a_y1": 0.5 if has_y1 else 0.0,
            "a_y2": 0.5 if has_y2 else 0.0,
            "a_ref": 0.0 if (has_y1 or has_y2) else 1.0
        },
        "beta_scale": {"b0": 0.5, "b1": 0.3, "b2": 0.2},
        "ref_weights": {"scale0": [], "scale1": [], "scale2": []},
        "y3_llm": 0.5,
        "explanation": "LLM 响应无效，使用默认权重",
        "uncertainty": 1.0
    }


