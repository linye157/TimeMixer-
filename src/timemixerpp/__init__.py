"""
TimeMixer++ Implementation for Binary Classification

A modular implementation of the TimeMixer++ architecture with:
- MRTI: Multi-Resolution Time Imaging
- TID: Time Image Decomposition
- MCM: Multi-Scale Mixing
- MRM: Multi-Resolution Mixing

Also includes:
- Metric Learning: TemporalConvEmbedder for sequence embeddings
- Losses: SupConLoss for contrastive learning
- Qdrant Utils: Vector database integration
"""

from .config import TimeMixerPPConfig
from .model import TimeMixerPPEncoder, TimeMixerPPForBinaryCls
from .block import MixerBlock
from .mrti import MRTI
from .tid import TID
from .mcm import MCM
from .mrm import MRM

# Metric learning components
from .metric_encoder import TemporalConvEmbedder, MultiScaleEmbedder, AttentionPooling
from .losses import SupConLoss, CombinedSupConBCELoss, MultiScaleSupConLoss

# LLM integration components
from .ollama_client import OllamaClient, build_prediction_prompt, validate_llm_response
from .evidence_builder import build_evidence_pack, compute_query_stats, compute_final_probability

__version__ = "1.0.0"
__all__ = [
    # Core TimeMixer++ components
    "TimeMixerPPConfig",
    "TimeMixerPPEncoder",
    "TimeMixerPPForBinaryCls",
    "MixerBlock",
    "MRTI",
    "TID",
    "MCM",
    "MRM",
    # Metric learning components
    "TemporalConvEmbedder",
    "MultiScaleEmbedder",
    "AttentionPooling",
    "SupConLoss",
    "CombinedSupConBCELoss",
    "MultiScaleSupConLoss",
    # LLM integration
    "OllamaClient",
    "build_prediction_prompt",
    "validate_llm_response",
    "build_evidence_pack",
    "compute_query_stats",
    "compute_final_probability",
]

