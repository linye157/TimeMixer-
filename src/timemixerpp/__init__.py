"""
TimeMixer++ Implementation for Binary Classification

A modular implementation of the TimeMixer++ architecture with:
- MRTI: Multi-Resolution Time Imaging
- TID: Time Image Decomposition
- MCM: Multi-Scale Mixing
- MRM: Multi-Resolution Mixing
"""

from .config import TimeMixerPPConfig
from .model import TimeMixerPPEncoder, TimeMixerPPForBinaryCls
from .block import MixerBlock
from .mrti import MRTI
from .tid import TID
from .mcm import MCM
from .mrm import MRM

__version__ = "1.0.0"
__all__ = [
    "TimeMixerPPConfig",
    "TimeMixerPPEncoder",
    "TimeMixerPPForBinaryCls",
    "MixerBlock",
    "MRTI",
    "TID",
    "MCM",
    "MRM",
]

