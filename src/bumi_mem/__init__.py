"""Bumi_Mem layered companion memory framework."""

from bumi_mem.engine import BumiMemory, BumiMemoryConfig
from bumi_mem.schemas import ConsolidationResult, ConsolidationSubmission, MemoryJob, MessageRecord, SessionRecord, TurnResult

__all__ = [
    "BumiMemory",
    "BumiMemoryConfig",
    "ConsolidationResult",
    "ConsolidationSubmission",
    "MemoryJob",
    "MessageRecord",
    "SessionRecord",
    "TurnResult",
]

__version__ = "0.4.0"
