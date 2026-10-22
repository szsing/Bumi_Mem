"""Bumi_Mem 分层伴随记忆框架。

本包提供对话式 AI 的分层记忆管理：
- L1（短期）：当前对话窗口的即时消息
- L2（会话）：已完成会话的摘要和整理结果
- L3（长期）：跨会话的语义记忆，通过向量检索召回

公开导出：
- BumiMemory / BumiMemoryConfig：主引擎和配置
- MessageRecord / SessionRecord / MemoryJob：核心数据模型
- ConsolidationResult / ConsolidationSubmission：整理结果和提交
- TurnResult：单轮交互结果
"""

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

__version__ = "0.6.0"
