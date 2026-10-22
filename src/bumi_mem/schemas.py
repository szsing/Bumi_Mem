"""Bumi_Mem 核心数据模型：消息记录、会话记录、整理结果、作业和错误类型。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from bumi_mem.temporal import normalize_relative_dates


def canonical_messages_digest(messages: Sequence[dict[str, Any]]) -> str:
    """Stable digest of one chunk's source messages.

    Single source of truth for replay authorization: the durable store binds a
    grant to this digest, and the backend re-derives it before re-entering an
    incomplete write, so a changed chunk can never inherit an old grant.
    """
    canonical = json.dumps([{k: m.get(k) for k in sorted(m)} for m in messages],
                           ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf8')).hexdigest()


@dataclass(slots=True, frozen=True)
class MessageRecord:
    """不可变消息记录：存储单条对话消息的完整信息。

    属性:
        message_id:        消息唯一标识
        session_id:        所属会话 ID
        sequence_no:       会话内序号
        role:              角色（'user'/'assistant'/'system'）
        content:           消息正文
        token_count:       token 数量
        context_only:      是否仅作为上下文（不参与抽取）
        source_session_id: 跨来源会话 ID（跨会话携带时非空）
        created_at:        创建时间（ISO 8601）
        consolidated_at:   整理时间（已整理时非空）
        user_id:           用户 ID（仅 user 角色有意义）
        agent_id:          Agent ID
    """
    message_id: str
    session_id: str
    sequence_no: int
    role: str
    content: str
    token_count: int
    context_only: bool = False
    source_session_id: str | None = None
    created_at: str = ""
    consolidated_at: str | None = None
    user_id: str = ""
    agent_id: str = ""

    @staticmethod
    def attributed_content(role: str, content: str, user_id: str, agent_id: str,
                           created_at: str | None = None) -> str:
        """生成带说话人属性头的内容：在正文前附加 [speaker {...}] 头。

        参数:
            role:       角色
            content:    消息正文
            user_id:    用户 ID
            agent_id:   Agent ID
            created_at: 创建时间（可选）

        返回:
            带说话人头的完整内容字符串。
        """
        speaker = {"agent_id": agent_id, "speaker_role": role}
        if created_at is not None:
            speaker["created_at"] = created_at
        if role == 'user':
            # 局部导入：identity 的验证错误定义在本模块中，避免循环导入
            from bumi_mem.identity import is_guest

            speaker['user_id'] = user_id
            if is_guest(user_id):
                speaker['identity_status'] = 'unconfirmed'
        header = json.dumps(speaker, ensure_ascii=False)
        return "[speaker " + header + "]\n" + content

    def to_shared_chat_message(self) -> dict[str, str]:
        """转换为带说话人属性的聊天消息格式（用于共享上下文）。"""
        return {"role": self.role, "content": self.attributed_content(
            self.role, self.content, self.user_id, self.agent_id, self.created_at)}

    def to_extraction_message(self) -> dict[str, str]:
        """转换为抽取消息格式：带来源头和时间归一化，供 LLM 记忆抽取使用。

        返回:
            含 role、content、observed_at、source_message_id 的字典。
            content 前附加 [source {...}] 头，包含来源消息 ID、观察时间等。
        """
        normalized = normalize_relative_dates(self.content, self.created_at)
        source = {"source_message_id": self.message_id, "observed_at": self.created_at,
                  "speaker_role": self.role, "agent_id": self.agent_id,
                  "original_content": normalized.original_text}
        if self.role == "user":
            from bumi_mem.identity import is_guest

            source["user_id"] = self.user_id
            if is_guest(self.user_id):
                source["identity_status"] = "unconfirmed"
        return {"role": self.role,
                "content": "[source " + json.dumps(source, ensure_ascii=False) + "]\n" + normalized.text,
                "observed_at": self.created_at, "source_message_id": self.message_id}

    def to_chat_message(self) -> dict[str, str]:
        """转换为简单聊天消息格式（仅 role 和 content，无附加头）。"""
        return {"role": self.role, "content": self.content}


@dataclass(slots=True, frozen=True)
class SessionRecord:
    """不可变会话记录：存储会话的完整生命周期信息。

    属性:
        session_id:             会话唯一标识
        user_id:                用户 ID
        agent_id:               Agent ID
        status:                 状态（'active'/'processing'/'failed'/'consolidated'/'discarded'）
        started_at:             开始时间
        closed_at:              关闭时间（未关闭时为 None）
        close_reason:           关闭原因
        summary:                L2 摘要文本
        summary_provider:       摘要提供者
        summary_status:         摘要状态（'pending'/'success'/'fallback'）
        summary_error:          摘要错误
        consolidation_status:   整理状态（'pending'/'queued'/'running'/'retry_wait'/'failed'/'success'/'skipped'/'cancelled'）
        consolidation_attempts: 整理尝试次数
        last_error:             最近错误
    """
    session_id: str
    user_id: str
    agent_id: str
    status: str
    started_at: str
    closed_at: str | None = None
    close_reason: str | None = None
    summary: str | None = None
    summary_provider: str | None = None
    summary_status: str = "pending"
    summary_error: str | None = None
    consolidation_status: str = "pending"
    consolidation_attempts: int = 0
    last_error: str | None = None


@dataclass(slots=True, frozen=True)
class SessionStats:
    """会话统计信息：有效消息数、上下文消息数和 token 数。

    属性:
        session_id:            会话 ID
        status:                会话状态
        effective_messages:    有效消息数（排除 context_only）
        context_only_messages: 仅上下文消息数
        effective_tokens:      有效 token 数
    """
    session_id: str
    status: str
    effective_messages: int
    context_only_messages: int
    effective_tokens: int


@dataclass(slots=True)
class ConsolidationResult:
    """整理结果：记录整理是否成功、摘要、事件和错误信息。

    属性:
        success:        是否成功
        session_id:     源会话 ID
        new_session_id: 新会话 ID
        reason:         整理原因
        attempt:        尝试次数
        summary:        摘要文本
        summary_status: 摘要状态
        events:         事件列表（ADD/UPDATE/DELETE 等）
        error:          错误信息
    """
    success: bool
    session_id: str
    new_session_id: str | None = None
    reason: str = "manual_new"
    attempt: int = 0
    summary: str | None = None
    summary_status: str = "pending"
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def event_counts(self) -> dict[str, int]:
        """统计各类型事件的数量。"""
        counts: dict[str, int] = {}
        for event in self.events:
            name = str(event.get("event", "UNKNOWN"))
            counts[name] = counts.get(name, 0) + 1
        return counts


@dataclass(slots=True)
class TurnResult:
    """对话轮次结果：包含回答、检索到的长期记忆和自动整理提交回执。

    属性:
        session_id:         当前会话 ID
        active_session_id:  活跃会话 ID
        answer:             回答文本
        recalled:           检索到的长期记忆列表
        prompt_messages:    发送给 LLM 的提示消息列表
        auto_consolidation: 自动整理提交回执（未触发时为 None）
        recall_status:      检索状态（'ok'/'unavailable'/'deleting'/'disabled'/'not_requested'）
        user_id:            用户 ID
        agent_id:           Agent ID
    """
    session_id: str
    active_session_id: str
    answer: str
    recalled: list[dict[str, Any]] = field(default_factory=list)
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    auto_consolidation: ConsolidationSubmission | None = None
    recall_status: str = "ok"
    user_id: str = ""
    agent_id: str = ""


class BumiMemoryError(RuntimeError):
    """框架基础错误。"""


class SessionBlockedError(BumiMemoryError):
    """会话阻塞错误：失败的或正在处理的会话必须先解决。"""


class ProviderConfigurationError(BumiMemoryError):
    """提供者配置错误：配置不完整。"""


@dataclass(slots=True, frozen=True)
class ConsolidationSubmission:
    """整理提交回执：持久化凭证，不保证 L2/L3 已完成。"""

    job_id: str
    session_id: str
    new_session_id: str
    status: str
    accepted: bool = True


@dataclass(slots=True, frozen=True)
class MemoryJob:
    """持久化记忆作业：后台整理或删除任务的完整状态。

    属性:
        job_id:          作业唯一 ID
        kind:            类型（'consolidate'/'delete'）
        session_id:      源会话 ID
        new_session_id:  新会话 ID
        user_id:         用户 ID
        agent_id:        Agent ID
        generation:      作用域代数
        status:          状态（'queued'/'running'/'retry_wait'/'succeeded'/'failed'/'cancelled'）
        attempts:        已尝试次数
        max_attempts:    最大尝试次数
        available_at:    可执行时间（Unix 时间戳）
        owner:           当前持有者
        lease_until:     租约到期时间
        reason:          原因
        payload_json:    载荷 JSON
        created_at:      创建时间
        updated_at:      更新时间
        last_error:      最近错误
    """
    job_id: str
    kind: str
    session_id: str | None
    new_session_id: str | None
    user_id: str
    agent_id: str
    generation: int
    status: str
    attempts: int
    max_attempts: int
    available_at: float
    owner: str | None
    lease_until: float | None
    reason: str
    payload_json: str
    created_at: str
    updated_at: str
    last_error: str | None = None


class JobCancelledError(BumiMemoryError):
    """作业已取消错误：租约丢失、被取消或早于删除操作。"""


class MemoryReadUnavailable(BumiMemoryError):
    """记忆读取不可用错误：读取不得等待后台 SDK 工作。"""


class RetryableMemoryError(BumiMemoryError):
    """可重试记忆错误：后端保证此失败尝试未写入记忆。"""


class MemoryWriteUncertainError(BumiMemoryError):
    """写入不确定错误：写入可能已提交，盲目重放需显式授权。"""


class IdentityRequiredError(BumiMemoryError):
    """身份必需错误：无显式用户身份时长期记忆不可用。"""


class MemoryScopeError(BumiMemoryError):
    """作用域错误：后端返回了请求身份作用域之外的记录。"""


@dataclass(slots=True, frozen=True)
class SavedMemory:
    """已保存的记忆记录：记忆 ID、归属和内容哈希。

    属性:
        memory_id:      记忆唯一 ID
        user_id:        用户 ID
        agent_id:       Agent ID
        content_sha256: 内容 SHA-256 哈希
    """
    memory_id: str
    user_id: str
    agent_id: str
    content_sha256: str


@dataclass(slots=True, frozen=True)
class MemoryWriteReceipt:
    """记忆写入回执：已确认的记录，而非仅排队的抽取任务。

    属性:
        user_id:            用户 ID
        agent_id:           Agent ID
        records:            已保存的记忆记录元组
        confirmed_at:       确认时间
        memory_generation:  记忆代数
        confirmed:          是否已确认
    """
    user_id: str
    agent_id: str
    records: tuple[SavedMemory, ...]
    confirmed_at: str
    memory_generation: int
    confirmed: bool = True