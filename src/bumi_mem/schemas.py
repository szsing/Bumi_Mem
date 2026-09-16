"""Core data models for Bumi_Mem."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class MessageRecord:
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
    def attributed_content(role: str, content: str, user_id: str, agent_id: str) -> str:
        speaker = {"agent_id": agent_id, "speaker_role": role}
        if role == 'user':
            speaker['user_id'] = user_id
        header = json.dumps(speaker, ensure_ascii=False)
        return "[speaker " + header + "]\n" + content

    def to_shared_chat_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.attributed_content(
            self.role, self.content, self.user_id, self.agent_id)}

    def to_chat_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True, frozen=True)
class SessionRecord:
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
    session_id: str
    status: str
    effective_messages: int
    context_only_messages: int
    effective_tokens: int


@dataclass(slots=True)
class ConsolidationResult:
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
        counts: dict[str, int] = {}
        for event in self.events:
            name = str(event.get("event", "UNKNOWN"))
            counts[name] = counts.get(name, 0) + 1
        return counts


@dataclass(slots=True)
class TurnResult:
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
    """Base framework error."""


class SessionBlockedError(BumiMemoryError):
    """Raised when a failed or processing session must be resolved first."""


class ProviderConfigurationError(BumiMemoryError):
    """Raised for incomplete provider configuration."""


@dataclass(slots=True, frozen=True)
class ConsolidationSubmission:
    """A durable receipt, NOT a claim that L2/L3 has completed."""

    job_id: str
    session_id: str
    new_session_id: str
    status: str
    accepted: bool = True


@dataclass(slots=True, frozen=True)
class MemoryJob:
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
    """The job lost its lease, was cancelled, or predates a deletion."""


class MemoryReadUnavailable(BumiMemoryError):
    """Memory reads must not wait behind background SDK work."""


class RetryableMemoryError(BumiMemoryError):
    """The backend guarantees this failed attempt has not written memories."""


class MemoryWriteUncertainError(BumiMemoryError):
    """A write may have committed; blind replay needs explicit authorization."""


class IdentityRequiredError(BumiMemoryError):
    """Long-term memory is unavailable without an explicit user identity."""


class MemoryScopeError(BumiMemoryError):
    """Backend returned a record outside the requested identity scope."""


@dataclass(slots=True, frozen=True)
class SavedMemory:
    memory_id: str
    user_id: str
    agent_id: str
    content_sha256: str


@dataclass(slots=True, frozen=True)
class MemoryWriteReceipt:
    """Confirmed records, not merely a queued extraction task."""
    user_id: str
    agent_id: str
    records: tuple[SavedMemory, ...]
    confirmed_at: str
    memory_generation: int
    confirmed: bool = True
