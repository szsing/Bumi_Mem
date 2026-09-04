"""Core data models for Bumi_Mem."""

from __future__ import annotations

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
    auto_consolidation: ConsolidationResult | None = None


class BumiMemoryError(RuntimeError):
    """Base framework error."""


class SessionBlockedError(BumiMemoryError):
    """Raised when a failed or processing session must be resolved first."""


class ProviderConfigurationError(BumiMemoryError):
    """Raised for incomplete provider configuration."""
