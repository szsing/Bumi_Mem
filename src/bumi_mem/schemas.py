"""Core data models for Bumi_Mem."""

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
        speaker = {"agent_id": agent_id, "speaker_role": role}
        if created_at is not None:
            speaker["created_at"] = created_at
        if role == 'user':
            # Local import: identity's validation errors are defined in this module.
            from bumi_mem.identity import is_guest

            speaker['user_id'] = user_id
            if is_guest(user_id):
                speaker['identity_status'] = 'unconfirmed'
        header = json.dumps(speaker, ensure_ascii=False)
        return "[speaker " + header + "]\n" + content

    def to_shared_chat_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.attributed_content(
            self.role, self.content, self.user_id, self.agent_id, self.created_at)}

    def to_extraction_message(self) -> dict[str, str]:
        """Render a derived view without changing the API's original message."""
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
