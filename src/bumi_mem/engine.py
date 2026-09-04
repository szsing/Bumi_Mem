"""High-level Bumi_Mem L1/L2/L3 orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from bumi_mem.consolidator import SessionConsolidator
from bumi_mem.context import ContextBuilder
from bumi_mem.long_term import LongTermMemory
from bumi_mem.providers import TextProvider
from bumi_mem.schemas import (
    ConsolidationResult,
    MessageRecord,
    SessionBlockedError,
    SessionRecord,
    SessionStats,
    TurnResult,
)
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import TokenCounter


@dataclass(slots=True)
class BumiMemoryConfig:
    max_messages: int = 20
    max_l1_tokens: int = 8000
    top_k: int = 5
    search_threshold: float = 0.1
    max_answer_tokens: int = 512


class BumiMemory:
    def __init__(
        self,
        *,
        store: SQLiteSessionStore,
        long_term: LongTermMemory,
        chat_provider: TextProvider,
        consolidator: SessionConsolidator,
        chat_token_counter: TokenCounter,
        config: BumiMemoryConfig | None = None,
    ) -> None:
        self.store = store
        self.long_term = long_term
        self.chat_provider = chat_provider
        self.consolidator = consolidator
        self.chat_token_counter = chat_token_counter
        self.config = config or BumiMemoryConfig()
        self.context_builder = ContextBuilder(
            max_messages=self.config.max_messages,
            max_l1_tokens=self.config.max_l1_tokens,
        )

    def get_or_create_session(self, *, user_id: str, agent_id: str) -> SessionRecord:
        return self.store.get_or_create_session(user_id, agent_id)

    def _active_for_chat(self, user_id: str, agent_id: str) -> SessionRecord:
        session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
        if session.status != "active":
            raise SessionBlockedError(
                f"session {session.session_id} is {session.status}; use /new to retry or /discard"
            )
        return session

    def record_user_message(self, session_id: str, text: str) -> MessageRecord:
        return self.store.append_message(
            session_id,
            role="user",
            content=text,
            token_count=self.chat_token_counter.count_text(text) + 4,
        )

    def record_assistant_message(self, session_id: str, text: str) -> MessageRecord:
        return self.store.append_message(
            session_id,
            role="assistant",
            content=text,
            token_count=self.chat_token_counter.count_text(text) + 4,
        )

    def search_long_term(
        self, *, user_id: str, agent_id: str, query: str
    ) -> list[dict[str, Any]]:
        return self.long_term.search(
            user_id=user_id,
            agent_id=agent_id,
            query=query,
            top_k=self.config.top_k,
            threshold=self.config.search_threshold,
        )

    def build_context(
        self,
        *,
        session_id: str,
        recalled: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        return self.context_builder.build(
            recalled=recalled,
            l1_messages=self.store.list_messages(session_id),
        )

    def chat(
        self,
        *,
        user_id: str,
        agent_id: str,
        text: str,
        recall_enabled: bool = True,
        on_answer: Callable[[str], None] | None = None,
    ) -> TurnResult:
        session = self._active_for_chat(user_id, agent_id)
        self.record_user_message(session.session_id, text)
        recalled = (
            self.search_long_term(user_id=user_id, agent_id=agent_id, query=text)
            if recall_enabled
            else []
        )
        prompt_messages = self.build_context(
            session_id=session.session_id, recalled=recalled
        )
        answer = self.chat_provider.generate(
            prompt_messages, max_tokens=self.config.max_answer_tokens
        ).strip()
        self.record_assistant_message(session.session_id, answer)
        if on_answer is not None:
            on_answer(answer)

        stats = self.store.stats(session.session_id)
        reason: str | None = None
        if stats.effective_messages >= self.config.max_messages:
            reason = "message_limit"
        elif stats.effective_tokens >= self.config.max_l1_tokens:
            reason = "token_limit"

        auto: ConsolidationResult | None = None
        active_session_id = session.session_id
        if reason is not None:
            auto = self.consolidator.consolidate(session.session_id, reason=reason)
            if auto.success and auto.new_session_id:
                active_session_id = auto.new_session_id

        return TurnResult(
            session_id=session.session_id,
            active_session_id=active_session_id,
            answer=answer,
            recalled=recalled,
            prompt_messages=prompt_messages,
            auto_consolidation=auto,
        )

    def end_session(
        self, *, user_id: str, agent_id: str, reason: str = "manual_new"
    ) -> ConsolidationResult:
        session = self.store.get_open_session(user_id, agent_id)
        if session is None:
            session = self.store.create_session(user_id, agent_id)
        if session.status == "processing":
            raise SessionBlockedError("session consolidation is already processing")
        return self.consolidator.consolidate(session.session_id, reason=reason)

    def discard_session(self, *, user_id: str, agent_id: str) -> SessionRecord:
        session = self.store.get_open_session(user_id, agent_id)
        if session is None:
            return self.store.create_session(user_id, agent_id)
        return self.store.discard_and_create(session.session_id)

    def session_stats(self, *, user_id: str, agent_id: str) -> SessionStats:
        session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
        return self.store.stats(session.session_id)

    def get_memories(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        return self.long_term.get_all(user_id=user_id, agent_id=agent_id)

    def forget_all(self, *, user_id: str, agent_id: str) -> None:
        self.long_term.delete_all(user_id=user_id, agent_id=agent_id)
