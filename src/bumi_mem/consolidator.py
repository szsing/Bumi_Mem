"""Session-boundary L1 -> L2/L3 consolidation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from bumi_mem.long_term import LongTermMemory
from bumi_mem.prompts import MEMORY_EXTRACTION_INSTRUCTIONS, summary_messages
from bumi_mem.providers import TextProvider
from bumi_mem.schemas import ConsolidationResult, MessageRecord
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import TokenCounter


@dataclass(slots=True)
class ConsolidatorConfig:
    carryover_messages: int = 4
    local_chunk_tokens: int = 3500
    summary_max_tokens: int = 512


class SessionConsolidator:
    def __init__(
        self,
        *,
        store: SQLiteSessionStore,
        long_term: LongTermMemory,
        summary_provider: TextProvider,
        memory_provider_name: str,
        memory_token_counter: TokenCounter,
        config: ConsolidatorConfig | None = None,
    ) -> None:
        self.store = store
        self.long_term = long_term
        self.summary_provider = summary_provider
        self.memory_provider_name = memory_provider_name
        self.memory_token_counter = memory_token_counter
        self.config = config or ConsolidatorConfig()

    @staticmethod
    def _chat_messages(messages: Sequence[MessageRecord]) -> list[dict[str, str]]:
        return [message.to_chat_message() for message in messages]

    def _local_chunks(self, messages: Sequence[MessageRecord]) -> list[list[MessageRecord]]:
        """Pack complete user/assistant rounds without splitting a message."""
        rounds: list[list[MessageRecord]] = []
        current: list[MessageRecord] = []
        for message in messages:
            if message.role == "user" and current:
                rounds.append(current)
                current = [message]
            else:
                current.append(message)
        if current:
            rounds.append(current)

        chunks: list[list[MessageRecord]] = []
        chunk: list[MessageRecord] = []
        chunk_tokens = 0
        for round_messages in rounds:
            round_tokens = sum(
                self.memory_token_counter.count_text(item.content) + 4
                for item in round_messages
            )
            if chunk and chunk_tokens + round_tokens > self.config.local_chunk_tokens:
                chunks.append(chunk)
                chunk = []
                chunk_tokens = 0
            chunk.extend(round_messages)
            chunk_tokens += round_tokens
        if chunk:
            chunks.append(chunk)
        return chunks or [[]]

    def consolidate(self, session_id: str, *, reason: str) -> ConsolidationResult:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(f"unknown session: {session_id}")
        messages = self.store.effective_messages(session_id)
        if not messages:
            new_session = self.store.rotate_empty_session(session_id, reason=reason)
            return ConsolidationResult(
                success=True,
                session_id=session_id,
                new_session_id=new_session.session_id,
                reason=reason,
                summary="Empty session",
                summary_status="fallback",
            )

        attempt = self.store.begin_consolidation(session_id)
        chat_messages = self._chat_messages(messages)
        summary: str | None = None
        summary_error: str | None = None
        try:
            summary = self.summary_provider.generate(
                summary_messages(chat_messages),
                max_tokens=self.config.summary_max_tokens,
                temperature=0.0,
            ).strip()
            if not summary:
                raise RuntimeError("summary provider returned an empty response")
            self.store.save_summary(
                session_id,
                summary=summary,
                provider=self.summary_provider.name,
                status="success",
            )
        except Exception as exc:
            summary_error = str(exc)
            self.store.save_summary(
                session_id,
                summary=None,
                provider=self.summary_provider.name,
                status="failed",
                error=summary_error,
            )

        if self.memory_provider_name == "local":
            chunks = self._local_chunks(messages)
        else:
            chunks = [list(messages)]
        run_ids = self.store.create_consolidation_runs(
            session_id,
            attempt=attempt,
            provider=self.memory_provider_name,
            chunk_count=len(chunks),
        )

        all_events: list[dict] = []
        try:
            for chunk_index, (run_id, chunk) in enumerate(zip(run_ids, chunks)):
                self.store.start_run(run_id)
                metadata = {
                    "source": "l1_session",
                    "session_id": session_id,
                    "consolidation_id": run_id,
                    "consolidation_attempt": attempt,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                }
                try:
                    events = self.long_term.add_session(
                        user_id=session.user_id,
                        agent_id=session.agent_id,
                        messages=self._chat_messages(chunk),
                        metadata=metadata,
                        prompt=MEMORY_EXTRACTION_INSTRUCTIONS,
                    )
                except Exception as exc:
                    self.store.finish_run(run_id, error=str(exc))
                    raise
                all_events.extend(events)
                self.store.finish_run(
                    run_id,
                    events_json=json.dumps(events, ensure_ascii=False, default=str),
                )
        except Exception as exc:
            error = str(exc)
            self.store.fail_consolidation(session_id, error)
            return ConsolidationResult(
                success=False,
                session_id=session_id,
                reason=reason,
                attempt=attempt,
                events=all_events,
                error=error,
                summary=None,
                summary_status="pending",
            )

        summary_status = "success"
        if summary_error is not None:
            user_count = sum(item.role == "user" for item in messages)
            assistant_count = sum(item.role == "assistant" for item in messages)
            summary = f"会话包含 {user_count} 条用户消息和 {assistant_count} 条机器人消息"
            summary_status = "fallback"
            self.store.save_summary(
                session_id,
                summary=summary,
                provider=self.summary_provider.name,
                status="fallback",
                error=summary_error,
            )

        carryover = list(messages[-self.config.carryover_messages :])
        new_session = self.store.complete_consolidation(
            session_id, reason=reason, carryover=carryover
        )
        return ConsolidationResult(
            success=True,
            session_id=session_id,
            new_session_id=new_session.session_id,
            reason=reason,
            attempt=attempt,
            summary=summary,
            summary_status=summary_status,
            events=all_events,
        )
