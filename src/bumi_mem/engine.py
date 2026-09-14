"""Non-blocking chat orchestration with durable background memory submission."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from bumi_mem.consolidator import SessionConsolidator
from bumi_mem.context import ContextBuilder
from bumi_mem.long_term import LongTermMemory, Mem0LongTermMemory
from bumi_mem.memory_access import SerialMemoryAccess
from bumi_mem.providers import TextProvider
from bumi_mem.schemas import ConsolidationSubmission, MemoryJob, MessageRecord, SessionRecord, SessionStats, TurnResult
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import TokenCounter
from bumi_mem.worker import MemoryWorker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BumiMemoryConfig:
    max_messages: int = 20
    max_l1_tokens: int = 8000
    top_k: int = 5
    search_threshold: float = 0.1
    max_answer_tokens: int = 512
    memory_read_timeout: float = 0.2


class BumiMemory:
    def __init__(self, *, store: SQLiteSessionStore, long_term: LongTermMemory,
                 chat_provider: TextProvider, consolidator: SessionConsolidator,
                 chat_token_counter: TokenCounter, config: BumiMemoryConfig | None = None,
                 auto_start: bool = True) -> None:
        if isinstance(long_term, Mem0LongTermMemory):
            raise ValueError('Construct native Mem0 through SerialMemoryAccess(factory=...); '
                             'an already-created SDK may be bound to its original thread')
        self.store = store
        self.config = config or BumiMemoryConfig()
        self.long_term = (long_term if isinstance(long_term, SerialMemoryAccess) else
                          SerialMemoryAccess(lambda: long_term, read_timeout=self.config.memory_read_timeout))
        self.chat_provider = chat_provider
        self.consolidator = consolidator
        self.consolidator.long_term = self.long_term
        self.chat_token_counter = chat_token_counter
        self.context_builder = ContextBuilder(max_messages=self.config.max_messages,
                                               max_l1_tokens=self.config.max_l1_tokens)
        self._locks_guard = threading.Lock()
        self._turn_locks: dict[tuple[str, str], threading.RLock] = {}
        self._closed = False
        self.worker = MemoryWorker(consolidator)
        self.consolidator.recover_legacy()
        if auto_start:
            self.start()

    def _pair_lock(self, user_id: str, agent_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._turn_locks.setdefault((user_id, agent_id), threading.RLock())

    def start(self) -> None:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        self.worker.start()

    def warmup(
        self,
        *,
        user_id: str,
        agent_id: str,
        query: str = '记忆系统预热',
        timeout: float = 120.0,
    ) -> tuple[float, int]:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        started = time.perf_counter()
        rows = self.long_term.warmup(
            timeout=timeout,
            user_id=user_id,
            agent_id=agent_id,
            query=query,
            top_k=1,
            threshold=self.config.search_threshold,
        )
        return (time.perf_counter() - started) * 1000, len(rows)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.worker.stop()
        self.long_term.close()

    def __enter__(self) -> BumiMemory:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get_or_create_session(self, *, user_id: str, agent_id: str) -> SessionRecord:
        return self.store.get_or_create_session(user_id, agent_id)

    def record_user_message(self, session_id: str, text: str) -> MessageRecord:
        return self.store.append_message(session_id, role='user', content=text,
                                         token_count=self.chat_token_counter.count_text(text) + 4)

    def record_assistant_message(self, session_id: str, text: str) -> MessageRecord:
        return self.store.append_message(session_id, role='assistant', content=text,
                                         token_count=self.chat_token_counter.count_text(text) + 4)

    def search_long_term(self, *, user_id: str, agent_id: str, query: str) -> list[dict[str, Any]]:
        before = self.store.read_scope(user_id, agent_id)
        if before[1]:
            return []
        rows = self.long_term.search(user_id=user_id, agent_id=agent_id, query=query,
                                     top_k=self.config.top_k, threshold=self.config.search_threshold)
        # Deletion may have been requested while the read was executing.
        return rows if self.store.read_scope(user_id, agent_id) == before else []

    def build_context(self, *, session_id: str, recalled: list[dict[str, Any]]) -> list[dict[str, str]]:
        return self.context_builder.build(recalled=recalled, l1_messages=self.store.list_messages(session_id))

    def chat(self, *, user_id: str, agent_id: str, text: str, recall_enabled: bool = True,
             on_answer: Callable[[str], None] | None = None) -> TurnResult:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        with self._pair_lock(user_id, agent_id):
            session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
            self.record_user_message(session.session_id, text)
            recalled: list[dict[str, Any]] = []
            recall_status = 'disabled'
            if recall_enabled:
                if self.store.reads_blocked(user_id, agent_id):
                    recall_status = 'deleting'
                else:
                    try:
                        recalled = self.search_long_term(user_id=user_id, agent_id=agent_id, query=text)
                        recall_status = 'ok'
                    except Exception as exc:
                        # No fallback ever invents/reuses another user's memories.
                        logger.debug('L3 unavailable for this turn: %s', type(exc).__name__)
                        recall_status = 'unavailable'
            prompt = self.build_context(session_id=session.session_id, recalled=recalled)
            answer = self.chat_provider.generate(prompt, max_tokens=self.config.max_answer_tokens).strip()
            self.record_assistant_message(session.session_id, answer)
            if on_answer is not None:
                on_answer(answer)
            stats = self.store.stats(session.session_id)
            reason = ('message_limit' if stats.effective_messages >= self.config.max_messages else
                      'token_limit' if stats.effective_tokens >= self.config.max_l1_tokens else None)
            receipt = self.consolidator.submit(session.session_id, reason=reason) if reason else None
            if receipt:
                self.worker.wake()
            return TurnResult(session_id=session.session_id,
                              active_session_id=receipt.new_session_id if receipt else session.session_id,
                              answer=answer, recalled=recalled, prompt_messages=prompt,
                              auto_consolidation=receipt, recall_status=recall_status)

    def end_session(self, *, user_id: str, agent_id: str, reason: str = 'manual_new') -> ConsolidationSubmission:
        with self._pair_lock(user_id, agent_id):
            session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
            receipt = self.consolidator.submit(session.session_id, reason=reason)
            self.worker.wake()
            return receipt

    def discard_session(self, *, user_id: str, agent_id: str) -> SessionRecord:
        with self._pair_lock(user_id, agent_id):
            session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
            return self.store.discard_and_create(session.session_id)

    def session_stats(self, *, user_id: str, agent_id: str) -> SessionStats:
        session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
        return self.store.stats(session.session_id)

    def get_memories(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        before = self.store.read_scope(user_id, agent_id)
        if before[1]:
            return []
        rows = self.long_term.get_all(user_id=user_id, agent_id=agent_id)
        return rows if self.store.read_scope(user_id, agent_id) == before else []

    def forget_all(self, *, user_id: str, agent_id: str) -> MemoryJob:
        with self._pair_lock(user_id, agent_id):
            job = self.store.request_forget(user_id=user_id, agent_id=agent_id,
                                           max_attempts=self.consolidator.config.max_attempts)
            self.worker.wake()
            return job

    def memory_jobs(self, *, user_id: str, agent_id: str) -> list[MemoryJob]:
        return self.store.list_jobs(user_id=user_id, agent_id=agent_id)

    def retry_memory_job(self, job_id: str, *, user_id: str, agent_id: str,
                         allow_replay: bool = False) -> MemoryJob:
        job = self.store.retry_job(job_id, user_id=user_id, agent_id=agent_id,
                                  max_attempts=self.consolidator.config.max_attempts, allow_replay=allow_replay)
        self.worker.wake()
        return job
