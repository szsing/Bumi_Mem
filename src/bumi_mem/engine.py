"""Non-blocking chat orchestration with durable background memory submission."""
from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from bumi_mem.identity import require_pair
from bumi_mem.schemas import MemoryWriteReceipt, SavedMemory, MemoryWriteUncertainError, SessionBlockedError, MemoryScopeError
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
    memory_warmup_enabled: bool = True
    memory_warmup_timeout: float = 120.0
    memory_warmup_query: str = "记忆系统预热"


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
        self._turn_locks: dict[str, threading.RLock] = {}
        self._warmup_lock = threading.Lock()
        self._warmup_elapsed_ms: float | None = None
        self._closed = False
        self.worker = MemoryWorker(consolidator)
        self.consolidator.recover_legacy()
        if auto_start:
            self.start()

    def _pair_lock(self, user_id: str, agent_id: str) -> threading.RLock:
        require_pair(user_id, agent_id)
        # Same-agent users share L1: keep each user/assistant turn and rotation
        # contiguous. Different agents can still run independently.
        with self._locks_guard:
            return self._turn_locks.setdefault(agent_id, threading.RLock())

    def start(self) -> float | None:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        elapsed_ms = None
        if self.config.memory_warmup_enabled:
            elapsed_ms = self.warmup(
                query=self.config.memory_warmup_query,
                timeout=self.config.memory_warmup_timeout,
            )
        self.worker.start()
        return elapsed_ms

    def warmup(
        self,
        *,
        query: str = "记忆系统预热",
        timeout: float = 120.0,
    ) -> float:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        with self._warmup_lock:
            if self._warmup_elapsed_ms is not None:
                return self._warmup_elapsed_ms
            started = time.perf_counter()
            self.long_term.warmup(timeout=timeout, query=query)
            self._warmup_elapsed_ms = (time.perf_counter() - started) * 1000
            return self._warmup_elapsed_ms

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
        require_pair(user_id, agent_id)
        return self.store.get_or_create_session(user_id, agent_id)

    def _owned_session(self, session_id, user_id, agent_id):
        require_pair(user_id, agent_id)
        session = self.store.get_session(session_id)
        if session is None or session.user_id != user_id or session.agent_id != agent_id:
            raise MemoryScopeError('session does not belong to requested user/agent')
        return session

    @staticmethod
    def _scoped_results(rows, user_id, agent_id):
        if any(not isinstance(r, dict) or r.get('user_id') != user_id or r.get('agent_id') != agent_id for r in rows):
            raise MemoryScopeError('unscoped or foreign memory cannot enter user context')
        return rows

    def record_user_message(self, session_id: str, text: str, *, user_id: str, agent_id: str) -> MessageRecord:
        self._owned_session(session_id, user_id, agent_id)
        return self.store.append_message(session_id, role='user', content=text,
                                         token_count=self.chat_token_counter.count_text(
                                             MessageRecord.attributed_content('user', text, user_id, agent_id)) + 4)

    def record_assistant_message(self, session_id: str, text: str, *, user_id: str, agent_id: str) -> MessageRecord:
        self._owned_session(session_id, user_id, agent_id)
        return self.store.append_message(session_id, role='assistant', content=text,
                                         token_count=self.chat_token_counter.count_text(
                                             MessageRecord.attributed_content('assistant', text, user_id, agent_id)) + 4)

    def search_long_term(self, *, user_id: str, agent_id: str, query: str) -> list[dict[str, Any]]:
        require_pair(user_id, agent_id)
        before = self.store.read_scope(user_id, agent_id)
        if before[1]:
            return []
        rows = self.long_term.search(user_id=user_id, agent_id=agent_id, query=query,
                                     top_k=self.config.top_k, threshold=self.config.search_threshold)
        # Deletion may have been requested while the read was executing. Drop
        # stale rows before inspecting them so they can never enter user context.
        if self.store.read_scope(user_id, agent_id) != before:
            return []
        return self._scoped_results(rows, user_id, agent_id)

    def build_context(self, *, session_id: str, recalled: list[dict[str, Any]], user_id: str, agent_id: str) -> list[dict[str, str]]:
        self._owned_session(session_id, user_id, agent_id)
        self._scoped_results(recalled, user_id, agent_id)
        return self.context_builder.build(recalled=recalled, l1_messages=self.store.list_shared_messages(agent_id),
                                          current_user_id=user_id, current_agent_id=agent_id)

    def chat(self, *, user_id: str, agent_id: str, text: str, recall_enabled: bool = True,
             on_answer: Callable[[str], None] | None = None) -> TurnResult:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        with self._pair_lock(user_id, agent_id):
            session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
            self.record_user_message(session.session_id, text, user_id=user_id, agent_id=agent_id)
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
            prompt = self.build_context(session_id=session.session_id, recalled=recalled, user_id=user_id, agent_id=agent_id)
            answer = self.chat_provider.generate(prompt, max_tokens=self.config.max_answer_tokens).strip()
            self.record_assistant_message(session.session_id, answer, user_id=user_id, agent_id=agent_id)
            if on_answer is not None:
                on_answer(answer)
            stats = self.store.shared_stats(agent_id, session_id=session.session_id)
            reason = ('message_limit' if stats.effective_messages >= self.config.max_messages else
                      'token_limit' if stats.effective_tokens >= self.config.max_l1_tokens else None)
            receipt = self.consolidator.submit(session.session_id, reason=reason) if reason else None
            if receipt:
                self.worker.wake()
            return TurnResult(session_id=session.session_id,
                              active_session_id=receipt.new_session_id if receipt else session.session_id,
                              answer=answer, recalled=recalled, prompt_messages=prompt,
                              auto_consolidation=receipt, recall_status=recall_status,
                              user_id=user_id, agent_id=agent_id)

    def end_session(self, *, user_id: str, agent_id: str, reason: str = 'manual_new') -> ConsolidationSubmission:
        with self._pair_lock(user_id, agent_id):
            session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
            receipt = self.consolidator.submit(session.session_id, reason=reason)
            self.worker.wake()
            return receipt

    def discard_session(self, *, user_id: str, agent_id: str) -> SessionRecord:
        with self._pair_lock(user_id, agent_id):
            session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
            return self.store.discard_shared(session.session_id)

    def session_stats(self, *, user_id: str, agent_id: str) -> SessionStats:
        session = self.get_or_create_session(user_id=user_id, agent_id=agent_id)
        return self.store.shared_stats(agent_id, session_id=session.session_id)

    def get_memories(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        require_pair(user_id, agent_id)
        before = self.store.read_scope(user_id, agent_id)
        if before[1]:
            return []
        rows = self.long_term.get_all(user_id=user_id, agent_id=agent_id)
        if self.store.read_scope(user_id, agent_id) != before:
            return []
        return self._scoped_results(rows, user_id, agent_id)

    def forget_all(self, *, user_id: str, agent_id: str) -> MemoryJob:
        with self._pair_lock(user_id, agent_id):
            job = self.store.request_forget(user_id=user_id, agent_id=agent_id,
                                           max_attempts=self.consolidator.config.max_attempts)
            self.worker.wake()
            return job

    def memory_jobs(self, *, user_id: str, agent_id: str) -> list[MemoryJob]:
        require_pair(user_id, agent_id)
        return self.store.list_jobs(user_id=user_id, agent_id=agent_id)

    def retry_memory_job(self, job_id: str, *, user_id: str, agent_id: str,
                         allow_replay: bool = False) -> MemoryJob:
        require_pair(user_id, agent_id)
        job = self.store.retry_job(job_id, user_id=user_id, agent_id=agent_id,
                                  max_attempts=self.consolidator.config.max_attempts, allow_replay=allow_replay)
        self.worker.wake()
        return job


    def remember(self, *, user_id: str, agent_id: str, contents: list[str]) -> MemoryWriteReceipt:
        """Confirm explicitly supplied long-term records only after scoped read-back.

        Ordinary queued consolidation remains pending until that job completes.
        This method does not use an LLM or change retrieval/model settings.
        """
        if isinstance(contents, (str, bytes)) or not contents or any(not isinstance(x, str) or not x.strip() for x in contents):
            raise ValueError('contents must contain nonempty memory strings')
        with self._pair_lock(user_id, agent_id):
            before = self.store.read_scope(user_id, agent_id)
            if before[1]:
                raise SessionBlockedError('deletion is pending; no new memory confirmation')
            rows = self.long_term.remember(user_id=user_id, agent_id=agent_id, contents=contents,
                                          metadata={'source':'explicit_memory','write_batch_id':uuid4().hex,'memory_generation':before[0]})
            if self.store.read_scope(user_id, agent_id) != before or len(rows) != len(contents):
                raise MemoryWriteUncertainError('scope or write state changed before confirmation')
            if any(x.get('user_id') != user_id or x.get('agent_id') != agent_id for x in rows):
                raise MemoryWriteUncertainError('ownership verification failed')
            if len({x.get('id') for x in rows}) != len(rows) or any(not x.get('id') for x in rows) or Counter(x.get('memory') for x in rows) != Counter(x.strip() for x in contents):
                raise MemoryWriteUncertainError('content/ID verification failed')
            records = tuple(SavedMemory(x['id'], user_id, agent_id, hashlib.sha256(x['memory'].encode('utf8')).hexdigest()) for x in rows)
            return MemoryWriteReceipt(user_id, agent_id, records, datetime.now(timezone.utc).isoformat(), before[0])

    def read_memory(self, *, user_id: str, agent_id: str, memory_id: str) -> dict[str, Any] | None:
        require_pair(user_id, agent_id)
        before = self.store.read_scope(user_id, agent_id)
        if before[1]:
            return None
        row = self.long_term.get_memory(user_id=user_id, agent_id=agent_id, memory_id=memory_id)
        if self.store.read_scope(user_id, agent_id) != before:
            return None
        if row is not None:
            self._scoped_results([row], user_id, agent_id)
            if row.get("id") != memory_id:
                raise MemoryScopeError("backend returned a different memory ID")
        return row
