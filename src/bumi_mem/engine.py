"""Non-blocking chat orchestration with durable background memory submission."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from bumi_mem.identity import require_pair, require_id, require_person, is_guest
from bumi_mem.semantic_state import memory_conflicts
from bumi_mem.event_time import message_time
from bumi_mem.temporal import normalize_relative_dates
from bumi_mem.agent_engine import AgentEngineMixin
from bumi_mem.turn_store import TurnConflict
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
    explicit_recall_timeout: float = 1.0
    memory_warmup_enabled: bool = True
    memory_warmup_timeout: float = 120.0
    memory_warmup_query: str = "记忆系统预热"


class BumiMemory(AgentEngineMixin):
    def __init__(self, *, store: SQLiteSessionStore, long_term: LongTermMemory,
                 chat_provider: TextProvider | None, consolidator: SessionConsolidator,
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
                                             MessageRecord.attributed_content('user', text, user_id, agent_id, message_time())) + 4)

    def record_assistant_message(self, session_id: str, text: str, *, user_id: str, agent_id: str) -> MessageRecord:
        self._owned_session(session_id, user_id, agent_id)
        return self.store.append_message(session_id, role='assistant', content=text,
                                         token_count=self.chat_token_counter.count_text(
                                             MessageRecord.attributed_content('assistant', text, user_id, agent_id, message_time())) + 4)

    def search_long_term(self, *, user_id: str, agent_id: str, query: str, read_timeout: float | None = None) -> list[dict[str, Any]]:
        require_pair(user_id, agent_id)
        if is_guest(user_id):
            return []
        before = self.store.read_scope(user_id, agent_id)
        if before[1]:
            return []
        search_args = dict(user_id=user_id, agent_id=agent_id, query=query,
                           top_k=self.config.top_k, threshold=self.config.search_threshold)
        rows = (self.long_term.search(**search_args) if read_timeout is None else
                self.long_term.search_with_timeout(timeout=read_timeout, **search_args))
        # Deletion may have been requested while the read was executing. Drop
        # stale rows before inspecting them so they can never enter user context.
        if self.store.read_scope(user_id, agent_id) != before:
            return []
        rows = self._scoped_results(rows, user_id, agent_id)
        rows = self.store.filter_memory_rows(rows, user_id=user_id, agent_id=agent_id)
        current = self.store.semantic_context(user_id=user_id, agent_id=agent_id)
        historical = bool(re.search(r'以前|过去|曾经|当时|去年|原来|\bpreviously\b|\bused to\b|\bin the past\b', query, re.I))
        result = []
        for row in rows:
            conflict = memory_conflicts(str(row.get('memory', '')), current)
            if conflict and not historical:
                continue
            result.append(dict(row, fact_status='historical_superseded') if conflict else row)
        return result

    def build_context(self, *, session_id: str, recalled: list[dict[str, Any]], user_id: str, agent_id: str,
                      memory_only: bool = False, core_rows=None, core_write=None, history_rows=None) -> list[dict[str, str]]:
        self._owned_session(session_id, user_id, agent_id)
        self._scoped_results(recalled, user_id, agent_id)
        l1 = self.store.list_shared_messages(agent_id)
        selected = self.context_builder.select_l1(l1)
        refs = self.store.shared_memory_refs(agent_id, selected_message_ids=[m.message_id for m in selected])
        return self.context_builder.build(recalled=recalled, l1_messages=l1,
                                          current_user_id=user_id, current_agent_id=agent_id, memory_only=memory_only,
                                          shared_refs=refs, core_rows=core_rows or [], core_write=core_write, history_rows=history_rows or [])

    def prepare_prompt(self, *, user_id: str, agent_id: str, query: str,
                       idempotency_key: str | None = None, ttl_seconds: float = 300.0,
                       recall_enabled: bool = True, memory_only: bool = True) -> dict[str, Any]:
        """Persist input and return context; never invoke the answering model."""
        if self._closed:
            raise RuntimeError('memory engine is closed')
        require_pair(user_id, agent_id)
        if not isinstance(query, str) or not query.strip() or len(query) > 16000:
            raise ValueError('query must contain 1..16000 characters')
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError('turn ttl must be positive and finite')
        if idempotency_key is not None:
            require_id(idempotency_key, 'Idempotency-Key')
        query = query.strip()
        with self._pair_lock(user_id, agent_id):
            turn = self.store.reserve_turn(
                user_id=user_id, agent_id=agent_id, query=query,
                token_count=self.chat_token_counter.count_text(
                    MessageRecord.attributed_content('user', query, user_id, agent_id, message_time())) + 4,
                idempotency_key=idempotency_key, ttl_seconds=ttl_seconds,
                settings=self.consolidator._settings(), max_messages=self.config.max_messages,
                max_l1_tokens=self.config.max_l1_tokens)
            self.worker.wake()  # a prior timed-out window may have rotated
            return self._prompt_for_turn(turn, recall_enabled=recall_enabled, memory_only=memory_only)

    def _prompt_for_turn(self, turn, *, recall_enabled=True, memory_only=True):
        user_id, agent_id, query = turn['user_id'], turn['agent_id'], turn['query']
        if self.store.read_scope(user_id, agent_id)[0] != turn['generation']:
            raise TurnConflict('memory_scope_changed')
        if turn['prompt_json']:
            if turn['context_epoch'] != self.store.context_epoch(agent_id):
                raise TurnConflict('shared_memory_context_changed')
            return json.loads(turn['prompt_json'])
        core_write = {'status': 'not_requested'}
        core_rows = []
        if not is_guest(user_id):
            try:
                source = next((m for m in self.store.list_messages(turn['session_id'])
                               if m.message_id == turn['source_message_id']), None)
                if source is None:
                    raise RuntimeError('original turn evidence is missing')
                core_write = self.store.apply_semantic_turn(
                    user_id=user_id, agent_id=agent_id, source_message_id=source.message_id,
                    text=source.content, observed_at=source.created_at, generation=turn['generation'])
                core_rows = self.store.semantic_context(user_id=user_id, agent_id=agent_id)
            except Exception as exc:
                logger.warning('semantic memory unavailable: %s', type(exc).__name__)
                core_write = {'status': 'failed', 'reason': 'persistence_not_confirmed'}
                # A rejected new key must not erase unrelated established boundaries.
                try:
                    core_rows = self.store.semantic_context(user_id=user_id, agent_id=agent_id)
                except Exception:
                    core_rows = []
        epoch = self.store.context_epoch(agent_id)
        if not is_guest(user_id):
            try:
                core_rows = self.store.semantic_context(user_id, agent_id)
            except Exception:
                core_rows = []
                core_write = {'status': 'failed', 'reason': 'core_read_unavailable'}
        history_rows = []
        if not is_guest(user_id) and re.search(r'以前|过去|曾经|当时|原来|\bpreviously\b|\bused to\b', query, re.I):
            try:
                history_rows = self.store.semantic_history(user_id, agent_id)
            except Exception:
                logger.warning('semantic history unavailable')
        recalled: list[dict[str, Any]] = []
        recall_status = 'disabled'
        if recall_enabled and not is_guest(user_id):
            if self.store.reads_blocked(user_id, agent_id):
                recall_status = 'deleting'
            else:
                try:
                    explicit_recall = bool(re.search(r'还记得|记不记得|回忆|上次.{0,20}(说|聊|提)|\bremember\b|\blast time\b', query, re.I))
                    recalled = self.search_long_term(user_id=user_id, agent_id=agent_id, query=query,
                        read_timeout=self.config.explicit_recall_timeout if explicit_recall else None)
                    recall_status = 'ok'
                except Exception as exc:
                    logger.debug('L3 unavailable: %s', type(exc).__name__)
                    recall_status = 'unavailable'
        if recalled:
            self.store.attach_memory_refs(user_id=user_id, agent_id=agent_id,
                                          anchor_message_id=turn['source_message_id'], rows=recalled,
                                          generation=turn['generation'],
                                          visible_message_ids={m.message_id for m in self.context_builder.select_l1(
                                              self.store.list_shared_messages(agent_id))})
        # Successful recall is now represented by owner-labelled shared references.
        messages = self.build_context(session_id=turn['session_id'], recalled=[],
                                      user_id=user_id, agent_id=agent_id, memory_only=memory_only,
                                      core_rows=core_rows, core_write=core_write, history_rows=history_rows)
        return self.store.save_turn_prompt(turn['turn_id'], messages=messages, recall_status=recall_status,
                                           recalled=recalled, core_write=core_write, expected_epoch=epoch)

    def process_context(self, *, user_id: str, agent_id: str, previous_ai_reply: str | None,
                        query: str | None, idempotency_key: str | None = None,
                        query_occurred_at: str | None = None, previous_reply_occurred_at: str | None = None) -> dict[str, Any]:
        """Single external call: append previous agent speech, then prepare current context."""
        if self._closed:
            raise RuntimeError('memory engine is closed')
        require_pair(user_id, agent_id)
        if query is not None and (not isinstance(query, str) or not query.strip() or len(query) > 16000):
            raise ValueError('query must be null or contain 1..16000 characters')
        if previous_ai_reply is not None and (not isinstance(previous_ai_reply, str) or len(previous_ai_reply) > 64000):
            raise ValueError('previous_ai_reply must be null or at most 64000 characters')
        if idempotency_key is not None:
            require_id(idempotency_key, 'Idempotency-Key')
        query = query.strip() if query is not None else None
        previous_ai_reply = (previous_ai_reply or '').strip()
        query_occurred_at = message_time(query_occurred_at) if query_occurred_at is not None else None
        previous_reply_occurred_at = message_time(previous_reply_occurred_at) if previous_reply_occurred_at is not None else None
        if query is None and query_occurred_at is not None:
            raise ValueError('query timestamp requires a query')
        if not previous_ai_reply and previous_reply_occurred_at is not None:
            raise ValueError('reply timestamp requires a nonempty reply')
        with self._pair_lock(user_id, agent_id):
            request = self.store.begin_context_request(
                user_id=user_id, agent_id=agent_id, query=query, previous_ai_reply=previous_ai_reply,
                query_occurred_at=query_occurred_at, previous_reply_occurred_at=previous_reply_occurred_at,
                idempotency_key=idempotency_key,
                query_tokens=self.chat_token_counter.count_text(
                    MessageRecord.attributed_content('user', query or '', user_id, agent_id, query_occurred_at or message_time())) + 4,
                reply_tokens=self.chat_token_counter.count_text(
                    MessageRecord.attributed_content('assistant', previous_ai_reply, '', agent_id, previous_reply_occurred_at or message_time())) + 4,
                settings=self.consolidator._settings(), max_messages=self.config.max_messages,
                max_l1_tokens=self.config.max_l1_tokens)
            self.worker.wake()
            if request['response_json']:
                return json.loads(request['response_json'])
            if request['previous_turn_id']:
                self._record_spoken_promise(request['previous_turn_id'], agent_id)
            messages, recall_status = [], 'not_requested'
            core_write = {'status': 'not_requested'}
            epoch = self.store.context_epoch(agent_id)
            if request['turn_id']:
                turn = self.store.get_turn(request['turn_id'], agent_id)
                prepared = self._prompt_for_turn(turn)
                messages, recall_status = prepared['messages'], prepared['recall_status']
                core_write = prepared['core_write']
                epoch = self.store.get_turn(request['turn_id'], agent_id)['context_epoch']
            return self.store.save_context_response(request['request_id'], messages=messages, recall_status=recall_status,
                                                     core_write=core_write, expected_epoch=epoch)


    def record_reply(self, *, turn_id: str, agent_id: str, reply: str | None = None,
                     status: str = 'completed') -> dict[str, Any]:
        """Persist agent speech in the preceding conversation, then rotate L1 atomically."""
        if self._closed:
            raise RuntimeError('memory engine is closed')
        require_id(turn_id, 'turn_id')
        require_id(agent_id, 'agent_id')
        if status not in ('completed', 'interrupted', 'cancelled', 'failed'):
            raise ValueError('invalid reply status')
        if reply is not None and (not isinstance(reply, str) or len(reply) > 64000):
            raise ValueError('reply must contain at most 64000 characters')
        reply = (reply or '').strip()
        if status == 'completed' and not reply:
            raise ValueError('completed turns require a reply')
        if status in ('cancelled', 'failed') and reply:
            raise ValueError('cancelled/failed turns must not contain a reply; use interrupted for partial output')
        turn = self.store.get_turn(turn_id, agent_id)
        with self._pair_lock(turn['user_id'], agent_id):
            result = self.store.finish_turn(
                turn_id=turn_id, agent_id=agent_id, reply=reply, status=status,
                token_count=self.chat_token_counter.count_text(
                    MessageRecord.attributed_content('assistant', reply, turn['user_id'], agent_id, message_time())) + 4,
                settings=self.consolidator._settings(), max_messages=self.config.max_messages,
                max_l1_tokens=self.config.max_l1_tokens)
            self.worker.wake()
            self._record_spoken_promise(turn['turn_id'], agent_id)
            return result

    def chat(self, *, user_id: str, agent_id: str, text: str, recall_enabled: bool = True,
             on_answer: Callable[[str], None] | None = None) -> TurnResult:
        """CLI adapter for the same prepare/reply flow used by external agents."""
        if self.chat_provider is None:
            raise RuntimeError('no answering model configured; use prepare_prompt and record_reply')
        with self._pair_lock(user_id, agent_id):
            prepared = self.prepare_prompt(user_id=user_id, agent_id=agent_id, query=text,
                                           recall_enabled=recall_enabled, memory_only=False)
            turn = self.store.get_turn(prepared['turn_id'], agent_id)
            try:
                answer = self.chat_provider.generate(prepared['messages'],
                                                      max_tokens=self.config.max_answer_tokens).strip()
                result = self.record_reply(turn_id=turn['turn_id'], agent_id=agent_id, reply=answer)
            except Exception:
                # Do not release an already persisted response or hide the original error.
                try:
                    self.record_reply(turn_id=turn['turn_id'], agent_id=agent_id, status='failed')
                except Exception:
                    pass
                raise
            if on_answer is not None:
                on_answer(answer)
            receipt = ConsolidationSubmission(**result['submission']) if result['submission'] else None
            return TurnResult(session_id=turn['session_id'],
                              active_session_id=receipt.new_session_id if receipt else turn['session_id'],
                              answer=answer, recalled=prepared.get('recalled', []),
                              prompt_messages=prepared['messages'], auto_consolidation=receipt,
                              recall_status=prepared['recall_status'], user_id=user_id, agent_id=agent_id)

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
        require_person(user_id)
        require_pair(user_id, agent_id)
        before = self.store.read_scope(user_id, agent_id)
        if before[1]:
            return []
        rows = self.long_term.get_all(user_id=user_id, agent_id=agent_id)
        if self.store.read_scope(user_id, agent_id) != before:
            return []
        return self.store.filter_memory_rows(self._scoped_results(rows, user_id, agent_id), user_id=user_id, agent_id=agent_id)

    def forget_all(self, *, user_id: str, agent_id: str) -> MemoryJob:
        require_person(user_id)
        with self._pair_lock(user_id, agent_id):
            job = self.store.request_forget(user_id=user_id, agent_id=agent_id,
                                           max_attempts=self.consolidator.config.max_attempts)
            self.worker.wake()
            return job

    def memory_jobs(self, *, user_id: str, agent_id: str) -> list[MemoryJob]:
        require_person(user_id)
        require_pair(user_id, agent_id)
        return self.store.list_jobs(user_id=user_id, agent_id=agent_id)

    def retry_memory_job(self, job_id: str, *, user_id: str, agent_id: str,
                         allow_replay: bool = False) -> MemoryJob:
        require_person(user_id)
        require_pair(user_id, agent_id)
        job = self.store.retry_job(job_id, user_id=user_id, agent_id=agent_id,
                                  max_attempts=self.consolidator.config.max_attempts, allow_replay=allow_replay)
        self.worker.wake()
        return job


    def remember(self, *, user_id: str, agent_id: str, contents: list[str], observed_at: str | None = None) -> MemoryWriteReceipt:
        """Confirm explicitly supplied long-term records only after scoped read-back.

        Ordinary queued consolidation remains pending until that job completes.
        This method does not use an LLM or change retrieval/model settings.
        """
        require_person(user_id)
        if isinstance(contents, (str, bytes)) or not contents or any(not isinstance(x, str) or not x.strip() for x in contents):
            raise ValueError('contents must contain nonempty memory strings')
        observed_at = message_time(observed_at)
        original_contents = list(contents)
        contents = [normalize_relative_dates(x.strip(), observed_at).text for x in contents]
        with self._pair_lock(user_id, agent_id):
            before = self.store.read_scope(user_id, agent_id)
            if before[1]:
                raise SessionBlockedError('deletion is pending; no new memory confirmation')
            rows = self.long_term.remember(user_id=user_id, agent_id=agent_id, contents=contents,
                                          metadata={'source':'explicit_memory','write_batch_id':uuid4().hex,'memory_generation':before[0],
                                                    'observed_at': observed_at, 'original_contents': original_contents})
            if self.store.read_scope(user_id, agent_id) != before or len(rows) != len(contents):
                raise MemoryWriteUncertainError('scope or write state changed before confirmation')
            if any(x.get('user_id') != user_id or x.get('agent_id') != agent_id for x in rows):
                raise MemoryWriteUncertainError('ownership verification failed')
            if len({x.get('id') for x in rows}) != len(rows) or any(not x.get('id') for x in rows) or Counter(x.get('memory') for x in rows) != Counter(x.strip() for x in contents):
                raise MemoryWriteUncertainError('content/ID verification failed')
            records = tuple(SavedMemory(x['id'], user_id, agent_id, hashlib.sha256(x['memory'].encode('utf8')).hexdigest()) for x in rows)
            return MemoryWriteReceipt(user_id, agent_id, records, datetime.now(timezone.utc).isoformat(), before[0])

    def read_memory(self, *, user_id: str, agent_id: str, memory_id: str) -> dict[str, Any] | None:
        require_person(user_id)
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
        if row is not None and not self.store.filter_memory_rows([row], user_id=user_id, agent_id=agent_id):
            return None
        return row

    def forget_memory(self, *, user_id: str, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Source-aware logical deletion. Use forget_all for backend physical erasure."""
        require_person(user_id)
        with self._pair_lock(user_id, agent_id):
            row = self.read_memory(user_id=user_id, agent_id=agent_id, memory_id=memory_id)
            if row is None:
                return {'status': 'not_found', 'memory_id': memory_id}
            return self.store.request_forget_memory(user_id=user_id, agent_id=agent_id, memory=row)

    def _record_spoken_promise(self, turn_id: str, agent_id: str) -> None:
        turn = self.store.get_turn(turn_id, agent_id)
        if is_guest(turn['user_id']) or not turn.get('reply_message_id'):
            return
        # Never promote a delayed response based on a revoked prompt to new evidence.
        with self.store._connect() as conn:
            source = conn.execute('SELECT * FROM messages WHERE message_id=? AND memory_excluded=0',
                                  (turn['reply_message_id'],)).fetchone()
        if source is None:
            return
        try:
            receipt = self.store.record_assistant_promise(
                user_id=turn['user_id'], agent_id=agent_id, source_message_id=source['message_id'],
                text=source['content'], observed_at=source['created_at'], generation=turn['generation'])
        except Exception as exc:
            logger.warning('spoken promise not confirmed: %s', type(exc).__name__)
