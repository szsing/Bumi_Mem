# 本次改动（2026-10-21）：对话入口归一身份，隔离长期引用，切换用户撤销缓存，检索只发布当前身份授权材料。
"""非阻塞对话编排 + 持久化后台记忆提交。"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, deque
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
from bumi_mem.long_term import LongTermMemory
from bumi_mem.memory_access import SerialMemoryAccess, _is_fail_fast, _cause_chain_names
from bumi_mem.prefetch import RetrievalPrefetchCache
from bumi_mem.providers import TextProvider
from bumi_mem.schemas import ConsolidationSubmission, MemoryJob, MessageRecord, SessionRecord, SessionStats, TurnResult
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import TokenCounter
from bumi_mem.turn_prompt import MAX_CLIENT_TURN_ID
from bumi_mem.turn_context import render_turn_context
from bumi_mem.worker import MemoryWorker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BumiMemoryConfig:
    registered_isolation: bool = False
    semantic_state_enabled: bool = True  # Legacy harness only; Graphiti service disables it.
    max_messages: int = 20
    max_l1_tokens: int = 8000
    top_k: int = 5
    search_threshold: float = 0.1
    max_answer_tokens: int = 512
    memory_read_timeout: float = 0.06
    explicit_recall_timeout: float = 0.2
    memory_warmup_enabled: bool = True
    memory_warmup_timeout: float = 120.0
    memory_warmup_query: str = "记忆系统预热"
    prefetch_ttl_seconds: float = 3.0
    prefetch_max_growth_chars: int = 12
    prefetch_max_entries: int = 256
    long_term_degrade_enabled: bool = True
    long_term_degrade_cooloff_seconds: float = 60.0
    long_term_degrade_probe_timeout: float = 5.0


class BumiMemory(AgentEngineMixin):
    """主记忆引擎：非阻塞对话编排 + 持久化后台记忆提交。

    核心职责：
    - 管理会话和 L1 窗口
    - 组装提示词（L3 检索 + 共享引用 + 核心状态 + L1 消息）
    - 提交会话整理到后台作业队列
    - 管理流式前缀预取缓存
    - 启动/停止后台工作线程
    """
    def __init__(self, *, store: SQLiteSessionStore, long_term: LongTermMemory,
                 chat_provider: TextProvider | None, consolidator: SessionConsolidator,
                 chat_token_counter: TokenCounter, config: BumiMemoryConfig | None = None,
                 auto_start: bool = True) -> None:
        if type(long_term).__name__ == 'Mem0LongTermMemory':
            raise ValueError('Construct native Mem0 through SerialMemoryAccess(factory=...); '
                             'an already-created SDK may be bound to its original thread')
        self.store = store
        self.config = config or BumiMemoryConfig()
        if self.config.registered_isolation:
            self.store.enable_isolation()
        self.long_term = (long_term if isinstance(long_term, SerialMemoryAccess) else
                          SerialMemoryAccess(lambda: long_term, read_timeout=self.config.memory_read_timeout,
                                            degrade_cooloff_seconds=self.config.long_term_degrade_cooloff_seconds,
                                            degrade_probe_timeout=self.config.long_term_degrade_probe_timeout))
        self.chat_provider = chat_provider
        self.consolidator = consolidator
        self.consolidator.long_term = self.long_term
        self.chat_token_counter = chat_token_counter
        self.context_builder = ContextBuilder(max_messages=self.config.max_messages,
                                               max_l1_tokens=self.config.max_l1_tokens)
        # 流式前缀预取缓存：只在内存里，随进程结束消失；不写 L1、不建轮次、不发引用。
        self.prefetch = RetrievalPrefetchCache(
            ttl_seconds=self.config.prefetch_ttl_seconds,
            max_growth_chars=self.config.prefetch_max_growth_chars,
            max_entries=self.config.prefetch_max_entries)
        self._prefetch_serial_ms: deque[float] = deque(maxlen=200)
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
        """获取 agent 级别的可重入锁：同 agent 用户共享 L1，保持轮次和轮转连续。"""
        require_pair(user_id, agent_id)
        # 同 agent 用户共享 L1：保持每个 user/assistant 轮次和轮转连续。
        # 不同 agent 可以独立运行。
        with self._locks_guard:
            return self._turn_locks.setdefault(agent_id, threading.RLock())

    @property
    def long_term_status(self) -> str:
        """长期记忆可用性三态：'unchecked'/'ready'/'degraded'（委托 SerialMemoryAccess）。"""
        return self.long_term.status

    @property
    def long_term_degraded(self) -> bool:
        return self.long_term.degraded

    def start(self) -> float | None:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        elapsed_ms = None
        if self.config.memory_warmup_enabled:
            try:
                elapsed_ms = self.warmup(
                    query=self.config.memory_warmup_query,
                    timeout=self.config.memory_warmup_timeout,
                )
            except Exception as exc:
                if _is_fail_fast(exc, lenient_unknown=True) or not self.config.long_term_degrade_enabled:
                    raise
                logger.warning('long-term memory warmup failed; starting in degraded mode '
                               '(short-term only): %s', _cause_chain_names(exc))
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
        user_id = self.store.resolve_registered_user(user_id, agent_id)
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
        user_id = self.store.resolve_registered_user(user_id, agent_id)
        if is_guest(user_id):
            return []
        before = self.store.read_scope(user_id, agent_id)
        if before[1]:
            return []
        search_args = dict(user_id=user_id, agent_id=agent_id, query=query,
                           top_k=self.config.top_k, threshold=self.config.search_threshold)
        if self.store.isolation_enabled():
            search_args.update(registered_isolation=True, registered_users=self.store.registered_users(agent_id))
        rows = (self.long_term.search(**search_args) if read_timeout is None else
                self.long_term.search_with_timeout(timeout=read_timeout, **search_args))
        return self._finalize_recalled(rows, user_id=user_id, agent_id=agent_id,
                                       query=query, expected_scope=before)

    def _finalize_recalled(self, rows, *, user_id: str, agent_id: str, query: str,
                           expected_scope) -> list[dict[str, Any]]:
        """检索结果的公共后处理：删除即失效、作用域校验、核心状态冲突过滤。

        预取复用走同一条路径——缓存里的候选也要重新校验，不能因为"提前算过"就跳过。
        """
        # 读取执行期间可能已请求删除。丢弃过期行，防止进入用户上下文。
        if self.store.read_scope(user_id, agent_id) != expected_scope:
            return []
        rows = self._scoped_results(rows, user_id, agent_id)
        rows = self.store.filter_memory_rows(rows, user_id=user_id, agent_id=agent_id)
        if not self.config.semantic_state_enabled:
            return rows
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
        refs = (recalled if self.store.isolation_enabled() else
                self.store.shared_memory_refs(agent_id, selected_message_ids=[m.message_id for m in selected]))
        return self.context_builder.build(recalled=recalled, l1_messages=l1,
                                          current_user_id=user_id, current_agent_id=agent_id, memory_only=memory_only,
                                          shared_refs=refs, core_rows=core_rows or [], core_write=core_write,
                                          history_rows=history_rows or [], current_recalled=recalled)

    def prepare_prompt(self, *, user_id: str, agent_id: str, query: str,
                       idempotency_key: str | None = None, ttl_seconds: float = 300.0,
                       recall_enabled: bool = True, memory_only: bool = True) -> dict[str, Any]:
        """Persist input and return context; never invoke the answering model."""
        if self._closed:
            raise RuntimeError('memory engine is closed')
        self.worker.note_foreground()
        require_pair(user_id, agent_id)
        user_id = self.store.resolve_registered_user(user_id, agent_id)
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
            if self.store.isolation_switch(user_id,agent_id):
                self.prefetch.clear()
                with self.store._connect(write=True) as conn:
                    conn.execute('UPDATE interaction_turns SET context_epoch=? WHERE turn_id=?',
                        (self.store._context_epoch(conn,agent_id),turn['turn_id']))
                turn = self.store.get_turn(turn['turn_id'],agent_id)
            self.worker.wake()  # a prior timed-out window may have rotated
            return self._prompt_for_turn(turn, recall_enabled=recall_enabled, memory_only=memory_only)

    def _cached_turn_prompt(self, turn):
        """Replay an already assembled prompt, or None when it must be built."""
        user_id, agent_id = turn['user_id'], turn['agent_id']
        if self.store.read_scope(user_id, agent_id)[0] != turn['generation']:
            raise TurnConflict('memory_scope_changed')
        if turn['prompt_json']:
            if turn['context_epoch'] != self.store.context_epoch(agent_id):
                raise TurnConflict('shared_memory_context_changed')
            return json.loads(turn['prompt_json'])
        return None

    def _context_bundle(self, turn, *, recall_enabled=True, prefetched=None):
        """Collect this turn's structured context inputs and apply its side effects.

        Returns the same rows both renderers consume; neither of them parses or
        re-splits an assembled prompt string, nor parses an already-rendered response.

        prefetched：来自 operation=prefetch 的候选；命中时跳过这次检索，只做后处理。
        """
        user_id, agent_id, query = turn['user_id'], turn['agent_id'], turn['query']
        core_write = {'status': 'not_requested'}
        core_rows = []
        if self.config.semantic_state_enabled and not is_guest(user_id):
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
                # 被拒绝的新键不得擦除无关的已建立边界
                try:
                    core_rows = self.store.semantic_context(user_id=user_id, agent_id=agent_id)
                except Exception:
                    core_rows = []
        epoch = self.store.context_epoch(agent_id)
        if self.config.semantic_state_enabled and not is_guest(user_id):
            try:
                core_rows = self.store.semantic_context(user_id, agent_id)
            except Exception:
                core_rows = []
                core_write = {'status': 'failed', 'reason': 'core_read_unavailable'}
        history_rows = []
        if self.config.semantic_state_enabled and not is_guest(user_id) and re.search(r'以前|过去|曾经|当时|原来|\bpreviously\b|\bused to\b', query, re.I):
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
                    if prefetched is not None and not self.store.isolation_enabled():
                        # 复用预取候选：仍走同一套删除/作用域/核心冲突校验，
                        # 但不再触发一次新的检索（这正是串行路径只剩拼装的原因）。
                        recalled = self._finalize_recalled(
                            prefetched, user_id=user_id, agent_id=agent_id, query=query,
                            expected_scope=self.store.read_scope(user_id, agent_id))
                    else:
                        # 记得.{0,6}吗 专门补上"你记得吗"这类不含"还/不"的疑问句——
                        # 原来只认"还记得"/"记不记得"，漏掉了这个常见措辞。
                        explicit_recall = bool(re.search(
                            r'还记得|记不记得|记得.{0,6}吗|回忆|上次.{0,20}(说|聊|提)|\bremember\b|\blast time\b',
                            query, re.I))
                        recalled = self.search_long_term(user_id=user_id, agent_id=agent_id, query=query,
                            read_timeout=self.config.explicit_recall_timeout if explicit_recall else None)
                    recall_status = 'ok'
                    if not recalled:
                        logger.debug('L3 search returned no results for this turn')
                except Exception as exc:
                    # 默认级别（不是 debug）：验收要求"发生几次降级，默认日志就能查到几条对应
                    # 记录"，不能要求运维开 DEBUG 才看得到长期记忆被跳过。超时/短路/真实后端
                    # 故障都走这里，每一轮都单独记一条，不只记第一次状态翻转的那一条。
                    logger.warning('long-term memory recall skipped this turn: %s', _cause_chain_names(exc))
                    recall_status = 'unavailable'
        # Recall references are refreshed as part of publishing this turn's
        # recall, in one write transaction. A separate DELETE pass would add a
        # second write transaction on the foreground read path for no gain.
        if not is_guest(user_id) and (recalled or not self.config.semantic_state_enabled):
            self.store.attach_memory_refs(user_id=user_id, agent_id=agent_id,
                                          anchor_message_id=turn['source_message_id'], rows=recalled,
                                          generation=turn['generation'],
                                          replace_user_refs=not self.config.semantic_state_enabled,
                                          visible_message_ids={m.message_id for m in self.context_builder.select_l1(
                                              self.store.list_shared_messages(agent_id))})
        # 成功检索现在由带所有者标签的共享引用表示
        self._owned_session(turn['session_id'], user_id, agent_id)
        l1 = self.store.list_shared_messages(agent_id)
        selected = self.context_builder.select_l1(l1)
        refs = (recalled if self.store.isolation_enabled() else
                self.store.shared_memory_refs(agent_id, selected_message_ids=[m.message_id for m in selected]))
        return dict(user_id=user_id, agent_id=agent_id, session_id=turn['session_id'], l1=l1,
                    selected=selected, refs=refs, core_rows=core_rows, core_write=core_write,
                    history_rows=history_rows, recalled=recalled, recall_status=recall_status, epoch=epoch)

    def _prompt_for_turn(self, turn, *, recall_enabled=True, memory_only=True, prefetched=None):
        cached = self._cached_turn_prompt(turn)
        if cached is not None:
            return cached
        bundle = self._context_bundle(turn, recall_enabled=recall_enabled, prefetched=prefetched)
        messages = self.context_builder.build(
            recalled=[], l1_messages=bundle['l1'], current_user_id=bundle['user_id'],
            current_agent_id=bundle['agent_id'], memory_only=memory_only, shared_refs=bundle['refs'],
            core_rows=bundle['core_rows'], core_write=bundle['core_write'],
            history_rows=bundle['history_rows'], current_recalled=bundle['recalled'])
        return self.store.save_turn_prompt(turn['turn_id'], messages=messages, recall_status=bundle['recall_status'],
                                           recalled=bundle['recalled'], core_write=bundle['core_write'],
                                           expected_epoch=bundle['epoch'])

    def _turn_context_payload(self, turn, *, client_turn_id):
        """Build the structured four-field response for one single-turn request."""
        cached = self._cached_turn_prompt(turn)
        if cached is not None and cached.get('context_payload') is not None:
            return cached['context_payload']
        bundle = self._context_bundle(turn)
        payload = render_turn_context(
            turn_id=client_turn_id, user_id=bundle['user_id'], selected=bundle['selected'],
            refs=bundle['refs'], core_rows=bundle['core_rows'], history_rows=bundle['history_rows'],
            max_reference_chars=self.context_builder.max_reference_chars,
            current_recalled=bundle['recalled'])
        self.store.save_turn_prompt(
            turn['turn_id'], messages=[], recall_status=bundle['recall_status'],
            recalled=bundle['recalled'], core_write=bundle['core_write'],
            expected_epoch=bundle['epoch'], context_payload=payload)
        return payload

    def process_context(self, *, user_id: str, agent_id: str, previous_ai_reply: str | None,
                        query: str | None, idempotency_key: str | None = None,
                        query_occurred_at: str | None = None, previous_reply_occurred_at: str | None = None) -> dict[str, Any]:
        """Single external call: append previous agent speech, then prepare current context."""
        if self._closed:
            raise RuntimeError('memory engine is closed')
        self.worker.note_foreground()
        require_pair(user_id, agent_id)
        user_id = self.store.resolve_registered_user(user_id, agent_id)
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
            if self.store.isolation_switch(user_id, agent_id):
                self.prefetch.clear()
                if request['turn_id']:
                    with self.store._connect(write=True) as conn:
                        conn.execute('UPDATE interaction_turns SET context_epoch=? WHERE turn_id=?',
                            (self.store._context_epoch(conn,agent_id),request['turn_id']))
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

    def process_turn_context(self, *, turn_id: str, user_id: str, agent_id: str,
                             query: str | None, previous_ai_reply: str | None = None,
                             query_occurred_at: str | None = None,
                             previous_reply_occurred_at: str | None = None, _control_hook=None) -> dict[str, Any]:
        """One request per turn: settle the previous reply, save the current query,
        and return four structured memory fields.

        turn_id is the caller's identifier for THIS request only. It provides
        per-request idempotency; it does not associate out-of-order replies.
        """
        if self._closed:
            raise RuntimeError('memory engine is closed')
        require_pair(user_id, agent_id)
        user_id = self.store.resolve_registered_user(user_id, agent_id)
        require_id(turn_id, 'turn_id')
        if len(turn_id) > MAX_CLIENT_TURN_ID:
            raise ValueError(f'turn_id must contain at most {MAX_CLIENT_TURN_ID} characters')
        if query is not None and (not isinstance(query, str) or not query.strip() or len(query) > 16000):
            raise ValueError('query must be null or contain 1..16000 nonblank characters')
        if previous_ai_reply is not None and (not isinstance(previous_ai_reply, str) or len(previous_ai_reply) > 64000):
            raise ValueError('previous_ai_reply must be null or at most 64000 characters')
        query = query.strip() if query is not None else None
        previous_ai_reply = (previous_ai_reply or '').strip()
        query_occurred_at = message_time(query_occurred_at) if query_occurred_at is not None else None
        previous_reply_occurred_at = (message_time(previous_reply_occurred_at)
                                      if previous_reply_occurred_at is not None else None)
        if query is None and query_occurred_at is not None:
            raise ValueError('query timestamp requires a query')
        if not previous_ai_reply and previous_reply_occurred_at is not None:
            raise ValueError('reply timestamp requires a nonempty reply')
        with self._pair_lock(user_id, agent_id):
            request = self.store.begin_context_request(
                user_id=user_id, agent_id=agent_id, query=query, previous_ai_reply=previous_ai_reply,
                query_occurred_at=query_occurred_at, previous_reply_occurred_at=previous_reply_occurred_at,
                idempotency_key=turn_id, request_kind='single',
                query_tokens=self.chat_token_counter.count_text(
                    MessageRecord.attributed_content('user', query or '', user_id, agent_id, query_occurred_at or message_time())) + 4,
                reply_tokens=self.chat_token_counter.count_text(
                    MessageRecord.attributed_content('assistant', previous_ai_reply, '', agent_id, previous_reply_occurred_at or message_time())) + 4,
                settings=self.consolidator._settings(), max_messages=self.config.max_messages,
                max_l1_tokens=self.config.max_l1_tokens)
            if self.store.isolation_switch(user_id, agent_id):
                self.prefetch.clear()
                if request['turn_id']:
                    with self.store._connect(write=True) as conn:
                        conn.execute('UPDATE interaction_turns SET context_epoch=? WHERE turn_id=?',
                            (self.store._context_epoch(conn,agent_id),request['turn_id']))
            if _control_hook is not None:
                _control_hook(request)
            self.worker.wake()
            if request['response_json']:
                return json.loads(request['response_json'])
            if request['previous_turn_id']:
                self._record_spoken_promise(request['previous_turn_id'], agent_id)
            # 仅结算请求：保存上一轮回复但不开始新轮次，无当前上下文数组可返回
            payload = {'turn_id': turn_id, 'current_user': user_id,
                       'long_term_memory': [], 'short_term_memory': []}
            epoch = self.store.context_epoch(agent_id)
            if request['turn_id']:
                internal_turn = self.store.get_turn(request['turn_id'], agent_id)
                payload = self._turn_context_payload(internal_turn, client_turn_id=turn_id)
                # 核心更新可能在构建期间提升共享上下文代数
                epoch = self.store.get_turn(request['turn_id'], agent_id)['context_epoch']
            return self.store.save_turn_context_response(
                request['request_id'], payload=payload, expected_epoch=epoch)

    def record_reply(self, *, turn_id: str, agent_id: str, reply: str | None = None,
                     status: str = 'completed') -> dict[str, Any]:
        """Persist agent speech in the preceding conversation, then rotate L1 atomically."""
        if self._closed:
            raise RuntimeError('memory engine is closed')
        self.worker.note_foreground()
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
        self.worker.note_foreground()
        with self._pair_lock(user_id, agent_id):
            prepared = self.prepare_prompt(user_id=user_id, agent_id=agent_id, query=text,
                                           recall_enabled=recall_enabled, memory_only=False)
            turn = self.store.get_turn(prepared['turn_id'], agent_id)
            try:
                answer = self.chat_provider.generate(prepared['messages'],
                                                      max_tokens=self.config.max_answer_tokens).strip()
                result = self.record_reply(turn_id=turn['turn_id'], agent_id=agent_id, reply=answer)
            except Exception:
                # 不得释放已持久化的响应或隐藏原始错误
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
        self.worker.note_foreground()
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
        require_person(self.store.resolve_registered_user(user_id,agent_id))
        require_pair(user_id, agent_id)
        before = self.store.read_scope(user_id, agent_id)
        if before[1]:
            return []
        rows = self.long_term.get_all(user_id=user_id, agent_id=agent_id)
        if self.store.read_scope(user_id, agent_id) != before:
            return []
        return self.store.filter_memory_rows(self._scoped_results(rows, user_id, agent_id), user_id=user_id, agent_id=agent_id)

    def forget_all(self, *, user_id: str, agent_id: str) -> MemoryJob:
        require_person(self.store.resolve_registered_user(user_id,agent_id))
        with self._pair_lock(user_id, agent_id):
            # 预取候选必须随删除立即失效（generation 校验是兜底，不是唯一防线）。
            self.prefetch.invalidate_scope(user_id=user_id, agent_id=agent_id)
            job = self.store.request_forget(user_id=user_id, agent_id=agent_id,
                                           max_attempts=self.consolidator.config.max_attempts)
            self.worker.wake()
            return job

    def memory_jobs(self, *, user_id: str, agent_id: str) -> list[MemoryJob]:
        require_person(self.store.resolve_registered_user(user_id,agent_id))
        require_pair(user_id, agent_id)
        return self.store.list_jobs(user_id=user_id, agent_id=agent_id)

    def retry_memory_job(self, job_id: str, *, user_id: str, agent_id: str,
                         allow_replay: bool = False) -> MemoryJob:
        require_person(self.store.resolve_registered_user(user_id,agent_id))
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
        if getattr(self.long_term, 'backend_name', '') == 'graphiti':
            raise NotImplementedError('Explicit remember is disabled for Graphiti')
        self.worker.note_foreground()
        require_person(self.store.resolve_registered_user(user_id,agent_id))
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
        require_person(self.store.resolve_registered_user(user_id,agent_id))
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
        require_person(self.store.resolve_registered_user(user_id,agent_id))
        with self._pair_lock(user_id, agent_id):
            self.prefetch.invalidate_scope(user_id=user_id, agent_id=agent_id)
            row = self.read_memory(user_id=user_id, agent_id=agent_id, memory_id=memory_id)
            if row is None:
                return {'status': 'not_found', 'memory_id': memory_id}
            return self.store.request_forget_memory(user_id=user_id, agent_id=agent_id, memory=row)

    def _record_spoken_promise(self, turn_id: str, agent_id: str) -> None:
        if not self.config.semantic_state_enabled:
            return
        turn = self.store.get_turn(turn_id, agent_id)
        if is_guest(turn['user_id']) or not turn.get('reply_message_id'):
            return
        # 永不将基于已撤销提示的延迟响应提升为新证据
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
