"""Engine operations for the custom Agent, without any Pipecat runtime coupling."""
import logging
import time
from collections import deque

from bumi_mem.agent_protocol import (AgentPrepareRequest, AgentCompleteRequest,
                                     AgentEventRequest, AgentPrefetchRequest)
from bumi_mem.agent_store import canonical
from copy import deepcopy
from bumi_mem.event_time import message_time
from bumi_mem.identity import GUEST_USER_ID, is_guest, require_id
from bumi_mem.schemas import MessageRecord

logger = logging.getLogger(__name__)


class AgentEngineMixin:
    def prefetch_agent(self, *, request: AgentPrefetchRequest, agent_id: str) -> dict:
        """流式前缀预取：在 ASR 稳定前缀上提前跑检索。

        只返回候选与状态，**不写 L1、不建轮次、不发布 shared_memory_refs**；
        候选进入有界 TTL 缓存，prepare 命中才使用。
        """
        if self._closed:
            raise RuntimeError('memory engine is closed')
        require_id(agent_id, 'agent_id')
        user_id = request.memory_user_id
        self.worker.note_foreground()
        started = time.perf_counter()
        scope = self.store.read_scope(user_id, agent_id)
        rows: list[dict] = []
        if is_guest(user_id):
            # 陌生身份没有个人 L3，预取无意义：不占缓存。
            self.prefetch.stats.rejected_guest += 1
            return dict(operation='prefetch', prefetch_id=None, client_turn_id=request.client_turn_id,
                        user_id=user_id, agent_id=agent_id, recall_status='not_requested',
                        candidate_count=0, expires_in_ms=0,
                        took_ms=(time.perf_counter() - started) * 1000)
        if scope[1]:
            recall_status = 'deleting'
        else:
            try:
                rows = self.search_long_term(user_id=user_id, agent_id=agent_id,
                                             query=request.query_prefix)
                recall_status = 'ok'
            except Exception as exc:  # noqa: BLE001 - 预取失败不能让后续 prepare 也失败
                self.prefetch.record_error(type(exc).__name__)
                recall_status = 'unavailable'
        entry = self.prefetch.put(agent_id=agent_id, user_id=user_id,
                                  client_turn_id=request.client_turn_id,
                                  query_prefix=request.query_prefix, generation=scope[0],
                                  recall_status=recall_status, rows=rows)
        return dict(operation='prefetch', prefetch_id=entry.prefetch_id,
                    client_turn_id=request.client_turn_id, user_id=user_id, agent_id=agent_id,
                    recall_status=recall_status, candidate_count=len(rows),
                    expires_in_ms=int(self.prefetch.ttl_seconds * 1000),
                    took_ms=(time.perf_counter() - started) * 1000)

    def prefetch_stats(self) -> dict:
        serial = sorted(self._prefetch_serial_ms)

        def pct(ratio: float) -> float | None:
            if not serial:
                return None
            return serial[min(len(serial) - 1, max(0, int(round(ratio * (len(serial) - 1)))))]

        return {**self.prefetch.stats.as_dict(), 'pending': self.prefetch.pending(),
                'serial_samples': len(serial),
                'serial_p50_ms': pct(0.5), 'serial_p90_ms': pct(0.9),
                'serial_max_ms': (max(serial) if serial else None)}

    def _record_prefetch(self, outcome: str, serial_ms: float) -> None:
        self._prefetch_serial_ms.append(serial_ms)
        if outcome not in {'hit', 'none'}:
            # 未命中/陈旧/过期都要能从日志看出来，便于线上定位预取是否有效。
            logger.debug('prefetch reuse outcome=%s serial_ms=%.1f', outcome, serial_ms)

    def prepare_agent(self, *, request: AgentPrepareRequest, agent_id: str) -> dict:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        require_id(agent_id, 'agent_id')
        with self._pair_lock(request.memory_user_id, agent_id):
            prefetch_info: dict = {'status': 'none'}
            prefetched: list[dict] | None = None
            if request.prefetch_id:
                scope = self.store.read_scope(request.memory_user_id, agent_id)
                outcome, rows, age_ms = self.prefetch.take(
                    request.prefetch_id, agent_id=agent_id, user_id=request.memory_user_id,
                    client_turn_id=request.client_turn_id, query=request.query,
                    generation=scope[0])
                prefetch_info = {'status': outcome, 'age_ms': age_ms}
                if outcome == 'hit':
                    prefetched = rows
            serial_started = time.perf_counter()
            turn = self.store.reserve_agent_turn(request=request, agent_id=agent_id,
                token_count=self.chat_token_counter.count_text(MessageRecord.attributed_content(
                    'user', request.query, request.memory_user_id, agent_id, request.occurred_at or message_time())) + 4,
                settings=self.consolidator._settings(), max_messages=self.config.max_messages,
                max_l1_tokens=self.config.max_l1_tokens)
            self.worker.wake()
            prepared = self._prompt_for_turn(turn, prefetched=prefetched)
            self._record_prefetch(prefetch_info['status'],
                                  (time.perf_counter() - serial_started) * 1000)
            with self.store._connect() as conn:
                source_time = conn.execute('SELECT created_at FROM messages WHERE message_id=?', (turn['source_message_id'],)).fetchone()[0]
            supplement = deepcopy(prepared['messages'][:1])
            supplement[0]['content'] += (
                '\n[Agent-owned conversation] 对话正文由 Agent 管理，可能使用本地可信身份前缀而非 Bumi JSON 标头。'
                '本块只补充记忆，不替换 Agent 的人设、工具调用或本地控制记录。'
                '历史发言缺失身份或时间时不要猜测；记忆条目使用自己的原始时间依据。')
            return dict(operation='prepare', turn_id=turn['turn_id'], client_turn_id=request.client_turn_id,
                        user_id=turn['user_id'], agent_id=agent_id, messages=prepared['messages'],
                        source_message_id=turn['source_message_id'], occurred_at=source_time,
                        # The actual Agent owns tool/local-control history. This is
                        # only the memory-owned supplement, not a replacement history.
                        memory_context=supplement, has_prompt=True,
                        recall_status=prepared['recall_status'], core_write=prepared['core_write'],
                        **self.structured_context(user_id=turn['user_id'], agent_id=agent_id))

    def complete_agent(self, *, request: AgentCompleteRequest, agent_id: str) -> dict:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        turn = self.store.get_turn(request.turn_id, agent_id)
        with self._pair_lock(turn['user_id'], agent_id):
            notice = self.store.speech_notice(request.speech.model_dump(), turn_id=request.turn_id)
            result = self.store.complete_agent_turn(request=request, agent_id=agent_id,
                token_count=self.chat_token_counter.count_text(MessageRecord.attributed_content(
                    'assistant', request.reply, '', agent_id, request.occurred_at or message_time())) + 4,
                notice_tokens=self.chat_token_counter.count_text(notice) + 40,
                settings=self.consolidator._settings(), max_messages=self.config.max_messages,
                max_l1_tokens=self.config.max_l1_tokens)
            self.worker.wake()
            self._record_spoken_promise(request.turn_id, agent_id)
            return result

    def record_agent_event(self, *, request: AgentEventRequest, agent_id: str) -> dict:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        if request.kind == 'local_control':
            content = canonical(dict(event='local_control', **request.control.model_dump()))
        else:
            content = (request.text or '').strip() or self.store.speech_notice(request.speech.model_dump())
        with self._pair_lock(GUEST_USER_ID, agent_id):
            result = self.store.record_agent_event(request=request, agent_id=agent_id, content=content,
                token_count=self.chat_token_counter.count_text(content) + 40,
                notice_tokens=(self.chat_token_counter.count_text(self.store.speech_notice(request.speech.model_dump())) + 40
                               if request.speech is not None else 0),
                settings=self.consolidator._settings(), max_messages=self.config.max_messages,
                max_l1_tokens=self.config.max_l1_tokens)
            self.worker.wake()
            return result
