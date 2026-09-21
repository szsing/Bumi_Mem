"""Engine operations for the custom Agent, without any Pipecat runtime coupling."""
from bumi_mem.agent_protocol import AgentPrepareRequest, AgentCompleteRequest, AgentEventRequest
from bumi_mem.agent_store import canonical
from copy import deepcopy
from bumi_mem.event_time import message_time
from bumi_mem.identity import GUEST_USER_ID, require_id
from bumi_mem.schemas import MessageRecord


class AgentEngineMixin:
    def prepare_agent(self, *, request: AgentPrepareRequest, agent_id: str) -> dict:
        if self._closed:
            raise RuntimeError('memory engine is closed')
        require_id(agent_id, 'agent_id')
        with self._pair_lock(request.memory_user_id, agent_id):
            turn = self.store.reserve_agent_turn(request=request, agent_id=agent_id,
                token_count=self.chat_token_counter.count_text(MessageRecord.attributed_content(
                    'user', request.query, request.memory_user_id, agent_id, request.occurred_at or message_time())) + 4,
                settings=self.consolidator._settings(), max_messages=self.config.max_messages,
                max_l1_tokens=self.config.max_l1_tokens)
            self.worker.wake()
            prepared = self._prompt_for_turn(turn)
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
                        recall_status=prepared['recall_status'], core_write=prepared['core_write'])

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
