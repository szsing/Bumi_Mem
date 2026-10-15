"""Transactional explicit-turn and independent-event storage for the Agent bridge."""
from __future__ import annotations

import json
from bumi_mem.identity import GUEST_USER_ID
from bumi_mem.turn_store import ACTIVE, TurnConflict, TurnNotFound


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


class AgentStoreMixin:
    @staticmethod
    def create_agent_tables(conn):
        conn.execute('''CREATE TABLE IF NOT EXISTS agent_turn_requests (
            agent_id TEXT NOT NULL, client_turn_id TEXT NOT NULL,
            turn_id TEXT NOT NULL UNIQUE REFERENCES interaction_turns(turn_id),
            payload_json TEXT NOT NULL, PRIMARY KEY(agent_id,client_turn_id))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS agent_event_receipts (
            agent_id TEXT NOT NULL, event_id TEXT NOT NULL, payload_json TEXT NOT NULL,
            response_json TEXT NOT NULL, PRIMARY KEY(agent_id,event_id))''')

    def pending_agent_turn(self, agent_id):
        with self._connect() as conn:
            row = conn.execute("SELECT t.turn_id,r.client_turn_id,t.user_id,t.protocol,t.status "
                               "FROM interaction_turns t LEFT JOIN agent_turn_requests r ON r.turn_id=t.turn_id "
                               "WHERE t.agent_id=? AND t.status IN ('preparing','awaiting_reply')", (agent_id,)).fetchone()
            return dict(row) if row else None

    def reserve_agent_turn(self, *, request, agent_id, token_count, settings,
                           max_messages, max_l1_tokens):
        payload = canonical({'user_id': request.memory_user_id, 'query': request.query,
                             'occurred_at': request.occurred_at})
        with self._connect(write=True) as conn:
            existing = conn.execute('SELECT * FROM agent_turn_requests WHERE agent_id=? AND client_turn_id=?',
                                    (agent_id, request.client_turn_id)).fetchone()
            if existing:
                if existing['payload_json'] != payload:
                    raise TurnConflict('idempotency_conflict')
                turn = conn.execute('SELECT * FROM interaction_turns WHERE turn_id=?', (existing['turn_id'],)).fetchone()
                if turn['status'] not in ACTIVE:
                    raise TurnConflict('turn_already_closed')
                return dict(turn)
            # Unlike legacy exchange, never close another pending turn implicitly.
            turn = self._reserve_turn(conn, user_id=request.memory_user_id, agent_id=agent_id,
                                      query=request.query, token_count=token_count, idempotency_key=None,
                                      ttl_seconds=300, protocol='agent', occurred_at=request.occurred_at,
                                      settings=settings, max_messages=max_messages, max_l1_tokens=max_l1_tokens)
            conn.execute('INSERT INTO agent_turn_requests VALUES(?,?,?,?)',
                         (agent_id, request.client_turn_id, turn['turn_id'], payload))
            return turn

    def complete_agent_turn(self, *, request, agent_id, token_count, notice_tokens, settings,
                            max_messages, max_l1_tokens):
        with self._connect(write=True) as conn:
            turn = conn.execute('SELECT * FROM interaction_turns WHERE turn_id=? AND agent_id=?',
                                (request.turn_id, agent_id)).fetchone()
            if turn is None:
                raise TurnNotFound('unknown turn')
            if turn['protocol'] != 'agent':
                raise TurnConflict('incompatible_turn_protocol')
            result = self._finish_turn(
                conn, turn_id=request.turn_id, agent_id=agent_id, reply=request.reply.strip(),
                status=request.status, token_count=token_count, occurred_at=request.occurred_at,
                reply_metadata=request.speech.model_dump(), notice_tokens=notice_tokens,
                settings=settings, max_messages=max_messages, max_l1_tokens=max_l1_tokens)
            return dict(operation='complete', turn_id=request.turn_id, user_id=turn['user_id'],
                        agent_id=agent_id, status=result['status'], has_prompt=False,
                        reply_saved=result['reply_saved'],
                        transcript_complete=request.speech.alignment_covers_output,
                        consolidation_status=result['consolidation_status'])

    @staticmethod
    def speech_notice(speech, *, turn_id=None):
        return canonical({'event': 'speech_history_incomplete', 'turn_id': turn_id,
                          'confirmed_prefix_only': True, 'reason': speech['alignment_status'],
                          'delivery_basis': speech['delivery_basis'],
                          'output_samples': speech['output_samples'], 'sample_rate': speech['sample_rate'],
                          'note': 'Only the submitted prefix is known. Missing text is not evidence of no speech; do not invent the tail. DDS publication is not physical playback confirmation.'})

    def record_agent_event(self, *, request, agent_id, content, token_count, notice_tokens,
                           settings, max_messages, max_l1_tokens):
        payload = canonical(request.model_dump())
        with self._connect(write=True) as conn:
            previous = conn.execute('SELECT * FROM agent_event_receipts WHERE agent_id=? AND event_id=?',
                                    (agent_id, request.event_id)).fetchone()
            if previous:
                if previous['payload_json'] != payload:
                    raise TurnConflict('idempotency_conflict')
                return json.loads(previous['response_json'])
            # Program-owned speech/control is shared, not attributed as somebody's
            # personal user fact, and never has an extraction owner with L3 access.
            owner = conn.execute("SELECT session_id FROM sessions WHERE user_id=? AND agent_id=? AND status='active'",
                                 (GUEST_USER_ID, agent_id)).fetchone()
            sid = owner['session_id'] if owner else self._insert_session(conn, GUEST_USER_ID, agent_id)
            role = 'assistant' if request.kind == 'speech' and (request.text or '').strip() else 'system'
            message = self._append_message(conn, sid, role=role, content=content,
                                           token_count=token_count, occurred_at=request.occurred_at,
                                           context_only=(role == 'system'))
            conn.execute('UPDATE messages SET memory_excluded=1 WHERE message_id=?', (message.message_id,))
            if request.kind == 'speech' and role == 'assistant' and not request.speech.alignment_covers_output:
                notice = self._append_message(conn, sid, role='system',
                    content=self.speech_notice(request.speech.model_dump()), token_count=notice_tokens,
                    occurred_at=request.occurred_at, context_only=True)
                conn.execute('UPDATE messages SET memory_excluded=1 WHERE message_id=?', (notice.message_id,))
            stats = conn.execute('SELECT COUNT(*) n,COALESCE(SUM(m.token_count),0) tokens FROM shared_l1 w '
                                 'JOIN messages m USING(message_id) WHERE w.agent_id=? AND w.context_only=0', (agent_id,)).fetchone()
            busy = conn.execute("SELECT 1 FROM interaction_turns WHERE agent_id=? AND status IN ('preparing','awaiting_reply')", (agent_id,)).fetchone()
            if settings.get('backend') != 'graphiti' and not busy and (stats['n'] >= max_messages or stats['tokens'] >= max_l1_tokens):
                self._submit_shared(conn, sid, reason='agent_event_boundary', settings=settings,
                                    carryover_messages=settings['carryover_messages'], max_attempts=settings['max_attempts'])
            if request.kind == 'local_control':
                # A stopped/sleeping conversation must not replay an old prompt.
                self._bump_context_epoch(conn, agent_id)
            result = dict(operation='agent_event', event_id=request.event_id, agent_id=agent_id,
                          kind=request.kind, message_id=message.message_id, has_prompt=False, saved=True)
            conn.execute('INSERT INTO agent_event_receipts VALUES(?,?,?,?)',
                         (agent_id, request.event_id, payload, canonical(result)))
            return result
