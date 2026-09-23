"""Single-call exchange: finish previous AI speech + reserve current query atomically."""
from __future__ import annotations

import json
import time
from uuid import uuid4

from bumi_mem.turn_store import ACTIVE, TurnConflict


class ContextStoreMixin:
    def _check_context_request(self, conn, row):
        if row['response_json'] and row['context_epoch'] != self._context_epoch(conn, row['agent_id']):
            raise TurnConflict('shared_memory_context_changed')
        if row['state'] == 'revoked':
            raise TurnConflict('memory_scope_changed')
        if row['turn_id']:
            turn = conn.execute('SELECT * FROM interaction_turns WHERE turn_id=?', (row['turn_id'],)).fetchone()
            if turn is None or turn['status'] not in ACTIVE:
                raise TurnConflict('context_request_closed')
            scope = conn.execute('SELECT generation FROM memory_scopes WHERE user_id=? AND agent_id=?',
                                 (row['user_id'], row['agent_id'])).fetchone()
            if scope is None or scope['generation'] != turn['generation']:
                raise TurnConflict('memory_scope_changed')

    def begin_context_request(self, *, user_id, agent_id, query, previous_ai_reply,
                              idempotency_key, query_tokens, reply_tokens, settings,
                              max_messages, max_l1_tokens, query_occurred_at=None, previous_reply_occurred_at=None):
        with self._connect(write=True) as conn:
            if idempotency_key is not None:
                existing = conn.execute('SELECT * FROM context_requests WHERE agent_id=? AND idempotency_key=?',
                                        (agent_id, idempotency_key)).fetchone()
                if existing:
                    if (existing['user_id'], existing['query'], existing['previous_ai_reply'], existing['query_occurred_at'], existing['previous_reply_occurred_at']) != (
                            user_id, query, previous_ai_reply, query_occurred_at, previous_reply_occurred_at):
                        raise TurnConflict('idempotency_conflict')
                    self._check_context_request(conn, existing)
                    return dict(existing)
            pending = conn.execute("SELECT * FROM interaction_turns WHERE agent_id=? "
                                   "AND status IN ('preparing','awaiting_reply')", (agent_id,)).fetchone()
            previous_turn_id = None
            previous_reply_saved = False
            consolidation_status = 'not_triggered'
            if pending:
                if pending['protocol'] != 'context':
                    raise TurnConflict('incompatible_pending_turn')
                if pending['status'] == 'preparing' and previous_ai_reply:
                    raise TurnConflict('previous_prompt_not_ready')
                previous_turn_id = pending['turn_id']
                result = self._finish_turn(
                    conn, turn_id=previous_turn_id, agent_id=agent_id,
                    reply=previous_ai_reply, status='completed' if previous_ai_reply else 'cancelled',
                    token_count=reply_tokens, settings=settings, max_messages=max_messages, occurred_at=previous_reply_occurred_at,
                    max_l1_tokens=max_l1_tokens)
                previous_reply_saved = result['reply_saved']
                consolidation_status = result['consolidation_status']
            elif previous_ai_reply:
                # There is no reliable original context for this reply. Do not
                # silently save it under the new user's extraction session.
                raise TurnConflict('no_previous_turn')
            request_id = 'ctx_' + uuid4().hex
            turn_id = None
            if query is not None:
                turn = self._reserve_turn(
                    conn, user_id=user_id, agent_id=agent_id, query=query, token_count=query_tokens,
                    idempotency_key=request_id, ttl_seconds=300, settings=settings,
                    max_messages=max_messages, max_l1_tokens=max_l1_tokens, protocol='context', occurred_at=query_occurred_at)
                turn_id = turn['turn_id']
            conn.execute('INSERT INTO context_requests(request_id,agent_id,user_id,query,previous_ai_reply,'
                         'idempotency_key,turn_id,previous_turn_id,previous_reply_saved,consolidation_status,created_at) '
                         'VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                         (request_id, agent_id, user_id, query, previous_ai_reply, idempotency_key,
                          turn_id, previous_turn_id, int(previous_reply_saved), consolidation_status, time.time()))
            conn.execute('UPDATE context_requests SET query_occurred_at=?,previous_reply_occurred_at=? WHERE request_id=?',
                         (query_occurred_at, previous_reply_occurred_at, request_id))
            return dict(conn.execute('SELECT * FROM context_requests WHERE request_id=?', (request_id,)).fetchone())

    def save_turn_context_response(self, request_id, *, turn_id, prompt, expected_epoch=None):
        """Persist the single-request protocol's two-field response for replay."""
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM context_requests WHERE request_id=?', (request_id,)).fetchone()
            self._check_context_request(conn, row)
            epoch = self._context_epoch(conn, row['agent_id'])
            if expected_epoch is not None and epoch != expected_epoch:
                raise TurnConflict('shared_memory_context_changed')
            result = {'turn_id': turn_id, 'prompt': prompt}
            conn.execute("UPDATE context_requests SET state='ready',response_json=?,context_epoch=? WHERE request_id=?",
                         (json.dumps(result, ensure_ascii=False), epoch, request_id))
            return result

    def save_context_response(self, request_id, *, messages, recall_status, core_write=None, expected_epoch=None):
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM context_requests WHERE request_id=?', (request_id,)).fetchone()
            self._check_context_request(conn, row)
            epoch = self._context_epoch(conn, row['agent_id'])
            if expected_epoch is not None and epoch != expected_epoch:
                raise TurnConflict('shared_memory_context_changed')
            result = {'request_id': request_id, 'user_id': row['user_id'], 'agent_id': row['agent_id'],
                      'messages': messages, 'has_prompt': row['turn_id'] is not None,
                      'recall_status': recall_status, 'previous_reply_saved': bool(row['previous_reply_saved']),
                      'consolidation_status': row['consolidation_status'],
                      'core_write': core_write or {'status': 'not_requested'}}
            conn.execute("UPDATE context_requests SET state='ready',response_json=?,context_epoch=? WHERE request_id=?",
                         (json.dumps(result, ensure_ascii=False), epoch, request_id))
            return result
