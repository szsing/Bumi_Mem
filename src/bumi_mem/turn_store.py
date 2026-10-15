"""Persistent two-phase turns. All transcript/turn/rotation writes are transactional.

Mixin for SQLiteSessionStore; never holds a database transaction across model I/O.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from uuid import uuid4

from bumi_mem.schemas import BumiMemoryError


class TurnConflict(BumiMemoryError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class TurnNotFound(BumiMemoryError):
    pass


ACTIVE = ('preparing', 'awaiting_reply')


class TurnStoreMixin:
    def reserve_turn(self, *, user_id, agent_id, query, token_count, idempotency_key,
                     ttl_seconds, settings, max_messages, max_l1_tokens):
        with self._connect(write=True) as conn:
            return self._reserve_turn(conn, user_id=user_id, agent_id=agent_id, query=query,
                                      token_count=token_count, idempotency_key=idempotency_key,
                                      ttl_seconds=ttl_seconds, settings=settings,
                                      max_messages=max_messages, max_l1_tokens=max_l1_tokens)

    def _reserve_turn(self, conn, *, user_id, agent_id, query, token_count, idempotency_key,
                      ttl_seconds, settings, max_messages, max_l1_tokens, protocol='direct', occurred_at=None):
        now = time.time()
        conn.execute("UPDATE interaction_turns SET status='expired',prompt_json=NULL "
                     "WHERE agent_id=? AND protocol='direct' AND status IN ('preparing','awaiting_reply') AND expires_at<=?",
                     (agent_id, now))
        if idempotency_key is not None:
            row = conn.execute('SELECT * FROM interaction_turns WHERE agent_id=? AND idempotency_key=?',
                               (agent_id, idempotency_key)).fetchone()
            if row:
                if row['user_id'] != user_id or row['query'] != query:
                    raise TurnConflict('idempotency_conflict')
                if row['status'] not in ACTIVE:
                    raise TurnConflict('turn_already_closed')
                return dict(row)
        if conn.execute("SELECT 1 FROM interaction_turns WHERE agent_id=? "
                        "AND status IN ('preparing','awaiting_reply')", (agent_id,)).fetchone():
            raise TurnConflict('agent_busy')
        session = conn.execute("SELECT session_id FROM sessions WHERE user_id=? AND agent_id=? "
                               "AND status='active'", (user_id, agent_id)).fetchone()
        sid = session['session_id'] if session else self._insert_session(conn, user_id, agent_id)
        # Timeout/cancelled queries still count as real speech. Bound the window
        # even if a client repeatedly disappears without sending a reply.
        stats = conn.execute('SELECT COUNT(*) AS n, COALESCE(SUM(m.token_count),0) AS tokens '
                             'FROM shared_l1 w JOIN messages m USING(message_id) '
                             'WHERE w.agent_id=? AND w.context_only=0', (agent_id,)).fetchone()
        graph_token_boundary = False
        if settings.get('backend') == 'graphiti':
            pending_tokens = conn.execute(
                "SELECT COALESCE(SUM(m.token_count),0) FROM messages m JOIN sessions s USING(session_id) "
                "WHERE s.agent_id=? AND s.status='active' AND m.context_only=0", (agent_id,)).fetchone()[0]
            graph_token_boundary = pending_tokens >= settings['extraction_tokens']
        if graph_token_boundary or (settings.get('backend') != 'graphiti' and (stats['n'] >= max_messages or stats['tokens'] >= max_l1_tokens)):
            receipt = self._submit_shared(conn, sid, reason='unfinished_turn_boundary', settings=settings,
                                          carryover_messages=settings['carryover_messages'],
                                          max_attempts=settings['max_attempts'])
            sid = receipt.new_session_id
        source = self._append_message(conn, sid, role='user', content=query, token_count=token_count, occurred_at=occurred_at)
        scope = self._scope(conn, user_id, agent_id)
        turn_id = 'turn_' + uuid4().hex
        conn.execute('INSERT INTO interaction_turns(turn_id,session_id,user_id,agent_id,query,'
                     'idempotency_key,status,expires_at,created_at,generation,protocol) '
                     "VALUES(?,?,?,?,?,?,'preparing',?,?,?,?)",
                     (turn_id, sid, user_id, agent_id, query, idempotency_key,
                      now + ttl_seconds, now, scope['generation'], protocol))
        conn.execute('UPDATE interaction_turns SET source_message_id=? WHERE turn_id=?', (source.message_id, turn_id))
        return dict(conn.execute('SELECT * FROM interaction_turns WHERE turn_id=?', (turn_id,)).fetchone())

    def save_turn_prompt(self, turn_id, *, messages, recall_status, recalled, core_write=None,
                         expected_epoch=None, prompt_text=None, context_payload=None):
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM interaction_turns WHERE turn_id=?', (turn_id,)).fetchone()
            if not row or row['status'] not in ACTIVE or (row['protocol'] == 'direct' and row['expires_at'] <= time.time()):
                raise TurnConflict('turn_not_active')
            scope = self._scope(conn, row['user_id'], row['agent_id'])
            if scope['generation'] != row['generation']:
                raise TurnConflict('memory_scope_changed')
            epoch = self._context_epoch(conn, row['agent_id'])
            if expected_epoch is not None and expected_epoch != epoch:
                raise TurnConflict('shared_memory_context_changed')
            response = {'turn_id': turn_id, 'user_id': row['user_id'], 'agent_id': row['agent_id'],
                        'messages': messages, 'recall_status': recall_status, 'expires_at': row['expires_at'],
                        'recalled': recalled, 'core_write': core_write or {'status': 'not_requested'}}
            if prompt_text is not None:
                response['prompt_text'] = prompt_text
            if context_payload is not None:
                response['context_payload'] = context_payload
            conn.execute("UPDATE interaction_turns SET status='awaiting_reply',prompt_json=?,context_epoch=? WHERE turn_id=?",
                         (json.dumps(response, ensure_ascii=False), epoch, turn_id))
            return response

    def finish_turn(self, *, turn_id, agent_id, reply, status, token_count, settings,
                    max_messages, max_l1_tokens):
        with self._connect(write=True) as conn:
            return self._finish_turn(conn, turn_id=turn_id, agent_id=agent_id, reply=reply,
                                     status=status, token_count=token_count, settings=settings,
                                     max_messages=max_messages, max_l1_tokens=max_l1_tokens)

    def _finish_turn(self, conn, *, turn_id, agent_id, reply, status, token_count, settings,
                     max_messages, max_l1_tokens, occurred_at=None, reply_metadata=None, notice_tokens=0):
        payload_data = {'status': status, 'reply': reply}
        if occurred_at is not None:
            payload_data['occurred_at'] = occurred_at
        if reply_metadata is not None:
            payload_data['speech'] = reply_metadata
        payload = json.dumps(payload_data, ensure_ascii=False, sort_keys=True)
        row = conn.execute('SELECT * FROM interaction_turns WHERE turn_id=? AND agent_id=?',
                           (turn_id, agent_id)).fetchone()
        if not row:
            raise TurnNotFound('unknown turn')
        if row['finish_payload'] is not None:
            if row['finish_payload'] != payload:
                raise TurnConflict('reply_conflict')
            return json.loads(row['completion_json'])
        if row['status'] not in ACTIVE or (row['protocol'] == 'direct' and row['expires_at'] <= time.time()):
            raise TurnConflict('turn_not_active')
        if row['status'] == 'preparing' and (status in ('completed', 'interrupted') or reply):
            raise TurnConflict('prompt_not_ready')
        session = conn.execute('SELECT status FROM sessions WHERE session_id=?', (row['session_id'],)).fetchone()
        if session is None or session['status'] != 'active':
            raise TurnConflict('session_changed')
        if reply:
            message = self._append_message(conn, row['session_id'], role='assistant', content=reply, token_count=token_count, occurred_at=occurred_at)
            conn.execute('UPDATE interaction_turns SET reply_message_id=? WHERE turn_id=?', (message.message_id, turn_id))
            scope = self._scope(conn, row['user_id'], agent_id)
            if (scope['generation'] != row['generation'] or
                    row['context_epoch'] != self._context_epoch(conn, agent_id) or
                    (reply_metadata is not None and (not reply_metadata['alignment_covers_output'] or status != 'completed'))):
                # Preserve spoken agent text in shared L1 after forget-all, but
                # do not let a delayed answer resurrect the deleted user's L3.
                conn.execute('UPDATE messages SET memory_excluded=1 WHERE message_id=?', (message.message_id,))
        if reply_metadata is not None and not reply_metadata['alignment_covers_output']:
            # An empty aligned prefix is NOT evidence that the robot was silent.
            notice = self._append_message(conn, row['session_id'], role='system',
                content=self.speech_notice(reply_metadata, turn_id=turn_id), token_count=notice_tokens,
                context_only=True, occurred_at=occurred_at)
            conn.execute('UPDATE messages SET memory_excluded=1 WHERE message_id=?', (notice.message_id,))
        # Release the turn within this same transaction, prior to shared rotation.
        conn.execute('UPDATE interaction_turns SET status=? WHERE turn_id=?', (status, turn_id))
        stats = conn.execute('SELECT COUNT(*) AS n,COALESCE(SUM(m.token_count),0) AS tokens '
                             'FROM shared_l1 w JOIN messages m USING(message_id) '
                             'WHERE w.agent_id=? AND w.context_only=0', (agent_id,)).fetchone()
        if settings.get('backend') == 'graphiti':
            completed = conn.execute(
                "SELECT COUNT(*) FROM interaction_turns t JOIN sessions s USING(session_id) "
                "WHERE s.agent_id=? AND s.status='active' AND t.status='completed'", (agent_id,)).fetchone()[0]
            tokens = conn.execute(
                "SELECT COALESCE(SUM(m.token_count),0) FROM messages m JOIN sessions s USING(session_id) "
                "WHERE s.agent_id=? AND s.status='active' AND m.context_only=0", (agent_id,)).fetchone()[0]
            reason = ('extraction_turn_limit' if completed >= settings['extraction_turns'] else
                      'extraction_token_limit' if tokens >= settings['extraction_tokens'] else None)
        else:
            reason = ('message_limit' if stats['n'] >= max_messages else
                      'token_limit' if stats['tokens'] >= max_l1_tokens else None)
        receipt = None
        consolidation_status = 'not_triggered'
        if reason:
            old_ids = [s['session_id'] for s in conn.execute(
                "SELECT session_id FROM sessions WHERE agent_id=? AND status='active'", (agent_id,))]
            receipt = self._submit_shared(conn, row['session_id'], reason=reason, settings=settings,
                                          carryover_messages=settings['carryover_messages'],
                                          max_attempts=settings['max_attempts'])
            states = [conn.execute('SELECT status FROM memory_jobs WHERE session_id=?', (sid,)).fetchone()[0]
                      for sid in old_ids]
            consolidation_status = 'queued' if 'queued' in states else 'succeeded'
        result = {'turn_id': turn_id, 'status': status, 'reply_saved': bool(reply),
                  'consolidation_status': consolidation_status,
                  'submission': asdict(receipt) if receipt else None}
        conn.execute('UPDATE interaction_turns SET finish_payload=?,completion_json=?,prompt_json=NULL '
                     'WHERE turn_id=?', (payload, json.dumps(result, ensure_ascii=False), turn_id))
        conn.execute('UPDATE context_requests SET response_json=NULL WHERE turn_id=?', (turn_id,))
        return result

    def get_turn(self, turn_id, agent_id):
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM interaction_turns WHERE turn_id=? AND agent_id=?',
                               (turn_id, agent_id)).fetchone()
            if not row:
                raise TurnNotFound('unknown turn')
            return dict(row)

    @staticmethod
    def _assert_agent_idle(conn, agent_id):
        if conn.execute("SELECT 1 FROM interaction_turns WHERE agent_id=? "
                        "AND status IN ('preparing','awaiting_reply') AND (protocol!='direct' OR expires_at>?)",
                        (agent_id, time.time())).fetchone():
            raise TurnConflict('agent_busy')
