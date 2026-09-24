"""Single-call exchange: finish previous AI speech + reserve current query atomically."""
from __future__ import annotations

import json
import time
from uuid import uuid4

from bumi_mem.turn_store import ACTIVE, TurnConflict
from bumi_mem.turn_prompt import TURN_REQUEST_PREFIX, LEGACY_TURN_REQUEST_PREFIX


def _json_object(value):
    try:
        decoded = json.loads(value) if value else None
    except (ValueError, TypeError):
        return None
    return decoded if isinstance(decoded, dict) else None


SINGLE_RESPONSE_KEYS_V7 = {'turn_id', 'prompt'}
SINGLE_RESPONSE_KEYS = {'turn_id', 'prompt', 'short_term', 'long_term', 'core_state'}


def _single_response(payload, client_id):
    """单轮协议的响应形态：v7 是两字段，当前是两字段 + 三个结构化记忆视图。"""
    if not isinstance(payload, dict) or payload.get('turn_id') != client_id:
        return False
    if not isinstance(payload.get('prompt'), str):
        return False
    keys = set(payload)
    if keys == SINGLE_RESPONSE_KEYS_V7:   # 旧缓存仍要在迁移/重放时认出来
        return True
    if keys != SINGLE_RESPONSE_KEYS:
        return False
    return all(isinstance(payload[key], list) for key in ('short_term', 'long_term', 'core_state'))


class ContextStoreMixin:
    def migrate_context_request_protocols(self, conn):
        """v8 migration; no network, transcript/L3 rewrite or guessed replay.

        v7 reused a printable prefix, and sometimes cleared every format marker
        after completion. Only provable single requests are rekeyed. Ambiguous
        old rows stay intact and explicitly block reuse until reviewed.
        """
        columns = {r['name'] for r in conn.execute('PRAGMA table_info(context_requests)')}
        if 'request_kind' not in columns:
            conn.execute("ALTER TABLE context_requests ADD COLUMN request_kind TEXT NOT NULL DEFAULT 'legacy'")
        if 'pre_v8_key' not in columns:
            conn.execute('ALTER TABLE context_requests ADD COLUMN pre_v8_key TEXT')
        conn.execute('CREATE INDEX IF NOT EXISTS context_requests_pre_v8_key ON context_requests(agent_id,pre_v8_key)')
        stale_agents = set()
        rows = conn.execute('SELECT r.*,t.prompt_json AS cached_turn_prompt FROM context_requests r '
                            'LEFT JOIN interaction_turns t ON t.turn_id=r.turn_id').fetchall()
        for row in rows:
            key = row['idempotency_key']
            if not isinstance(key, str):
                continue
            if key.startswith(TURN_REQUEST_PREFIX):
                # This representation could never be a valid external legacy ID.
                conn.execute("UPDATE context_requests SET request_kind='single' WHERE request_id=?", (row['request_id'],))
                continue
            if not key.startswith(LEGACY_TURN_REQUEST_PREFIX):
                continue
            # Keep the old name reserved even if the record moves. An older
            # crash/race may have lost the original caller mode; never use its
            # now-free printable key to silently create another input.
            conn.execute('UPDATE context_requests SET pre_v8_key=? WHERE request_id=?', (key, row['request_id']))
            client_id = key[len(LEGACY_TURN_REQUEST_PREFIX):]
            response = _json_object(row['response_json'])
            cached = _json_object(row['cached_turn_prompt'])
            proven_single = _single_response(response, client_id) or (
                response is None and cached is not None and isinstance(cached.get('prompt_text'), str))
            proven_legacy = (response is not None and response.get('request_id') == row['request_id']
                             and isinstance(response.get('messages'), list)) or (
                response is None and cached is not None and 'prompt_text' not in cached
                and isinstance(cached.get('messages'), list))
            if proven_single:
                new_key = TURN_REQUEST_PREFIX + client_id
                if conn.execute('SELECT 1 FROM context_requests WHERE agent_id=? AND idempotency_key=?',
                                (row['agent_id'], new_key)).fetchone():
                    raise RuntimeError('Conflicting persisted request namespaces; restore backup and audit before upgrading')
                conn.execute("UPDATE context_requests SET request_kind='single',idempotency_key=? WHERE request_id=?",
                             (new_key, row['request_id']))
                stale_agents.add(row['agent_id'])
            elif not proven_legacy:
                conn.execute("UPDATE context_requests SET request_kind='ambiguous' WHERE request_id=?", (row['request_id'],))
                stale_agents.add(row['agent_id'])
        # Never replay a pre-fix prompt that lost a failed write or misstated a
        # control result. Keep pending speech/source records for normal settlement.
        for agent_id in stale_agents:
            self._bump_context_epoch(conn, agent_id)

    @staticmethod
    def _check_response_kind(row):
        if row['request_kind'] == 'ambiguous':
            raise TurnConflict('request_protocol_ambiguous')
        if not row['response_json']:
            return
        payload = _json_object(row['response_json'])
        if row['request_kind'] == 'single':
            key = row['idempotency_key'] or ''
            if not key.startswith(TURN_REQUEST_PREFIX) or not _single_response(payload, key[len(TURN_REQUEST_PREFIX):]):
                raise TurnConflict('request_protocol_conflict')
        elif row['request_kind'] != 'legacy' or payload is None or (
                payload.get('request_id') != row['request_id'] or not isinstance(payload.get('messages'), list)):
            raise TurnConflict('request_protocol_conflict')

    def _check_context_request(self, conn, row):
        self._check_response_kind(row)
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
                              max_messages, max_l1_tokens, query_occurred_at=None, previous_reply_occurred_at=None,
                              request_kind='legacy'):
        if request_kind not in {'legacy', 'single'}:
            raise ValueError('invalid request protocol')
        if request_kind == 'single':
            if not isinstance(idempotency_key, str) or not idempotency_key:
                raise ValueError('single-turn requests require an ID')
            client_id = idempotency_key
            idempotency_key = TURN_REQUEST_PREFIX + client_id
        with self._connect(write=True) as conn:
            if idempotency_key is not None:
                existing = conn.execute('SELECT * FROM context_requests WHERE agent_id=? AND idempotency_key=?',
                                        (agent_id, idempotency_key)).fetchone()
                if existing:
                    if existing['request_kind'] == 'ambiguous':
                        raise TurnConflict('request_protocol_ambiguous')
                    if existing['request_kind'] != request_kind:
                        raise TurnConflict('request_protocol_conflict')
                    if (existing['user_id'], existing['query'], existing['previous_ai_reply'], existing['query_occurred_at'], existing['previous_reply_occurred_at']) != (
                            user_id, query, previous_ai_reply, query_occurred_at, previous_reply_occurred_at):
                        raise TurnConflict('idempotency_conflict')
                    self._check_context_request(conn, existing)
                    return dict(existing)
                old_key = LEGACY_TURN_REQUEST_PREFIX + client_id if request_kind == 'single' else idempotency_key
                if conn.execute('SELECT 1 FROM context_requests WHERE agent_id=? AND pre_v8_key=?',
                                (agent_id, old_key)).fetchone():
                    raise TurnConflict('request_protocol_ambiguous')
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
                         'idempotency_key,turn_id,previous_turn_id,previous_reply_saved,consolidation_status,created_at,request_kind) '
                         'VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                         (request_id, agent_id, user_id, query, previous_ai_reply, idempotency_key,
                          turn_id, previous_turn_id, int(previous_reply_saved), consolidation_status, time.time(), request_kind))
            conn.execute('UPDATE context_requests SET query_occurred_at=?,previous_reply_occurred_at=? WHERE request_id=?',
                         (query_occurred_at, previous_reply_occurred_at, request_id))
            return dict(conn.execute('SELECT * FROM context_requests WHERE request_id=?', (request_id,)).fetchone())

    def save_turn_context_response(self, request_id, *, turn_id, prompt, expected_epoch=None,
                                   structured=None):
        """Persist the single-request protocol's response for replay.

        structured = {'short_term': [...], 'long_term': [...], 'core_state': [...]}：
        结构化的短期/长期记忆视图；prompt 保留为同一内容的可读字符串形态。
        """
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM context_requests WHERE request_id=?', (request_id,)).fetchone()
            if row['request_kind'] != 'single' or row['idempotency_key'] != TURN_REQUEST_PREFIX + turn_id:
                raise TurnConflict('request_protocol_conflict')
            self._check_context_request(conn, row)
            epoch = self._context_epoch(conn, row['agent_id'])
            if expected_epoch is not None and epoch != expected_epoch:
                raise TurnConflict('shared_memory_context_changed')
            structured = structured or {}
            result = {'turn_id': turn_id, 'prompt': prompt,
                      'short_term': structured.get('short_term', []),
                      'long_term': structured.get('long_term', []),
                      'core_state': structured.get('core_state', [])}
            conn.execute("UPDATE context_requests SET state='ready',response_json=?,context_epoch=? WHERE request_id=?",
                         (json.dumps(result, ensure_ascii=False), epoch, request_id))
            return result

    def save_context_response(self, request_id, *, messages, recall_status, core_write=None, expected_epoch=None):
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM context_requests WHERE request_id=?', (request_id,)).fetchone()
            if row['request_kind'] != 'legacy':
                raise TurnConflict('request_protocol_conflict')
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
