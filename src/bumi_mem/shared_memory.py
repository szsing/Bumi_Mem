"""Bounded shared L3 references and source-aware read suppression.

References are DATA attached to a real user turn, never chat speech, never
consolidation input. All recipients may see them while the anchor is in L1.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from bumi_mem.identity import is_guest, require_pair, require_person
from bumi_mem.schemas import MemoryScopeError


def reference_key(row: dict[str, Any]) -> str:
    return str(row.get('id') or ('text:' + hashlib.sha256(
        str(row.get('memory', row.get('text', ''))).encode()).hexdigest()))


def source_ids(row: dict[str, Any]) -> set[str]:
    meta = row.get('metadata') or {}
    ids = meta.get('source_message_ids', [])
    if isinstance(ids, str):
        ids = [ids]
    return {x for x in ids if isinstance(x, str)}


class SharedMemoryStoreMixin:
    @staticmethod
    def create_shared_memory_tables(conn):
        statements = [
            '''CREATE TABLE IF NOT EXISTS shared_memory_refs (
                ref_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, user_id TEXT NOT NULL,
                anchor_message_id TEXT NOT NULL REFERENCES messages(message_id),
                memory_id TEXT NOT NULL, version_hash TEXT NOT NULL,
                generation INTEGER NOT NULL, payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL)''',
            'CREATE INDEX IF NOT EXISTS shared_refs_anchor ON shared_memory_refs(agent_id,anchor_message_id,status)',
            '''CREATE TABLE IF NOT EXISTS context_epochs (
                agent_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL DEFAULT 0)''',
            '''CREATE TABLE IF NOT EXISTS memory_tombstones (
                user_id TEXT NOT NULL, agent_id TEXT NOT NULL, memory_id TEXT NOT NULL,
                created_at TEXT NOT NULL, PRIMARY KEY(user_id,agent_id,memory_id))''',
            '''CREATE TABLE IF NOT EXISTS blocked_memory_sources (
                user_id TEXT NOT NULL, agent_id TEXT NOT NULL, source_message_id TEXT NOT NULL,
                PRIMARY KEY(user_id,agent_id,source_message_id))''',
        ]
        for sql in statements:
            conn.execute(sql)

    @staticmethod
    def _context_epoch(conn, agent_id):
        row = conn.execute('SELECT epoch FROM context_epochs WHERE agent_id=?', (agent_id,)).fetchone()
        return row['epoch'] if row else 0

    @staticmethod
    def _bump_context_epoch(conn, agent_id):
        conn.execute('INSERT INTO context_epochs(agent_id,epoch) VALUES(?,1) '
                     'ON CONFLICT(agent_id) DO UPDATE SET epoch=epoch+1', (agent_id,))

    def context_epoch(self, agent_id):
        with self._connect() as conn:
            return self._context_epoch(conn, agent_id)

    @staticmethod
    def _memory_allowed(conn, row, user_id, agent_id):
        if is_guest(user_id):
            return False
        if conn.execute('SELECT 1 FROM memory_tombstones WHERE user_id=? AND agent_id=? AND memory_id=?',
                        (user_id, agent_id, reference_key(row))).fetchone():
            return False
        ids = source_ids(row)
        if ids:
            blocked = {r[0] for r in conn.execute(
                'SELECT source_message_id FROM blocked_memory_sources WHERE user_id=? AND agent_id=?',
                (user_id, agent_id))}
            if ids & blocked:
                return False
        return True

    def filter_memory_rows(self, rows, *, user_id, agent_id):
        require_pair(user_id, agent_id)
        with self._connect() as conn:
            return [r for r in rows if self._memory_allowed(conn, r, user_id, agent_id)]

    def attach_memory_refs(self, *, user_id, agent_id, anchor_message_id, rows, generation,
                           visible_message_ids=None, replace_user_refs=False):
        require_person(user_id)
        require_pair(user_id, agent_id)
        if replace_user_refs and not rows:
            # Nothing to publish and nothing to replace: do not take a write lock
            # on the foreground read path just to confirm an empty change.
            with self._connect() as conn:
                stale = conn.execute("SELECT 1 FROM shared_memory_refs WHERE agent_id=? AND user_id=? "
                                     "AND status='active' LIMIT 1", (agent_id, user_id)).fetchone()
            if stale is None:
                return
        with self._connect(write=True) as conn:
            scope = self._scope(conn, user_id, agent_id)
            if scope['generation'] != generation or scope['delete_pending']:
                return
            anchor = conn.execute('SELECT user_id,agent_id,role FROM messages WHERE message_id=?',
                                  (anchor_message_id,)).fetchone()
            if not anchor or (anchor['user_id'], anchor['agent_id'], anchor['role']) != (user_id, agent_id, 'user'):
                raise MemoryScopeError('memory reference anchor does not belong to current user')
            # Defensive reclamation: L1 anchors alone define lifetime, no new TTL.
            conn.execute('DELETE FROM shared_memory_refs WHERE agent_id=? AND anchor_message_id NOT IN '
                         '(SELECT message_id FROM shared_l1 WHERE agent_id=?)', (agent_id, agent_id))
            if replace_user_refs:
                # Replace this user's previous recall refs with THIS turn's recall
                # inside the same transaction that publishes them. Retired memories
                # must not survive as fresh context, but a memory recalled again
                # keeps its existing row instead of churning status/version.
                kept = {reference_key(row) for row in rows}
                if kept:
                    placeholders = ','.join('?' for _ in kept)
                    conn.execute('UPDATE shared_memory_refs SET status=\'superseded\' '
                                 'WHERE agent_id=? AND user_id=? AND status=\'active\' '
                                 f'AND memory_id NOT IN ({placeholders})',
                                 (agent_id, user_id, *sorted(kept)))
                else:
                    conn.execute("UPDATE shared_memory_refs SET status='superseded' "
                                 "WHERE agent_id=? AND user_id=? AND status='active'", (agent_id, user_id))
            for row in rows:
                if (row.get('user_id'), row.get('agent_id')) != (user_id, agent_id):
                    raise MemoryScopeError('foreign memory cannot be published through another user gate')
                if not self._memory_allowed(conn, row, user_id, agent_id):
                    continue
                payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
                version_data = {k: v for k, v in row.items() if k not in {'score', 'distance', 'rank'}}
                version = hashlib.sha256(json.dumps(version_data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                mid = reference_key(row)
                existing = conn.execute('SELECT anchor_message_id FROM shared_memory_refs WHERE agent_id=? AND user_id=? '
                                        "AND memory_id=? AND version_hash=? AND status='active'",
                                        (agent_id, user_id, mid, version)).fetchone()
                if existing and (visible_message_ids is None or existing['anchor_message_id'] in visible_message_ids):
                    continue  # Do not renew a still-visible reference on prompt assembly.
                conn.execute("UPDATE shared_memory_refs SET status='superseded' "
                             "WHERE agent_id=? AND user_id=? AND memory_id=? AND status='active'",
                             (agent_id, user_id, mid))
                rid = hashlib.sha256((agent_id + '\0' + user_id + '\0' + mid + '\0' + version + '\0' + anchor_message_id).encode()).hexdigest()
                conn.execute('INSERT OR IGNORE INTO shared_memory_refs('
                             'ref_id,agent_id,user_id,anchor_message_id,memory_id,version_hash,generation,payload_json,created_at) '
                             'VALUES(?,?,?,?,?,?,?,?,?)',
                             (rid, agent_id, user_id, anchor_message_id, mid, version, generation, payload,
                              datetime.now(timezone.utc).isoformat()))

    def shared_memory_refs(self, agent_id, *, selected_message_ids):
        selected = set(selected_message_ids)
        if not selected:
            return []
        with self._connect() as conn:
            rows = conn.execute('SELECT r.* FROM shared_memory_refs r '
                                'JOIN shared_l1 w ON w.message_id=r.anchor_message_id '
                                'JOIN memory_scopes s ON s.user_id=r.user_id AND s.agent_id=r.agent_id '
                                "WHERE r.agent_id=? AND r.status='active' AND s.delete_pending=0 "
                                'AND r.generation=s.generation ORDER BY w.position,r.created_at', (agent_id,)).fetchall()
            return [dict(json.loads(r['payload_json']), ref_id=r['ref_id'],
                         anchor_message_id=r['anchor_message_id'], reference_status=r['status'])
                    for r in rows if r['anchor_message_id'] in selected and
                    self._memory_allowed(conn, json.loads(r['payload_json']), r['user_id'], agent_id)]

    def invalidate_conflicting_refs(self, *, user_id, agent_id, current_rows):
        from bumi_mem.semantic_state import memory_conflicts
        with self._connect(write=True) as conn:
            rows = conn.execute("SELECT ref_id,payload_json FROM shared_memory_refs WHERE user_id=? AND agent_id=? AND status='active'",
                                (user_id, agent_id)).fetchall()
            invalid = [r['ref_id'] for r in rows if memory_conflicts(
                str(json.loads(r['payload_json']).get('memory', '')), current_rows)]
            conn.executemany("UPDATE shared_memory_refs SET status='superseded' WHERE ref_id=?", [(x,) for x in invalid])
            # Core changes themselves change the meaning of a cached prompt.
            self._bump_context_epoch(conn, agent_id)

    def request_forget_memory(self, *, user_id, agent_id, memory):
        """Logical deletion + source suppression before any future read.

        Source-backed records conservatively suppress the whole extraction evidence
        set: an opaque LLM output cannot reliably be assigned to one source span.
        Backend record is retained as a tombstoned audit record (not physical erasure).
        """
        require_person(user_id)
        if (memory.get('user_id'), memory.get('agent_id')) != (user_id, agent_id):
            raise MemoryScopeError('cannot delete foreign memory')
        mid, ids = reference_key(memory), source_ids(memory)
        meta = memory.get('metadata') or {}
        if not ids and meta.get('source') != 'explicit_memory':
            raise ValueError('memory has no reliable source lineage; use forget_all for legacy records')
        with self._connect(write=True) as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute('INSERT OR IGNORE INTO memory_tombstones VALUES(?,?,?,?)', (user_id, agent_id, mid, now))
            conn.executemany('INSERT OR IGNORE INTO blocked_memory_sources VALUES(?,?,?)',
                             [(user_id, agent_id, x) for x in ids])
            conn.executemany('INSERT OR IGNORE INTO semantic_withdrawals VALUES(?,?,?)',
                             [(user_id, agent_id, x) for x in ids])
            for source in ids:
                conn.execute('UPDATE messages SET memory_excluded=1 WHERE message_id=? AND session_id IN '
                             '(SELECT session_id FROM sessions WHERE user_id=? AND agent_id=?)', (source, user_id, agent_id))
            for job in conn.execute("SELECT * FROM memory_jobs WHERE user_id=? AND agent_id=? AND kind='consolidate' "
                                    "AND status IN ('queued','running','retry_wait','failed')", (user_id, agent_id)).fetchall():
                payload = json.loads(job['payload_json'])
                if any(m['message_id'] in ids for m in payload.get('messages', [])):
                    conn.execute("UPDATE memory_jobs SET status='cancelled',last_error='Source withdrawn',updated_at=? WHERE job_id=?", (now, job['job_id']))
            for ref in conn.execute("SELECT ref_id,payload_json FROM shared_memory_refs WHERE user_id=? AND agent_id=? AND status='active'", (user_id, agent_id)).fetchall():
                payload = json.loads(ref['payload_json'])
                if reference_key(payload) == mid or source_ids(payload) & ids:
                    conn.execute("UPDATE shared_memory_refs SET status='deleted' WHERE ref_id=?", (ref['ref_id'],))
            self._bump_context_epoch(conn, agent_id)
        return {'status': 'deleted', 'memory_id': mid, 'physical_erasure': False,
                'suppressed_source_count': len(ids)}
