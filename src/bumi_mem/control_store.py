"""Durable control receipts and immutable idle snapshots (no model calls)."""
import json
import time
from datetime import datetime, timezone
from dataclasses import asdict
from uuid import uuid4


class ControlStoreMixin:
    def create_control_tables(self, conn):
        statements = [
            '''CREATE TABLE IF NOT EXISTS memory_control_requests (
                agent_id TEXT NOT NULL,request_key TEXT NOT NULL,user_id TEXT NOT NULL,
                signature TEXT NOT NULL,PRIMARY KEY(agent_id,request_key))''',
            '''CREATE TABLE IF NOT EXISTS memory_control_actors (agent_id TEXT PRIMARY KEY,user_id TEXT NOT NULL)''',
            '''CREATE TABLE IF NOT EXISTS memory_control_audit (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,operation_id TEXT NOT NULL,
                status TEXT NOT NULL,at REAL NOT NULL)''',
            '''CREATE TABLE IF NOT EXISTS memory_control_operations (
                operation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                request_key TEXT NOT NULL, action TEXT NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                UNIQUE(agent_id,request_key))''',
            '''CREATE TABLE IF NOT EXISTS memory_agent_activity (
                agent_id TEXT PRIMARY KEY, last_user_at REAL NOT NULL)''',
            '''CREATE TABLE IF NOT EXISTS memory_snapshot_sources (
                message_id TEXT PRIMARY KEY REFERENCES messages(message_id), job_id TEXT NOT NULL)''',
            '''CREATE TRIGGER IF NOT EXISTS memory_activity_on_user AFTER INSERT ON messages
                WHEN NEW.role='user' AND NEW.context_only=0 BEGIN
                INSERT INTO memory_agent_activity VALUES((SELECT agent_id FROM sessions WHERE session_id=NEW.session_id),
                    (julianday('now')-2440587.5)*86400.0)
                ON CONFLICT(agent_id) DO UPDATE SET last_user_at=excluded.last_user_at; END''',
        ]
        for statement in statements:
            conn.execute(statement)
        conn.execute('''INSERT OR IGNORE INTO memory_agent_activity(agent_id,last_user_at)
            SELECT s.agent_id,coalesce(max(t.created_at),min(?,max(cast(strftime('%s',m.created_at) AS REAL))))
            FROM sessions s JOIN messages m USING(session_id)
            LEFT JOIN interaction_turns t ON t.source_message_id=m.message_id
            WHERE s.status='active' AND m.role='user' AND m.context_only=0
            GROUP BY s.agent_id''', (time.time(),))

    def control_register_request(self, *, user_id, agent_id, request_key, signature):
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT user_id,signature FROM memory_control_requests WHERE agent_id=? AND request_key=?',
                               (agent_id,request_key)).fetchone()
            if row and (row['user_id']!=user_id or row['signature']!=signature):
                raise ValueError('request_conflict')
            conn.execute('INSERT OR IGNORE INTO memory_control_requests VALUES(?,?,?,?)',(agent_id,request_key,user_id,signature))
            actor = conn.execute('SELECT user_id FROM memory_control_actors WHERE agent_id=?',(agent_id,)).fetchone()
            if actor and actor['user_id']!=user_id:
                conn.execute("UPDATE memory_control_operations SET status='failed' WHERE agent_id=? AND status='needs_selection'",(agent_id,))
            conn.execute('INSERT INTO memory_control_actors VALUES(?,?) ON CONFLICT(agent_id) DO UPDATE SET user_id=excluded.user_id', (agent_id,user_id))

    def control_get(self, *, user_id, agent_id, request_key=None, operation_id=None):
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM memory_control_operations WHERE user_id=? AND agent_id=? AND '
                               + ('request_key=?' if request_key is not None else 'operation_id=?'),
                               (user_id, agent_id, request_key if request_key is not None else operation_id)).fetchone()
            return dict(row, payload=json.loads(row['payload'])) if row else None

    def control_save(self, *, user_id, agent_id, request_key, action, status, payload, operation_id=None):
        now = time.time()
        with self._connect(write=True) as conn:
            before = conn.execute('SELECT status FROM memory_control_operations WHERE agent_id=? AND request_key=?',(agent_id,request_key)).fetchone()
            conn.execute('''INSERT INTO memory_control_operations VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(agent_id,request_key) DO UPDATE SET
                status=excluded.status,payload=excluded.payload,updated_at=excluded.updated_at
                WHERE memory_control_operations.user_id=excluded.user_id''',
                (operation_id or uuid4().hex, user_id, agent_id, request_key, action, status,
                 json.dumps(payload, ensure_ascii=False), now, now))
            if before is None or before['status']!=status:
                conn.execute('INSERT INTO memory_control_audit(operation_id,status,at) SELECT operation_id,?,? FROM memory_control_operations WHERE agent_id=? AND request_key=?',(status,now,agent_id,request_key))
        return self.control_get(user_id=user_id, agent_id=agent_id, request_key=request_key)

    def control_related_sources(self, source_ids):
        related = set(source_ids)
        with self._connect() as conn:
            for mid in source_ids:
                related.update(r[0] for r in conn.execute(
                    'SELECT reply_message_id FROM interaction_turns WHERE source_message_id=? AND reply_message_id IS NOT NULL', (mid,)))
        return sorted(related)

    def control_exclude(self, *, user_id, agent_id, source_ids):
        """Do not cancel an entire mixed job when withdrawing some of its sources."""
        source_ids = self.control_related_sources(source_ids)
        with self._connect(write=True) as conn:
            for mid in source_ids:
                conn.execute('INSERT OR IGNORE INTO blocked_memory_sources(user_id,agent_id,source_message_id) VALUES(?,?,?)',
                             (user_id, agent_id, mid))
                conn.execute('UPDATE messages SET memory_excluded=1 WHERE message_id=? AND agent_id=? '
                             "AND (user_id=? OR role='assistant')", (mid, agent_id, user_id))
            conn.execute("UPDATE shared_memory_refs SET status='deleted' WHERE agent_id=? AND status='active'", (agent_id,))
            self._bump_context_epoch(conn, agent_id)

    def control_blocked(self, user_id, agent_id):
        with self._connect() as conn:
            return {r[0] for r in conn.execute('SELECT source_message_id FROM blocked_memory_sources '
                                              'WHERE user_id=? AND agent_id=?', (user_id, agent_id))}

    def control_source_status(self, source_id):
        with self._connect() as conn:
            source = conn.execute('SELECT memory_excluded FROM messages WHERE message_id=?', (source_id,)).fetchone()
            if source and source['memory_excluded']:
                return 'failed'
            jobs = conn.execute('SELECT j.status,j.last_error FROM memory_jobs j WHERE '
                                "EXISTS (SELECT 1 FROM json_each(j.payload_json,'$.messages') m "
                                "WHERE json_extract(m.value,'$.message_id')=?)", (source_id,)).fetchall()
            if any(r['status']=='succeeded' for r in jobs):
                return 'applied'
            if any(r['status']=='failed' for r in jobs):
                return 'uncertain' if any('unknown' in (r['last_error'] or '').lower() or 'uncertain' in (r['last_error'] or '').lower() for r in jobs) else 'failed'
            return 'queued'

    def submit_idle_snapshots(self, *, settings, idle_seconds=120, now=None):
        """Snapshot saved evidence without closing an active/pending turn's session."""
        now = time.time() if now is None else now
        count = 0
        with self._connect(write=True) as conn:
            active = conn.execute('''SELECT s.* FROM sessions s JOIN memory_agent_activity a USING(agent_id)
                WHERE s.status='active' AND a.last_user_at<=? AND s.user_id!='__guest__' ''', (now-idle_seconds,)).fetchall()
            for session in active:
                rows = conn.execute('''SELECT m.* FROM messages m WHERE m.session_id=?
                    AND m.context_only=0 AND m.memory_excluded=0 AND m.role IN ('user','assistant')
                    AND NOT EXISTS (SELECT 1 FROM memory_snapshot_sources c WHERE c.message_id=m.message_id)
                    ORDER BY m.sequence_no''', (session['session_id'],)).fetchall()
                if not rows:
                    continue
                sid, jid = 'idle_'+uuid4().hex, 'idle_'+uuid4().hex
                stamp = datetime.now(timezone.utc).isoformat()
                conn.execute('''INSERT INTO sessions(session_id,user_id,agent_id,status,started_at,created_at,
                    closed_at,close_reason,consolidation_status) VALUES(?,?,?,'processing',?,?,?,'idle_120s','queued')''',
                    (sid, session['user_id'], session['agent_id'], stamp, stamp, stamp))
                scope = self._scope(conn, session['user_id'], session['agent_id'])
                payload = json.dumps({'settings': settings, 'messages': [asdict(self._message(r)) for r in rows]}, ensure_ascii=False)
                conn.execute('''INSERT INTO memory_jobs(job_id,kind,session_id,new_session_id,user_id,agent_id,
                    generation,status,max_attempts,available_at,reason,payload_json,created_at,updated_at)
                    VALUES(?,'consolidate',?,?,?,?,?,'queued',?,?,'idle_120s',?,?,?)''',
                    (jid,sid,session['session_id'],session['user_id'],session['agent_id'],scope['generation'],
                     settings['max_attempts'],now,payload,stamp,stamp))
                conn.executemany('INSERT INTO memory_snapshot_sources VALUES(?,?)', [(r['message_id'],jid) for r in rows])
                count += 1
        return count
