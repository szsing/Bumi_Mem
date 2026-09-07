"""SQLite L1/L2 storage and durable, leased background memory jobs.

External model/vector calls never run inside a SQLite transaction. A session
rotation and its job snapshot commit together. Only active sessions are unique
per user/agent; frozen sessions can coexist with the new conversation.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4

from bumi_mem.schemas import (
    ConsolidationSubmission, JobCancelledError, MemoryJob, MessageRecord,
    SessionBlockedError, SessionRecord, SessionStats,
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteSessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] > 2:
                raise RuntimeError("Database schema is newer than this Bumi version; refusing to downgrade")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active','processing','failed','consolidated','discarded')),
                    started_at TEXT NOT NULL, closed_at TEXT, close_reason TEXT,
                    summary TEXT, summary_provider TEXT, summary_status TEXT NOT NULL DEFAULT 'pending',
                    summary_error TEXT, consolidation_status TEXT NOT NULL DEFAULT 'pending',
                    consolidation_attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    sequence_no INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                    content TEXT NOT NULL, token_count INTEGER NOT NULL,
                    context_only INTEGER NOT NULL DEFAULT 0, source_session_id TEXT,
                    created_at TEXT NOT NULL, consolidated_at TEXT,
                    memory_excluded INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(session_id,sequence_no)
                );
                CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session_id,sequence_no);
                CREATE TABLE IF NOT EXISTS consolidation_runs (
                    consolidation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    attempt INTEGER NOT NULL, chunk_index INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
                    provider TEXT NOT NULL, status TEXT NOT NULL,
                    started_at TEXT, finished_at TEXT, mem0_events_json TEXT, error TEXT
                );
                CREATE TABLE IF NOT EXISTS memory_scopes (
                    user_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 0,
                    delete_pending INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(user_id,agent_id)
                );
                CREATE TABLE IF NOT EXISTS memory_jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK(kind IN ('consolidate','delete')),
                    session_id TEXT UNIQUE REFERENCES sessions(session_id), new_session_id TEXT,
                    user_id TEXT NOT NULL, agent_id TEXT NOT NULL, generation INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','retry_wait','succeeded','failed','cancelled')),
                    attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL,
                    available_at REAL NOT NULL, owner TEXT, lease_until REAL,
                    reason TEXT NOT NULL, payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_error TEXT
                );
                CREATE INDEX IF NOT EXISTS memory_jobs_due ON memory_jobs(status,available_at);
                CREATE INDEX IF NOT EXISTS memory_jobs_scope ON memory_jobs(user_id,agent_id,created_at);
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    chunk_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES memory_jobs(job_id),
                    chunk_index INTEGER NOT NULL, messages_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','writing','succeeded','uncertain')),
                    events_json TEXT NOT NULL DEFAULT '[]', last_error TEXT,
                    UNIQUE(job_id,chunk_index)
                );
                CREATE TABLE IF NOT EXISTS memory_worker_lease (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), owner TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
            """)
            columns = {r['name'] for r in conn.execute('PRAGMA table_info(messages)')}
            if 'memory_excluded' not in columns:
                conn.execute('ALTER TABLE messages ADD COLUMN memory_excluded INTEGER NOT NULL DEFAULT 0')
            # v0.1 included processing/failed, which blocked every new conversation.
            old = conn.execute("SELECT sql FROM sqlite_master WHERE name='one_open_session_per_pair'").fetchone()
            if old and 'processing' in old['sql']:
                conn.execute('DROP INDEX one_open_session_per_pair')
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_open_session_per_pair "
                         "ON sessions(user_id,agent_id) WHERE status='active'")
            # Never relabel live running work just because another reader starts.
            conn.execute('PRAGMA user_version=2')

    @staticmethod
    def _session(row: sqlite3.Row | None) -> SessionRecord | None:
        if row is None:
            return None
        fields = SessionRecord.__dataclass_fields__
        return SessionRecord(**{name: row[name] for name in fields})

    @staticmethod
    def _message(row: sqlite3.Row) -> MessageRecord:
        values = {name: row[name] for name in MessageRecord.__dataclass_fields__}
        values['context_only'] = bool(values['context_only'])
        return MessageRecord(**values)

    @staticmethod
    def _job(row: sqlite3.Row | None) -> MemoryJob | None:
        return MemoryJob(**dict(row)) if row is not None else None

    @staticmethod
    def _scope(conn: sqlite3.Connection, user: str, agent: str) -> sqlite3.Row:
        conn.execute('INSERT OR IGNORE INTO memory_scopes(user_id,agent_id) VALUES(?,?)', (user, agent))
        return conn.execute('SELECT * FROM memory_scopes WHERE user_id=? AND agent_id=?', (user, agent)).fetchone()

    def _insert_session(self, conn: sqlite3.Connection, user: str, agent: str,
                        carryover: Sequence[MessageRecord] = ()) -> str:
        sid, now = uuid4().hex, utcnow()
        conn.execute("INSERT INTO sessions(session_id,user_id,agent_id,status,started_at,created_at) "
                     "VALUES(?,?,?,'active',?,?)", (sid, user, agent, now, now))
        for index, message in enumerate(carryover, 1):
            conn.execute('INSERT INTO messages(message_id,session_id,sequence_no,role,content,token_count,'
                         'context_only,source_session_id,created_at) VALUES(?,?,?,?,?,?,1,?,?)',
                         (uuid4().hex, sid, index, message.role, message.content,
                          message.token_count, message.session_id, now))
        return sid

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            return self._session(conn.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone())

    def get_open_session(self, user_id: str, agent_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            return self._session(conn.execute("SELECT * FROM sessions WHERE user_id=? AND agent_id=? "
                                             "AND status='active'", (user_id, agent_id)).fetchone())

    def create_session(self, user_id: str, agent_id: str, *,
                       carryover: Sequence[MessageRecord] = ()) -> SessionRecord:
        with self._connect(write=True) as conn:
            sid = self._insert_session(conn, user_id, agent_id, carryover)
        return self.get_session(sid)

    def get_or_create_session(self, user_id: str, agent_id: str) -> SessionRecord:
        with self._connect(write=True) as conn:
            row = conn.execute("SELECT * FROM sessions WHERE user_id=? AND agent_id=? AND status='active'",
                               (user_id, agent_id)).fetchone()
            if row is not None:
                return self._session(row)
            sid = self._insert_session(conn, user_id, agent_id)
        return self.get_session(sid)

    def append_message(self, session_id: str, *, role: str, content: str, token_count: int,
                       context_only: bool = False, source_session_id: str | None = None) -> MessageRecord:
        content = content.strip()
        if not content:
            raise ValueError('message content cannot be empty')
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT status FROM sessions WHERE session_id=?', (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            if row['status'] != 'active':
                raise SessionBlockedError(f'session {session_id} is frozen; use the active session')
            sequence = conn.execute('SELECT COALESCE(MAX(sequence_no),0)+1 FROM messages WHERE session_id=?',
                                    (session_id,)).fetchone()[0]
            mid = uuid4().hex
            conn.execute('INSERT INTO messages(message_id,session_id,sequence_no,role,content,token_count,'
                         'context_only,source_session_id,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                         (mid, session_id, sequence, role, content, max(0, token_count),
                          int(context_only), source_session_id, utcnow()))
            return self._message(conn.execute('SELECT * FROM messages WHERE message_id=?', (mid,)).fetchone())

    def list_messages(self, session_id: str, *, include_context_only: bool = True) -> list[MessageRecord]:
        extra = '' if include_context_only else ' AND context_only=0'
        with self._connect() as conn:
            return [self._message(r) for r in conn.execute('SELECT * FROM messages WHERE session_id=?' + extra +
                                                          ' ORDER BY sequence_no', (session_id,))]

    def effective_messages(self, session_id: str) -> list[MessageRecord]:
        return self.list_messages(session_id, include_context_only=False)

    def stats(self, session_id: str) -> SessionStats:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        messages = self.list_messages(session_id)
        real = [m for m in messages if not m.context_only]
        return SessionStats(session_id, session.status, len(real), len(messages)-len(real),
                            sum(m.token_count for m in real))

    def submit(self, session_id: str, *, reason: str, settings: dict[str, Any],
               carryover_messages: int = 4, max_attempts: int = 3, legacy: bool = False) -> ConsolidationSubmission:
        """Freeze, snapshot, enqueue and rotate in ONE transaction (no model calls)."""
        with self._connect(write=True) as conn:
            existing = conn.execute('SELECT * FROM memory_jobs WHERE session_id=?', (session_id,)).fetchone()
            if existing:
                return ConsolidationSubmission(existing['job_id'], session_id, existing['new_session_id'], existing['status'])
            old = conn.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone()
            if old is None:
                raise KeyError(session_id)
            if old['status'] != 'active' and not (legacy and old['status'] in ('processing', 'failed')):
                raise SessionBlockedError(f'cannot submit {old["status"]} session')
            rows = conn.execute('SELECT * FROM messages WHERE session_id=? AND context_only=0 ORDER BY sequence_no',
                                (session_id,)).fetchall()
            eligible = [self._message(r) for r in rows if not r['memory_excluded']]
            tail = [self._message(r) for r in rows[-carryover_messages:]] if carryover_messages else []
            scope = self._scope(conn, old['user_id'], old['agent_id'])
            now, jid = utcnow(), 'session_' + session_id
            state = 'queued' if eligible else 'succeeded'
            error = None
            if legacy and conn.execute("SELECT 1 FROM consolidation_runs WHERE session_id=? "
                                       "AND status IN ('processing','success','failed') LIMIT 1", (session_id,)).fetchone():
                state, error = 'failed', 'Legacy partial write: review L3 before explicitly allowing replay.'
            conn.execute('UPDATE sessions SET status=?,closed_at=?,close_reason=?,consolidation_status=? WHERE session_id=?',
                         ('consolidated' if not eligible else ('failed' if error else 'processing'),
                          now, reason, 'skipped' if not eligible else state, session_id))
            active = conn.execute("SELECT session_id FROM sessions WHERE user_id=? AND agent_id=? AND status='active'",
                                  (old['user_id'], old['agent_id'])).fetchone()
            new_sid = active['session_id'] if active else self._insert_session(conn, old['user_id'], old['agent_id'], tail)
            payload = json.dumps({'messages': [asdict(m) for m in eligible], 'settings': settings}, ensure_ascii=False)
            conn.execute('INSERT INTO memory_jobs(job_id,kind,session_id,new_session_id,user_id,agent_id,generation,'
                         'status,max_attempts,available_at,reason,payload_json,created_at,updated_at,last_error) '
                         "VALUES(?,'consolidate',?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (jid, session_id, new_sid, old['user_id'], old['agent_id'], scope['generation'], state,
                          max_attempts, time.time(), reason, payload, now, now, error))
            return ConsolidationSubmission(jid, session_id, new_sid, state)

    def legacy_sessions(self) -> list[SessionRecord]:
        with self._connect() as conn:
            return [self._session(r) for r in conn.execute("SELECT s.* FROM sessions s WHERE s.status IN ('processing','failed') "
                                                          'AND NOT EXISTS(SELECT 1 FROM memory_jobs j WHERE j.session_id=s.session_id)')]

    def discard_and_create(self, session_id: str, *, reason: str = 'manual_discard') -> SessionRecord:
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone()
            if row is None or row['status'] != 'active':
                raise SessionBlockedError('only the active session can be discarded')
            conn.execute("UPDATE sessions SET status='discarded',closed_at=?,close_reason=? WHERE session_id=?",
                         (utcnow(), reason, session_id))
            sid = self._insert_session(conn, row['user_id'], row['agent_id'])
        return self.get_session(sid)

    def get_job(self, job_id: str) -> MemoryJob | None:
        with self._connect() as conn:
            return self._job(conn.execute('SELECT * FROM memory_jobs WHERE job_id=?', (job_id,)).fetchone())

    def list_jobs(self, *, user_id: str, agent_id: str, limit: int = 50) -> list[MemoryJob]:
        with self._connect() as conn:
            return [self._job(r) for r in conn.execute('SELECT * FROM memory_jobs WHERE user_id=? AND agent_id=? '
                                                      'ORDER BY created_at DESC,rowid DESC LIMIT ?', (user_id, agent_id, limit))]

    def read_scope(self, user_id: str, agent_id: str) -> tuple[int, bool]:
        with self._connect() as conn:
            row = conn.execute('SELECT generation,delete_pending FROM memory_scopes WHERE user_id=? AND agent_id=?',
                               (user_id, agent_id)).fetchone()
            return (row['generation'], bool(row['delete_pending'])) if row else (0, False)

    def reads_blocked(self, user_id: str, agent_id: str) -> bool:
        return self.read_scope(user_id, agent_id)[1]

    def request_forget(self, *, user_id: str, agent_id: str, max_attempts: int = 3) -> MemoryJob:
        """Suppress reads immediately; serialize physical deletion behind in-flight writes."""
        with self._connect(write=True) as conn:
            scope = self._scope(conn, user_id, agent_id)
            generation = scope['generation'] + 1
            now, jid = utcnow(), 'delete_' + uuid4().hex
            conn.execute('UPDATE memory_scopes SET generation=?,delete_pending=1 WHERE user_id=? AND agent_id=?',
                         (generation, user_id, agent_id))
            conn.execute("UPDATE memory_jobs SET status='cancelled',updated_at=?,last_error='Cancelled by forget-all' "
                         "WHERE user_id=? AND agent_id=? AND status IN ('queued','running','retry_wait','failed')",
                         (now, user_id, agent_id))
            conn.execute("UPDATE sessions SET status='discarded',consolidation_status='cancelled' "
                         "WHERE user_id=? AND agent_id=? AND status IN ('processing','failed')", (user_id, agent_id))
            conn.execute('UPDATE messages SET memory_excluded=1 WHERE session_id IN '
                         '(SELECT session_id FROM sessions WHERE user_id=? AND agent_id=?)', (user_id, agent_id))
            conn.execute('INSERT INTO memory_jobs(job_id,kind,user_id,agent_id,generation,status,max_attempts,available_at,'
                         'reason,payload_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                         (jid, 'delete', user_id, agent_id, generation, 'queued', max_attempts,
                          time.time(), 'forget_all', '{}', now, now))
        return self.get_job(jid)

    def claim_job(self, owner: str, *, lease_seconds: float) -> MemoryJob | None:
        now = time.time()
        with self._connect(write=True) as conn:
            lock = conn.execute('SELECT * FROM memory_worker_lease WHERE singleton=1').fetchone()
            if lock and lock['owner'] != owner and lock['expires_at'] > now:
                return None
            conn.execute('INSERT INTO memory_worker_lease(singleton,owner,expires_at) VALUES(1,?,?) '
                         'ON CONFLICT(singleton) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at',
                         (owner, now + lease_seconds))
            # A persisted "writing" checkpoint after a crash is NOT evidence of failure.
            for row in conn.execute("SELECT * FROM memory_jobs WHERE status='running' AND lease_until<=?", (now,)).fetchall():
                uncertain = conn.execute("SELECT 1 FROM memory_chunks WHERE job_id=? AND status='writing'", (row['job_id'],)).fetchone()
                error = 'Write outcome unknown after interruption; explicit replay confirmation required.' if uncertain else 'Interrupted; recovered from lease expiry.'
                state = 'failed' if uncertain or row['attempts'] >= row['max_attempts'] else 'queued'
                conn.execute('UPDATE memory_jobs SET status=?,owner=NULL,lease_until=NULL,last_error=?,updated_at=? WHERE job_id=?',
                             (state, error, utcnow(), row['job_id']))
                if uncertain:
                    conn.execute("UPDATE memory_chunks SET status='uncertain' WHERE job_id=? AND status='writing'", (row['job_id'],))
                if row['session_id']:
                    conn.execute('UPDATE sessions SET consolidation_status=?,last_error=? WHERE session_id=?',
                                 (state, error, row['session_id']))
            row = conn.execute("""
                SELECT j.* FROM memory_jobs j JOIN memory_scopes s USING(user_id,agent_id)
                WHERE j.status IN ('queued','retry_wait') AND j.available_at<=? AND j.generation=s.generation
                  AND (j.kind='delete' OR s.delete_pending=0)
                  AND (j.kind='delete' OR NOT EXISTS (
                    SELECT 1 FROM memory_jobs older WHERE older.user_id=j.user_id AND older.agent_id=j.agent_id
                      AND older.status IN ('queued','running','retry_wait') AND older.rowid<j.rowid))
                ORDER BY CASE j.kind WHEN 'delete' THEN 0 ELSE 1 END,j.rowid LIMIT 1
            """, (now,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE memory_jobs SET status='running',attempts=attempts+1,owner=?,lease_until=?,updated_at=? WHERE job_id=?",
                         (owner, now + lease_seconds, utcnow(), row['job_id']))
            if row['session_id']:
                conn.execute("UPDATE sessions SET status='processing',consolidation_status='running',consolidation_attempts=? WHERE session_id=?",
                             (row['attempts'] + 1, row['session_id']))
            return self._job(conn.execute('SELECT * FROM memory_jobs WHERE job_id=?', (row['job_id'],)).fetchone())

    def renew_lease(self, owner: str, *, lease_seconds: float) -> bool:
        now = time.time()
        with self._connect(write=True) as conn:
            changed = conn.execute('UPDATE memory_worker_lease SET expires_at=? WHERE singleton=1 AND owner=? AND expires_at>?',
                                   (now + lease_seconds, owner, now)).rowcount
            if changed:
                conn.execute("UPDATE memory_jobs SET lease_until=? WHERE owner=? AND status='running'", (now + lease_seconds, owner))
            return bool(changed)

    def release_lease(self, owner: str) -> None:
        with self._connect(write=True) as conn:
            conn.execute('DELETE FROM memory_worker_lease WHERE singleton=1 AND owner=?', (owner,))

    def _assert_current(self, conn: sqlite3.Connection, job_id: str, owner: str) -> sqlite3.Row:
        row = conn.execute('SELECT j.*,s.generation AS live_generation,s.delete_pending '
                           'FROM memory_jobs j JOIN memory_scopes s USING(user_id,agent_id) WHERE job_id=?', (job_id,)).fetchone()
        lease = conn.execute('SELECT * FROM memory_worker_lease WHERE singleton=1').fetchone()
        if (row is None or row['status'] != 'running' or row['owner'] != owner
            or row['generation'] != row['live_generation']
            or (row['kind'] != 'delete' and row['delete_pending'])
            or not lease or lease['owner'] != owner or lease['expires_at'] <= time.time()):
            raise JobCancelledError('job cancelled or worker lease lost')
        return row

    def assert_current(self, job_id: str, owner: str) -> None:
        with self._connect() as conn:
            self._assert_current(conn, job_id, owner)

    def save_job_summary(self, job_id: str, owner: str, *, summary: str | None,
                         provider: str, status: str, error: str | None = None) -> None:
        with self._connect(write=True) as conn:
            row = self._assert_current(conn, job_id, owner)
            conn.execute('UPDATE sessions SET summary=?,summary_provider=?,summary_status=?,summary_error=? WHERE session_id=?',
                         (summary, provider, status, error, row['session_id']))

    def ensure_chunks(self, job_id: str, owner: str, chunks: Sequence[Sequence[dict[str, str]]]) -> None:
        with self._connect(write=True) as conn:
            self._assert_current(conn, job_id, owner)
            if conn.execute('SELECT 1 FROM memory_chunks WHERE job_id=?', (job_id,)).fetchone():
                return
            for index, messages in enumerate(chunks):
                conn.execute("INSERT INTO memory_chunks(chunk_id,job_id,chunk_index,messages_json,status) VALUES(?,?,?,?,'pending')",
                             (f'{job_id}:chunk:{index}', job_id, index, json.dumps(list(messages), ensure_ascii=False)))

    def chunks(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute('SELECT * FROM memory_chunks WHERE job_id=? ORDER BY chunk_index', (job_id,))]

    def set_chunk(self, job_id: str, owner: str, chunk_id: str, *, status: str,
                  events: list[dict] | None = None, error: str | None = None) -> None:
        with self._connect(write=True) as conn:
            self._assert_current(conn, job_id, owner)
            row = conn.execute('SELECT status FROM memory_chunks WHERE chunk_id=? AND job_id=?', (chunk_id, job_id)).fetchone()
            allowed = {'writing': {'pending'}, 'pending': {'writing'}, 'succeeded': {'writing'}, 'uncertain': {'writing'}}
            if not row or row['status'] not in allowed[status]:
                raise JobCancelledError('chunk state changed')
            conn.execute('UPDATE memory_chunks SET status=?,events_json=?,last_error=? WHERE chunk_id=?',
                         (status, json.dumps(events or [], ensure_ascii=False, default=str), error, chunk_id))

    def finish_job(self, job_id: str, owner: str) -> None:
        with self._connect(write=True) as conn:
            row = self._assert_current(conn, job_id, owner)
            if row['kind'] == 'consolidate':
                if conn.execute("SELECT 1 FROM memory_chunks WHERE job_id=? AND status!='succeeded'", (job_id,)).fetchone():
                    raise RuntimeError('cannot complete unfinished chunks')
                conn.execute("UPDATE sessions SET status='consolidated',consolidation_status='success',last_error=NULL WHERE session_id=?",
                             (row['session_id'],))
                mids = [m['message_id'] for m in json.loads(row['payload_json'])['messages']]
                conn.executemany('UPDATE messages SET consolidated_at=? WHERE message_id=?', [(utcnow(), mid) for mid in mids])
            else:
                conn.execute('UPDATE memory_scopes SET delete_pending=0 WHERE user_id=? AND agent_id=? AND generation=?',
                             (row['user_id'], row['agent_id'], row['generation']))
            conn.execute("UPDATE memory_jobs SET status='succeeded',owner=NULL,lease_until=NULL,last_error=NULL,updated_at=? WHERE job_id=?",
                         (utcnow(), job_id))

    def fail_job(self, job_id: str, owner: str, error: str, *, retry_delay: float,
                 retryable: bool = True) -> None:
        with self._connect(write=True) as conn:
            row = self._assert_current(conn, job_id, owner)
            state = 'retry_wait' if retryable and row['attempts'] < row['max_attempts'] else 'failed'
            conn.execute('UPDATE memory_jobs SET status=?,available_at=?,owner=NULL,lease_until=NULL,last_error=?,updated_at=? WHERE job_id=?',
                         (state, time.time()+retry_delay, error, utcnow(), job_id))
            if row['session_id']:
                conn.execute('UPDATE sessions SET status=?,consolidation_status=?,last_error=? WHERE session_id=?',
                             ('failed' if state == 'failed' else 'processing', state, error, row['session_id']))

    def retry_job(self, job_id: str, *, user_id: str, agent_id: str, max_attempts: int = 3,
                  allow_replay: bool = False) -> MemoryJob:
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM memory_jobs WHERE job_id=? AND user_id=? AND agent_id=?',
                               (job_id, user_id, agent_id)).fetchone()
            if row is None:
                raise KeyError('unknown job for this user/agent')
            scope = self._scope(conn, user_id, agent_id)
            if row['status'] != 'failed' or row['generation'] != scope['generation']:
                raise ValueError('only a failed, non-deleted job can be retried')
            uncertain = conn.execute("SELECT 1 FROM memory_chunks WHERE job_id=? AND status IN ('writing','uncertain')", (job_id,)).fetchone()
            if (uncertain or (row['last_error'] or '').startswith('Legacy partial')) and not allow_replay:
                raise ValueError('write outcome is unknown; inspect L3, then explicitly confirm replay')
            if allow_replay:
                conn.execute("UPDATE memory_chunks SET status='pending' WHERE job_id=? AND status IN ('writing','uncertain')", (job_id,))
            conn.execute("UPDATE memory_jobs SET status='queued',max_attempts=?,available_at=?,owner=NULL,lease_until=NULL,updated_at=? WHERE job_id=?",
                         (row['attempts'] + max_attempts, time.time(), utcnow(), job_id))
        return self.get_job(job_id)

    def list_consolidation_runs(self, session_id: str) -> list[dict[str, Any]]:
        """Expose stable chunk checkpoints; legacy audit rows remain preserved."""
        with self._connect() as conn:
            rows = conn.execute('SELECT c.*,j.attempts AS attempt FROM memory_chunks c JOIN memory_jobs j USING(job_id) '
                                'WHERE j.session_id=? ORDER BY c.chunk_index', (session_id,)).fetchall()
            return [dict(r) for r in rows] or [dict(r) for r in conn.execute(
                'SELECT * FROM consolidation_runs WHERE session_id=? ORDER BY attempt,chunk_index', (session_id,))]
