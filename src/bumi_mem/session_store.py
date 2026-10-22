# 本次改动（2026-10-21）：接入注册表存储并迁移至 schema 11，阻止旧程序降级读取新库。
"""SQLite 持久化存储：L1/L2 消息、会话、共享窗口、后台作业队列。

外部模型/向量调用不在 SQLite 事务内执行。会话窗口轮转和所有者作业快照
在同一事务中提交。每用户会话保持私有的抽取/审计流，shared_l1 引用构成
每个 agent 的一个按时间排序的前台上下文。冻结的所有者会话可与新对话共存。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4

from bumi_mem.schemas import (
    ConsolidationSubmission, JobCancelledError, MemoryJob, MemoryWriteUncertainError,
    MessageRecord, SessionBlockedError, SessionRecord, SessionStats,
    canonical_messages_digest,
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


from bumi_mem.registry import RegistryStoreMixin
from bumi_mem.control_store import ControlStoreMixin
from bumi_mem.turn_store import TurnStoreMixin
from bumi_mem.context_store import ContextStoreMixin
from bumi_mem.shared_memory import SharedMemoryStoreMixin
from bumi_mem.semantic_state import SemanticStateStoreMixin
from bumi_mem.identity import is_guest
from bumi_mem.event_time import message_time
from bumi_mem.agent_store import AgentStoreMixin


class SQLiteSessionStore(RegistryStoreMixin, ControlStoreMixin, TurnStoreMixin, ContextStoreMixin, SharedMemoryStoreMixin, SemanticStateStoreMixin, AgentStoreMixin):
    """SQLite 持久化存储：L1/L2 消息、会话、共享窗口、后台作业队列。

    设计原则：
    - 外部模型/向量调用不在 SQLite 事务内执行
    - 会话窗口轮转和所有者作业快照在同一事务中提交
    - 每用户会话保持私有的抽取/审计流
    - shared_l1 引用构成每个 agent 的一个按时间排序的前台上下文
    - 冻结的所有者会话可与新对话共存
    """
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
            if conn.execute("PRAGMA user_version").fetchone()[0] > 11:
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
                CREATE TABLE IF NOT EXISTS chunk_replay_auth (
                    chunk_id TEXT PRIMARY KEY REFERENCES memory_chunks(chunk_id) ON DELETE CASCADE,
                    job_id TEXT NOT NULL REFERENCES memory_jobs(job_id),
                    user_id TEXT NOT NULL, agent_id TEXT NOT NULL, generation INTEGER NOT NULL,
                    input_digest TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('issued','executing','consumed')),
                    issued_at REAL NOT NULL, consumed_at REAL,
                    attempt INTEGER NOT NULL DEFAULT 0, last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS chunk_replay_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL, job_id TEXT NOT NULL, event TEXT NOT NULL,
                    user_id TEXT NOT NULL, agent_id TEXT NOT NULL, generation INTEGER NOT NULL,
                    input_digest TEXT NOT NULL, at REAL NOT NULL, detail TEXT NOT NULL DEFAULT ''
                );
            """)
            columns = {r['name'] for r in conn.execute('PRAGMA table_info(messages)')}
            if 'memory_excluded' not in columns:
                conn.execute('ALTER TABLE messages ADD COLUMN memory_excluded INTEGER NOT NULL DEFAULT 0')
            # v3: immutable attribution plus a persisted per-agent shared window.
            for name in ('user_id', 'agent_id'):
                if name not in columns:
                    conn.execute(f"ALTER TABLE messages ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
                    conn.execute(f"UPDATE messages SET {name}=(SELECT {name} FROM sessions s "
                                 "WHERE s.session_id=messages.session_id)")
            conn.execute("CREATE TABLE IF NOT EXISTS shared_l1 ("
                         "position INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL, "
                         "message_id TEXT NOT NULL UNIQUE REFERENCES messages(message_id) ON DELETE CASCADE, "
                         "context_only INTEGER NOT NULL DEFAULT 0)")
            conn.execute("CREATE INDEX IF NOT EXISTS shared_l1_agent ON shared_l1(agent_id,position)")
            if conn.execute('PRAGMA user_version').fetchone()[0] < 3:
                # 只迁移活跃窗口，不复活已丢弃/冻结的历史
                # 携带副本保留其来源身份并保持 context-only
                conn.execute("INSERT OR IGNORE INTO shared_l1(agent_id,message_id,context_only) "
                             "SELECT s.agent_id,m.message_id,m.context_only FROM messages m "
                             "JOIN sessions s USING(session_id) WHERE s.status='active' "
                             "ORDER BY m.created_at,m.rowid")
            # v0.1 included processing/failed, which blocked every new conversation.
            old = conn.execute("SELECT sql FROM sqlite_master WHERE name='one_open_session_per_pair'").fetchone()
            if old and 'processing' in old['sql']:
                conn.execute('DROP INDEX one_open_session_per_pair')
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_open_session_per_pair "
                         "ON sessions(user_id,agent_id) WHERE status='active'")
            # 不因另一个读者启动就重新标记正在运行的工作
            conn.execute("CREATE TABLE IF NOT EXISTS interaction_turns ("
                         "turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id), "
                         "user_id TEXT NOT NULL, agent_id TEXT NOT NULL, query TEXT NOT NULL, "
                         "idempotency_key TEXT, status TEXT NOT NULL CHECK(status IN "
                         "('preparing','awaiting_reply','completed','interrupted','cancelled','failed','expired')), "
                         "expires_at REAL NOT NULL, created_at REAL NOT NULL, generation INTEGER NOT NULL, "
                         "prompt_json TEXT, finish_payload TEXT, completion_json TEXT, UNIQUE(agent_id,idempotency_key))")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_pending_turn_per_agent "
                         "ON interaction_turns(agent_id) WHERE status IN ('preparing','awaiting_reply')")
            turn_columns = {r['name'] for r in conn.execute('PRAGMA table_info(interaction_turns)')}
            if 'protocol' not in turn_columns:
                conn.execute("ALTER TABLE interaction_turns ADD COLUMN protocol TEXT NOT NULL DEFAULT 'direct'")
            conn.execute("CREATE TABLE IF NOT EXISTS context_requests ("
                         "request_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, user_id TEXT NOT NULL, "
                         "query TEXT, previous_ai_reply TEXT NOT NULL, idempotency_key TEXT, "
                         "turn_id TEXT REFERENCES interaction_turns(turn_id), previous_turn_id TEXT, "
                         "previous_reply_saved INTEGER NOT NULL, consolidation_status TEXT NOT NULL, "
                         "state TEXT NOT NULL DEFAULT 'preparing', response_json TEXT, created_at REAL NOT NULL, "
                         "UNIQUE(agent_id,idempotency_key))")
            if conn.execute('PRAGMA user_version').fetchone()[0] < 5:
                # assistant 以 agent 身份说话，而非 user。会话和旧任务快照仍保留抽取来源
                conn.execute("UPDATE messages SET user_id='' WHERE role!='user'")
                conn.execute("UPDATE interaction_turns SET prompt_json=NULL WHERE status IN ('preparing','awaiting_reply')")
            # v6: source-linked shared recall, semantic state and cross-user cache revocation.
            self.create_shared_memory_tables(conn)
            self.create_semantic_tables(conn)
            for table, additions in {
                'interaction_turns': [('source_message_id', 'TEXT'), ('reply_message_id', 'TEXT'), ('context_epoch', 'INTEGER NOT NULL DEFAULT 0')],
                'context_requests': [('context_epoch', 'INTEGER NOT NULL DEFAULT 0'), ('query_occurred_at', 'TEXT'), ('previous_reply_occurred_at', 'TEXT')],
            }.items():
                existing_columns = {r['name'] for r in conn.execute(f'PRAGMA table_info({table})')}
                for name, definition in additions:
                    if name not in existing_columns:
                        conn.execute(f'ALTER TABLE {table} ADD COLUMN {name} {definition}')
            # v5 待处理轮次可安全恢复其原始用户证据
            conn.execute("UPDATE interaction_turns SET source_message_id=(SELECT message_id FROM messages m "
                         "WHERE m.session_id=interaction_turns.session_id AND m.role='user' "
                         "AND m.content=interaction_turns.query ORDER BY m.sequence_no DESC LIMIT 1) "
                         "WHERE source_message_id IS NULL")
            if conn.execute('PRAGMA user_version').fetchone()[0] < 6:
                conn.execute("UPDATE interaction_turns SET prompt_json=NULL WHERE status IN ('preparing','awaiting_reply')")
                conn.execute("UPDATE context_requests SET response_json=NULL WHERE state!='revoked'")
            self.create_agent_tables(conn)
            self.create_control_tables(conn)
            self.create_registry_tables(conn)
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if version < 8:
                self.migrate_context_request_protocols(conn)
            if version < 9:
                self.migrate_structured_turn_responses(conn)
            conn.execute('PRAGMA user_version=11')

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
                         'context_only,source_session_id,created_at,user_id,agent_id) VALUES(?,?,?,?,?,?,1,?,?,?,?)',
                         (uuid4().hex, sid, index, message.role, message.content,
                          message.token_count, message.session_id, now, (message.user_id or user) if message.role == 'user' else '', message.agent_id or agent))
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
        with self._connect(write=True) as conn:
            owner = conn.execute('SELECT agent_id FROM sessions WHERE session_id=?', (session_id,)).fetchone()
            if owner is not None:
                self._assert_agent_idle(conn, owner['agent_id'])
            return self._append_message(conn, session_id, role=role, content=content, token_count=token_count,
                                        context_only=context_only, source_session_id=source_session_id)

    def _append_message(self, conn, session_id, *, role, content, token_count,
                        context_only=False, source_session_id=None, occurred_at=None):
        content = content.strip()
        if not content:
            raise ValueError('message content cannot be empty')
        row = conn.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        if row['status'] != 'active':
            raise SessionBlockedError(f'session {session_id} is frozen; use the active session')
        sequence = conn.execute('SELECT COALESCE(MAX(sequence_no),0)+1 FROM messages WHERE session_id=?',
                                (session_id,)).fetchone()[0]
        mid = uuid4().hex
        conn.execute('INSERT INTO messages(message_id,session_id,sequence_no,role,content,token_count,'
                     'context_only,source_session_id,created_at,user_id,agent_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                     (mid, session_id, sequence, role, content, max(0, token_count),
                      int(context_only), source_session_id, message_time(occurred_at), row['user_id'] if role == 'user' else '', row['agent_id']))
        conn.execute('INSERT INTO shared_l1(agent_id,message_id,context_only) VALUES(?,?,?)',
                     (row['agent_id'], mid, int(context_only)))
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
               carryover_messages: int = 20, max_attempts: int = 3, legacy: bool = False) -> ConsolidationSubmission:
        """Freeze, snapshot, enqueue and rotate in ONE transaction (no model calls)."""
        with self._connect(write=True) as conn:
            return self._submit(conn, session_id, reason=reason, settings=settings,
                                carryover_messages=carryover_messages, max_attempts=max_attempts, legacy=legacy)

    def _submit(self, conn: sqlite3.Connection, session_id: str, *, reason: str,
                settings: dict[str, Any], carryover_messages: int, max_attempts: int,
                legacy: bool = False) -> ConsolidationSubmission:
        existing = conn.execute('SELECT * FROM memory_jobs WHERE session_id=?', (session_id,)).fetchone()
        if existing:
            return ConsolidationSubmission(existing['job_id'], session_id, existing['new_session_id'], existing['status'])
        old = conn.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone()
        if old is None:
            raise KeyError(session_id)
        if old['status'] != 'active' and not (legacy and old['status'] in ('processing', 'failed')):
            raise SessionBlockedError(f'cannot submit {old["status"]} session')
        rows = conn.execute('SELECT * FROM messages WHERE session_id=? AND context_only=0 AND message_id NOT IN (SELECT message_id FROM memory_snapshot_sources) ORDER BY sequence_no',
                            (session_id,)).fetchall()
        eligible = [] if is_guest(old['user_id']) else [self._message(r) for r in rows if not r['memory_excluded']]
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

    def list_shared_messages(self, agent_id: str) -> list[MessageRecord]:
        with self._connect() as conn:
            rows = conn.execute('SELECT m.*,w.context_only AS shared_context_only '
                                'FROM shared_l1 w JOIN messages m USING(message_id) '
                                'WHERE w.agent_id=? ORDER BY w.position', (agent_id,)).fetchall()
            return [replace(self._message(row), context_only=bool(row['shared_context_only'])) for row in rows]

    def shared_stats(self, agent_id: str, *, session_id: str) -> SessionStats:
        messages = self.list_shared_messages(agent_id)
        real = [m for m in messages if not m.context_only]
        return SessionStats(session_id, 'active', len(real), len(messages)-len(real),
                            sum(m.token_count for m in real))

    def submit_shared(self, session_id: str, *, reason: str, settings: dict[str, Any],
                      carryover_messages: int = 20, max_attempts: int = 3) -> ConsolidationSubmission:
        """Rotate the agent window and enqueue separate owner snapshots atomically."""
        with self._connect(write=True) as conn:
            return self._submit_shared(conn, session_id, reason=reason, settings=settings,
                                       carryover_messages=carryover_messages, max_attempts=max_attempts)

    def _submit_shared(self, conn, session_id, *, reason, settings, carryover_messages=20, max_attempts=3):
        old = conn.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone()
        if old is None:
            raise KeyError(session_id)
        existing = conn.execute('SELECT * FROM memory_jobs WHERE session_id=?', (session_id,)).fetchone()
        if existing:
            return ConsolidationSubmission(existing['job_id'], session_id,
                                           existing['new_session_id'], existing['status'])
        if old['status'] != 'active':
            raise SessionBlockedError('only the active session can rotate shared L1')
        agent = old['agent_id']
        # Carryover must honour the real L1 window. Either the effective window
        # travels in the persisted snapshot, or the submission path states it
        # explicitly; there is no silent fallback to a guessed default.
        if settings.get('backend') == 'graphiti':
            configured = settings.get('l1_max_messages')
            if configured is None:
                raise ValueError('Graphiti submissions require l1_max_messages in the job snapshot')
            carryover_messages = max(carryover_messages, configured)
        self._assert_agent_idle(conn, agent)
        active = conn.execute("SELECT session_id FROM sessions WHERE agent_id=? AND status='active'",
                              (agent,)).fetchall()
        tail = conn.execute('SELECT message_id FROM shared_l1 WHERE agent_id=? '
                            'ORDER BY position DESC LIMIT ?', (agent, carryover_messages)).fetchall()
        receipts = {}
        for row in active:
            sid = row['session_id']
            receipts[sid] = self._submit(conn, sid, reason=reason, settings=settings,
                                         carryover_messages=carryover_messages,
                                         max_attempts=max_attempts)
        conn.execute('DELETE FROM shared_l1 WHERE agent_id=?', (agent,))
        conn.executemany('INSERT INTO shared_l1(agent_id,message_id,context_only) VALUES(?,?,1)',
                         [(agent, row['message_id']) for row in reversed(tail)])
        conn.execute('DELETE FROM shared_memory_refs WHERE agent_id=? AND anchor_message_id NOT IN '
                     '(SELECT message_id FROM shared_l1 WHERE agent_id=?)', (agent, agent))
        return receipts[session_id]

    def discard_shared(self, session_id: str) -> SessionRecord:
        """Explicit global discard for this agent only; already queued jobs survive."""
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone()
            if row is None or row['status'] != 'active':
                raise SessionBlockedError('only the active session can be discarded')
            self._assert_agent_idle(conn, row['agent_id'])
            conn.execute("UPDATE sessions SET status='discarded',closed_at=?,close_reason='manual_discard' "
                         "WHERE agent_id=? AND status='active'", (utcnow(), row['agent_id']))
            conn.execute('DELETE FROM shared_l1 WHERE agent_id=?', (row['agent_id'],))
            sid = self._insert_session(conn, row['user_id'], row['agent_id'])
        return self.get_session(sid)

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
        """请求遗忘全部记忆：立即抑制读取，将物理删除序列化到后台作业。

        流程：
        1. 递增作用域代数并标记 delete_pending
        2. 取消所有进行中的整理作业
        3. 废弃所有进行中/失败的会话
        4. 撤销所有待处理的上下文请求和交互轮次
        5. 删除所有共享引用和语义状态
        6. 排除所有消息的抽取资格
        7. 入队 delete 作业

        参数:
            user_id:      用户 ID
            agent_id:     Agent ID
            max_attempts: 最大重试次数
        返回:
            入队的 delete MemoryJob
        """
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
            conn.execute("UPDATE context_requests SET state='revoked',response_json=NULL "
                         "WHERE user_id=? AND agent_id=?", (user_id, agent_id))
            conn.execute("UPDATE interaction_turns SET prompt_json=NULL WHERE user_id=? AND agent_id=?",
                         (user_id, agent_id))
            conn.execute("UPDATE interaction_turns SET status='cancelled' "
                         "WHERE user_id=? AND agent_id=? AND protocol='direct' "
                         "AND status IN ('preparing','awaiting_reply')", (user_id, agent_id))
            conn.execute("UPDATE shared_memory_refs SET status='deleted' WHERE user_id=? AND agent_id=?", (user_id, agent_id))
            self.invalidate_semantic_scope(conn, user_id, agent_id)
            self._bump_context_epoch(conn, agent_id)
            conn.execute('UPDATE messages SET memory_excluded=1 WHERE session_id IN '
                         '(SELECT session_id FROM sessions WHERE user_id=? AND agent_id=?)', (user_id, agent_id))
            conn.execute('INSERT INTO memory_jobs(job_id,kind,user_id,agent_id,generation,status,max_attempts,available_at,'
                         'reason,payload_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                         (jid, 'delete', user_id, agent_id, generation, 'queued', max_attempts,
                          time.time(), 'forget_all', '{}', now, now))
        return self.get_job(jid)

    def oldest_due_job_age_seconds(self) -> float | None:
        """最早可执行后台任务的年龄（秒）；没有可执行任务时返回 None。只读，不认领任务。"""
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("""
                SELECT j.created_at FROM memory_jobs j JOIN memory_scopes s USING(user_id,agent_id)
                WHERE j.status IN ('queued','retry_wait') AND j.available_at<=? AND j.generation=s.generation
                  AND (j.kind='delete' OR s.delete_pending=0)
                  AND (j.kind='delete' OR NOT EXISTS (
                    SELECT 1 FROM memory_jobs older WHERE older.user_id=j.user_id AND older.agent_id=j.agent_id
                      AND older.status IN ('queued','running','retry_wait') AND older.rowid<j.rowid))
                ORDER BY CASE j.kind WHEN 'delete' THEN 0 ELSE 1 END,j.rowid LIMIT 1
            """, (now,)).fetchone()
        if row is None:
            return None
        try:
            return max(0.0, now - datetime.fromisoformat(row['created_at']).timestamp())
        except (TypeError, ValueError):
            return 0.0

    def has_pending_jobs(self) -> bool:
        """是否存在任何非终态作业（queued/retry_wait/running）。

        故意比 oldest_due_job_age_seconds() 更宽松：'running' 状态可能是租约已过期、
        等待 claim_job() 内部清理逻辑重新排队的僵尸作业（见 claim_job 开头的租约清理），
        这里不展开那层判断，只用于"要不要碰长期记忆"这个粗粒度决策，交给 claim_job() 处理细节。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM memory_jobs WHERE status IN ('queued','retry_wait','running') LIMIT 1").fetchone()
        return row is not None

    def claim_job(self, owner: str, *, lease_seconds: float) -> MemoryJob | None:
        now = time.time()
        with self._connect(write=True) as conn:
            lock = conn.execute('SELECT * FROM memory_worker_lease WHERE singleton=1').fetchone()
            if lock and lock['owner'] != owner and lock['expires_at'] > now:
                return None
            conn.execute('INSERT INTO memory_worker_lease(singleton,owner,expires_at) VALUES(1,?,?) '
                         'ON CONFLICT(singleton) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at',
                         (owner, now + lease_seconds))
            # 崩溃后持久化的 "writing" 检查点不是失败的证据
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
        """续租当前作业的工作锁，防止因租约过期被其他工作线程抢占。

        参数:
            owner:         工作线程标识
            lease_seconds: 新租约时长
        返回:
            是否续租成功
        """
        now = time.time()
        with self._connect(write=True) as conn:
            changed = conn.execute('UPDATE memory_worker_lease SET expires_at=? WHERE singleton=1 AND owner=? AND expires_at>?',
                                   (now + lease_seconds, owner, now)).rowcount
            if changed:
                conn.execute("UPDATE memory_jobs SET lease_until=? WHERE owner=? AND status='running'", (now + lease_seconds, owner))
            return bool(changed)

    def release_lease(self, owner: str) -> None:
        """释放工作锁，允许其他工作线程认领作业。"""
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
        """标记作业为成功完成。

        对于 consolidate 作业：检查所有 chunk 是否成功，标记消息为已整理。
        对于 delete 作业：清除 delete_pending 标记。

        参数:
            job_id: 作业 ID
            owner:  工作线程标识
        """
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
        """标记作业为失败，可重试时设为 retry_wait 状态。

        参数:
            job_id:      作业 ID
            owner:       工作线程标识
            error:       错误信息
            retry_delay: 重试等待时间（秒）
            retryable:   是否可重试
        """
        with self._connect(write=True) as conn:
            row = self._assert_current(conn, job_id, owner)
            state = 'retry_wait' if retryable and row['attempts'] < row['max_attempts'] else 'failed'
            conn.execute('UPDATE memory_jobs SET status=?,available_at=?,owner=NULL,lease_until=NULL,last_error=?,updated_at=? WHERE job_id=?',
                         (state, time.time()+retry_delay, error, utcnow(), job_id))
            if row['session_id']:
                conn.execute('UPDATE sessions SET status=?,consolidation_status=?,last_error=? WHERE session_id=?',
                             ('failed' if state == 'failed' else 'processing', state, error, row['session_id']))

    @staticmethod
    def chunk_input_digest(messages: Sequence[dict[str, Any]]) -> str:
        """Stable identity of a chunk's source payload.

        Delegates to the shared helper so the durable grant and the backend's
        re-computation can never drift apart.
        """
        return canonical_messages_digest(messages)

    def _grant_replays(self, conn: sqlite3.Connection, job_row: sqlite3.Row,
                       scope: sqlite3.Row) -> list[str]:
        """Issue one single-use replay authorization per uncertain chunk.

        Only chunks of THIS failed job qualify. Any authorization left over from
        an earlier decision is replaced, so a consumed or executing grant can
        never be handed out twice.
        """
        chunks = conn.execute("SELECT chunk_id,messages_json FROM memory_chunks "
                              "WHERE job_id=? AND status IN ('writing','uncertain') "
                              "ORDER BY chunk_index", (job_row['job_id'],)).fetchall()
        granted: list[str] = []
        for chunk in chunks:
            digest = self.chunk_input_digest(json.loads(chunk['messages_json']))
            conn.execute('DELETE FROM chunk_replay_auth WHERE chunk_id=?', (chunk['chunk_id'],))
            conn.execute('INSERT INTO chunk_replay_auth(chunk_id,job_id,user_id,agent_id,generation,'
                         "input_digest,status,issued_at,attempt) VALUES(?,?,?,?,?,?,'issued',?,0)",
                         (chunk['chunk_id'], job_row['job_id'], job_row['user_id'], job_row['agent_id'],
                          job_row['generation'], digest, time.time()))
            conn.execute('INSERT INTO chunk_replay_audit(chunk_id,job_id,event,user_id,agent_id,generation,'
                         "input_digest,at,detail) VALUES(?,?,'granted',?,?,?,?,?,'explicit allow_replay')",
                         (chunk['chunk_id'], job_row['job_id'], job_row['user_id'], job_row['agent_id'],
                          job_row['generation'], digest, time.time()))
            conn.execute("UPDATE memory_chunks SET status='pending' WHERE chunk_id=?", (chunk['chunk_id'],))
            granted.append(chunk['chunk_id'])
        return granted

    def take_replay_authorization(self, *, chunk_id: str, user_id: str, agent_id: str,
                                  generation: int, input_digest: str) -> dict[str, Any]:
        """Atomically claim a single-use replay authorization for one chunk.

        The row moves issued -> executing before any backend call, so a second
        caller cannot obtain the same grant. Scope, generation and input
        identity are re-checked here and in the adapter; a mismatch is recorded
        and refused rather than silently allowed.
        """
        # A refusal must be durable. Raising inside the write transaction would
        # roll the audit row back with it, so the claim is resolved to a value
        # first and the audit is committed in its own transaction afterwards.
        refusal: tuple[str, str, str] | None = None
        result: dict[str, Any] | None = None
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM chunk_replay_auth WHERE chunk_id=?', (chunk_id,)).fetchone()
            if row is None:
                refusal = ('no authorization was issued for this chunk', '', input_digest)
            elif row['status'] != 'issued':
                refusal = (f"authorization already {row['status']}", row['job_id'], row['input_digest'])
            elif (row['user_id'], row['agent_id']) != (user_id, agent_id):
                refusal = ('authorization belongs to another user/agent scope', row['job_id'], row['input_digest'])
            elif row['generation'] != generation:
                refusal = ('authorization belongs to a superseded memory generation',
                           row['job_id'], row['input_digest'])
            elif row['input_digest'] != input_digest:
                refusal = ('chunk input changed since the authorization was issued',
                           row['job_id'], row['input_digest'])
            else:
                changed = conn.execute("UPDATE chunk_replay_auth SET status='executing' "
                                       "WHERE chunk_id=? AND status='issued'", (chunk_id,)).rowcount
                if changed != 1:
                    refusal = ('authorization claimed concurrently', row['job_id'], row['input_digest'])
                else:
                    conn.execute('INSERT INTO chunk_replay_audit(chunk_id,job_id,event,user_id,agent_id,'
                                 "generation,input_digest,at,detail) "
                                 "VALUES(?,?,'claimed',?,?,?,?,?,'single-use grant claimed')",
                                 (chunk_id, row['job_id'], user_id, agent_id, generation,
                                  row['input_digest'], time.time()))
                    result = {'chunk_id': chunk_id, 'job_id': row['job_id'], 'generation': generation,
                              'attempt': row['attempt'] + 1, 'digest': row['input_digest'],
                              'user_id': user_id, 'agent_id': agent_id}
        if refusal is not None:
            reason, refused_job, refused_digest = refusal
            with self._connect(write=True) as conn:
                conn.execute('INSERT INTO chunk_replay_audit(chunk_id,job_id,event,user_id,agent_id,'
                             "generation,input_digest,at,detail) VALUES(?,?,'refused',?,?,?,?,?,?)",
                             (chunk_id, refused_job, user_id, agent_id, generation,
                              refused_digest, time.time(), reason))
            raise MemoryWriteUncertainError('Replay refused: ' + reason)
        return result

    def record_replay_outcome(self, *, chunk_id: str, success: bool, detail: str = '') -> None:
        """Consume the grant on success; on failure leave it burnt, never reusable."""
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT * FROM chunk_replay_auth WHERE chunk_id=?', (chunk_id,)).fetchone()
            if row is None:
                raise MemoryWriteUncertainError('Replay authorization disappeared mid-flight')
            if success:
                conn.execute("UPDATE chunk_replay_auth SET status='consumed',consumed_at=?,attempt=attempt+1,"
                             'last_error=NULL WHERE chunk_id=?', (time.time(), chunk_id))
                event = 'consumed'
            else:
                conn.execute("UPDATE chunk_replay_auth SET attempt=attempt+1,last_error=? WHERE chunk_id=?",
                             (detail[:500], chunk_id))
                event = 'failed'
            conn.execute('INSERT INTO chunk_replay_audit(chunk_id,job_id,event,user_id,agent_id,generation,'
                         "input_digest,at,detail) VALUES(?,?,?,?,?,?,?,?,?)",
                         (chunk_id, row['job_id'], event, row['user_id'], row['agent_id'],
                          row['generation'], row['input_digest'], time.time(), detail[:500]))

    def replay_authorization(self, chunk_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM chunk_replay_auth WHERE chunk_id=?', (chunk_id,)).fetchone()
            return dict(row) if row else None

    def replay_audit(self, *, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                'SELECT * FROM chunk_replay_audit WHERE job_id=? ORDER BY audit_id', (job_id,))]

    def retry_job(self, job_id: str, *, user_id: str, agent_id: str, max_attempts: int = 3,
                  allow_replay: bool = False) -> MemoryJob:
        """重试一个失败的作业。

        对于有不确定写入结果（writing/uncertain chunk）的作业，
        需要显式设置 allow_replay=True 才能重试。

        参数:
            job_id:       作业 ID
            user_id:      用户 ID
            agent_id:     Agent ID
            max_attempts: 追加的最大重试次数
            allow_replay: 是否允许重放不确定的写入
        返回:
            更新后的 MemoryJob
        """
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
                # Per-chunk, scope-bound, single-use grants persisted with the reset.
                self._grant_replays(conn, row, scope)
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