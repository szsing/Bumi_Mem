"""SQLite-backed L1 sessions and L2 consolidation metadata."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

from bumi_mem.schemas import MessageRecord, SessionBlockedError, SessionRecord, SessionStats

OPEN_STATUSES = ("active", "processing", "failed")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteSessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active','processing','failed','consolidated','discarded')),
                    started_at TEXT NOT NULL,
                    closed_at TEXT,
                    close_reason TEXT,
                    summary TEXT,
                    summary_provider TEXT,
                    summary_status TEXT NOT NULL DEFAULT 'pending',
                    summary_error TEXT,
                    consolidation_status TEXT NOT NULL DEFAULT 'pending',
                    consolidation_attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS one_open_session_per_pair
                ON sessions(user_id, agent_id)
                WHERE status IN ('active','processing','failed');

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    sequence_no INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    context_only INTEGER NOT NULL DEFAULT 0,
                    source_session_id TEXT,
                    created_at TEXT NOT NULL,
                    consolidated_at TEXT,
                    UNIQUE(session_id, sequence_no)
                );

                CREATE INDEX IF NOT EXISTS messages_by_session
                ON messages(session_id, sequence_no);

                CREATE TABLE IF NOT EXISTS consolidation_runs (
                    consolidation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    attempt INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending','processing','success','failed')),
                    started_at TEXT,
                    finished_at TEXT,
                    mem0_events_json TEXT,
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS consolidation_runs_by_session
                ON consolidation_runs(session_id, attempt, chunk_index);
                """
            )
            conn.execute(
                """
                UPDATE sessions
                SET status='failed', consolidation_status='failed',
                    last_error=COALESCE(last_error, 'Interrupted during consolidation')
                WHERE status='processing'
                """
            )

    @staticmethod
    def _session(row: sqlite3.Row | None) -> SessionRecord | None:
        if row is None:
            return None
        return SessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            status=row["status"],
            started_at=row["started_at"],
            closed_at=row["closed_at"],
            close_reason=row["close_reason"],
            summary=row["summary"],
            summary_provider=row["summary_provider"],
            summary_status=row["summary_status"],
            summary_error=row["summary_error"],
            consolidation_status=row["consolidation_status"],
            consolidation_attempts=int(row["consolidation_attempts"]),
            last_error=row["last_error"],
        )

    @staticmethod
    def _message(row: sqlite3.Row) -> MessageRecord:
        return MessageRecord(
            message_id=row["message_id"],
            session_id=row["session_id"],
            sequence_no=int(row["sequence_no"]),
            role=row["role"],
            content=row["content"],
            token_count=int(row["token_count"]),
            context_only=bool(row["context_only"]),
            source_session_id=row["source_session_id"],
            created_at=row["created_at"],
            consolidated_at=row["consolidated_at"],
        )

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return self._session(row)

    def get_open_session(self, user_id: str, agent_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM sessions
                WHERE user_id=? AND agent_id=? AND status IN ('active','processing','failed')
                ORDER BY created_at DESC LIMIT 1
                """,
                (user_id, agent_id),
            ).fetchone()
        return self._session(row)

    def create_session(
        self,
        user_id: str,
        agent_id: str,
        *,
        carryover: Sequence[MessageRecord] = (),
    ) -> SessionRecord:
        session_id = uuid4().hex
        now = utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(
                    session_id,user_id,agent_id,status,started_at,created_at
                ) VALUES(?,?,?,'active',?,?)
                """,
                (session_id, user_id, agent_id, now, now),
            )
            for sequence_no, message in enumerate(carryover, 1):
                conn.execute(
                    """
                    INSERT INTO messages(
                        message_id,session_id,sequence_no,role,content,token_count,
                        context_only,source_session_id,created_at
                    ) VALUES(?,?,?,?,?,?,1,?,?)
                    """,
                    (
                        uuid4().hex,
                        session_id,
                        sequence_no,
                        message.role,
                        message.content,
                        message.token_count,
                        message.session_id,
                        now,
                    ),
                )
        created = self.get_session(session_id)
        assert created is not None
        return created

    def get_or_create_session(self, user_id: str, agent_id: str) -> SessionRecord:
        session = self.get_open_session(user_id, agent_id)
        return session or self.create_session(user_id, agent_id)

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        token_count: int,
        context_only: bool = False,
        source_session_id: str | None = None,
    ) -> MessageRecord:
        content = content.strip()
        if not content:
            raise ValueError("message content cannot be empty")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown session: {session_id}")
            if row["status"] != "active":
                raise SessionBlockedError(
                    f"session {session_id} is {row['status']}; resolve consolidation first"
                )
            sequence_no = int(
                conn.execute(
                    "SELECT COALESCE(MAX(sequence_no),0)+1 FROM messages WHERE session_id=?",
                    (session_id,),
                ).fetchone()[0]
            )
            message_id = uuid4().hex
            now = utcnow()
            conn.execute(
                """
                INSERT INTO messages(
                    message_id,session_id,sequence_no,role,content,token_count,
                    context_only,source_session_id,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    message_id,
                    session_id,
                    sequence_no,
                    role,
                    content,
                    max(0, int(token_count)),
                    int(context_only),
                    source_session_id,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM messages WHERE message_id=?", (message_id,)
            ).fetchone()
        assert row is not None
        return self._message(row)

    def list_messages(
        self, session_id: str, *, include_context_only: bool = True
    ) -> list[MessageRecord]:
        query = "SELECT * FROM messages WHERE session_id=?"
        params: list[object] = [session_id]
        if not include_context_only:
            query += " AND context_only=0"
        query += " ORDER BY sequence_no"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._message(row) for row in rows]

    def effective_messages(self, session_id: str) -> list[MessageRecord]:
        return self.list_messages(session_id, include_context_only=False)

    def stats(self, session_id: str) -> SessionStats:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"unknown session: {session_id}")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN context_only=0 THEN 1 ELSE 0 END) AS effective_count,
                    SUM(CASE WHEN context_only=1 THEN 1 ELSE 0 END) AS context_count,
                    SUM(CASE WHEN context_only=0 THEN token_count ELSE 0 END) AS effective_tokens
                FROM messages WHERE session_id=?
                """,
                (session_id,),
            ).fetchone()
        return SessionStats(
            session_id=session_id,
            status=session.status,
            effective_messages=int(row["effective_count"] or 0),
            context_only_messages=int(row["context_count"] or 0),
            effective_tokens=int(row["effective_tokens"] or 0),
        )

    def begin_consolidation(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status,consolidation_attempts FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown session: {session_id}")
            if row["status"] not in {"active", "failed"}:
                raise SessionBlockedError(
                    f"session {session_id} cannot consolidate from {row['status']}"
                )
            attempt = int(row["consolidation_attempts"]) + 1
            conn.execute(
                """
                UPDATE sessions SET
                    status='processing', consolidation_status='processing',
                    consolidation_attempts=?, last_error=NULL,
                    summary=NULL, summary_provider=NULL, summary_status='pending',
                    summary_error=NULL
                WHERE session_id=?
                """,
                (attempt, session_id),
            )
        return attempt

    def save_summary(
        self,
        session_id: str,
        *,
        summary: str | None,
        provider: str,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions SET summary=?,summary_provider=?,summary_status=?,summary_error=?
                WHERE session_id=?
                """,
                (summary, provider, status, error, session_id),
            )

    def create_consolidation_runs(
        self,
        session_id: str,
        *,
        attempt: int,
        provider: str,
        chunk_count: int,
    ) -> list[str]:
        ids: list[str] = []
        with self._connect() as conn:
            for chunk_index in range(chunk_count):
                consolidation_id = f"{session_id}_attempt_{attempt}_chunk_{chunk_index}"
                ids.append(consolidation_id)
                conn.execute(
                    """
                    INSERT INTO consolidation_runs(
                        consolidation_id,session_id,attempt,chunk_index,chunk_count,provider,status
                    ) VALUES(?,?,?,?,?,?,'pending')
                    """,
                    (
                        consolidation_id,
                        session_id,
                        attempt,
                        chunk_index,
                        chunk_count,
                        provider,
                    ),
                )
        return ids

    def start_run(self, consolidation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE consolidation_runs SET status='processing',started_at=? WHERE consolidation_id=?",
                (utcnow(), consolidation_id),
            )

    def finish_run(
        self, consolidation_id: str, *, events_json: str = "[]", error: str | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE consolidation_runs SET status=?,finished_at=?,mem0_events_json=?,error=?
                WHERE consolidation_id=?
                """,
                ("failed" if error else "success", utcnow(), events_json, error, consolidation_id),
            )

    def fail_consolidation(self, session_id: str, error: str) -> None:
        """Fail atomically and discard an already-generated L2, per the design contract."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions SET
                    status='failed', consolidation_status='failed', last_error=?,
                    summary=NULL, summary_provider=NULL, summary_status='pending', summary_error=NULL
                WHERE session_id=?
                """,
                (error, session_id),
            )

    def complete_consolidation(
        self,
        session_id: str,
        *,
        reason: str,
        carryover: Sequence[MessageRecord],
    ) -> SessionRecord:
        old = self.get_session(session_id)
        if old is None:
            raise KeyError(f"unknown session: {session_id}")
        now = utcnow()
        new_session_id = uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "UPDATE messages SET consolidated_at=? WHERE session_id=? AND context_only=0",
                (now, session_id),
            )
            conn.execute(
                """
                UPDATE sessions SET status='consolidated',consolidation_status='success',
                    close_reason=?,closed_at=?,last_error=NULL
                WHERE session_id=?
                """,
                (reason, now, session_id),
            )
            conn.execute(
                """
                INSERT INTO sessions(session_id,user_id,agent_id,status,started_at,created_at)
                VALUES(?,?,?,'active',?,?)
                """,
                (new_session_id, old.user_id, old.agent_id, now, now),
            )
            for sequence_no, message in enumerate(carryover, 1):
                conn.execute(
                    """
                    INSERT INTO messages(
                        message_id,session_id,sequence_no,role,content,token_count,
                        context_only,source_session_id,created_at
                    ) VALUES(?,?,?,?,?,?,1,?,?)
                    """,
                    (
                        uuid4().hex,
                        new_session_id,
                        sequence_no,
                        message.role,
                        message.content,
                        message.token_count,
                        session_id,
                        now,
                    ),
                )
        created = self.get_session(new_session_id)
        assert created is not None
        return created

    def rotate_empty_session(self, session_id: str, *, reason: str) -> SessionRecord:
        old = self.get_session(session_id)
        if old is None:
            raise KeyError(f"unknown session: {session_id}")
        now = utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions SET status='consolidated',consolidation_status='success',
                    summary='Empty session',summary_status='fallback',close_reason=?,closed_at=?
                WHERE session_id=?
                """,
                (reason, now, session_id),
            )
        return self.create_session(old.user_id, old.agent_id)

    def discard_and_create(self, session_id: str, *, reason: str = "manual_discard") -> SessionRecord:
        old = self.get_session(session_id)
        if old is None:
            raise KeyError(f"unknown session: {session_id}")
        now = utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions SET status='discarded',consolidation_status='pending',
                    close_reason=?,closed_at=?,last_error=NULL
                WHERE session_id=?
                """,
                (reason, now, session_id),
            )
        return self.create_session(old.user_id, old.agent_id)

    def list_consolidation_runs(self, session_id: str) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM consolidation_runs WHERE session_id=? ORDER BY attempt,chunk_index",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]
