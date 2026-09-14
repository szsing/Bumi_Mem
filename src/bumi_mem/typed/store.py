"""Manual typed memory foundation; no LLM, embedding, vector store or legacy writes."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Callable
from uuid import uuid4
from pydantic import validate_call, ConfigDict

from .errors import AccessDenied, ConflictError, EvidenceError, SchemaMismatch, StaleRead
from .models import (Identifier, Epochs, GrantInput, MemoryDraft, ObservationInput, PAYLOADS,
                     ReadTicket, Scope, SpanInput, ViewDraft, VisibleView, iso, timestamp)
from .policy import PolicyEngine, Requirement, Rule
from .schema import APPLICATION_ID, SCHEMA, SCHEMA_VERSION


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode('utf8')).hexdigest()


class TypedMemoryStore:
    """Trusted in-process API. Scope identity must come from authenticated code.

    create_principal/create_space are provisioning APIs, never model tools.
    """
    retention_seconds = 14 * 24 * 3600

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time):
        original = Path(path).expanduser()
        if original.is_symlink():
            raise SchemaMismatch('database path must not be a symlink')
        self.path = original.resolve()
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        self._initialize()

    def _now(self):
        value = float(self.clock())
        if not math.isfinite(value):
            raise ValueError('invalid server clock')
        return value

    def _open(self):
        c = sqlite3.connect(self.path, timeout=2.0)
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA foreign_keys=ON')
        c.execute('PRAGMA synchronous=FULL')
        return c

    @staticmethod
    def _check_header(c):
        app = c.execute('PRAGMA application_id').fetchone()[0]
        version = c.execute('PRAGMA user_version').fetchone()[0]
        tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        if app == APPLICATION_ID and version == SCHEMA_VERSION:
            import re
            expected = set(re.findall(r'CREATE TABLE (\w+)', SCHEMA))
            if not expected <= set(tables):
                raise SchemaMismatch('typed schema is incomplete')
            return True
        if app or version or tables:
            raise SchemaMismatch('not a supported Stage-1 database; legacy/foreign/newer databases are not migrated')
        return False

    def _initialize(self):
        c = self._open()
        try:
            self._check_header(c)
            if c.execute('PRAGMA journal_mode=WAL').fetchone()[0].lower() != 'wal':
                raise SchemaMismatch('WAL is required')
            c.execute('BEGIN IMMEDIATE')
            if not self._check_header(c):
                for statement in SCHEMA.split(';'):
                    if statement.strip(): c.execute(statement)
                c.execute(f'PRAGMA application_id={APPLICATION_ID}')
                c.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    @contextmanager
    def _transaction(self, write=False):
        c = self._open()
        try:
            c.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    @staticmethod
    def _member(c, space, person):
        return c.execute("SELECT 1 FROM memberships WHERE space_id=? AND person_id=? AND status='active'", (space, person)).fetchone() is not None

    def _context_ok(self, c, ctx):
        return (ctx.identity_status == 'verified_by_upstream' and ctx.actor_id is not None
                and ctx.audience_confirmed and bool(ctx.audience_ids)
                and all(self._member(c, ctx.space_id, p) for p in ctx.recipients))

    def _auth(self, c, ctx):
        if not self._context_ok(c, ctx):
            raise AccessDenied('identity, audience or membership unavailable')

    def _admin(self, c, ctx):
        self._auth(c, ctx)
        row = c.execute('SELECT owner_id FROM spaces WHERE space_id=?', (ctx.space_id,)).fetchone()
        if row is None or row['owner_id'] != ctx.actor_id:
            raise AccessDenied('space administration is not authorized')

    @staticmethod
    def _epoch(c, space):
        r = c.execute('SELECT content,policy,deletion FROM scope_epochs WHERE space_id=?', (space,)).fetchone()
        if r is None: raise AccessDenied('scope unavailable')
        return Epochs(content=r['content'], policy=r['policy'], deletion=r['deletion'])

    def _bump(self, c, space, *, policy=False, deletion=False):
        c.execute('UPDATE scope_epochs SET content=content+1,policy=policy+?,deletion=deletion+? WHERE space_id=?', (int(policy), int(deletion), space))

    def _audit(self, c, ctx, operation, object_id):
        c.execute('INSERT INTO audit_events(space_id,actor_id,operation,object_id,created_at) VALUES(?,?,?,?,?)', (ctx.space_id, ctx.actor_id, operation, object_id, self._now()))

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def create_principal(self, person_id: str, display_name: str):
        from pydantic import TypeAdapter
        from .models import Identifier
        TypeAdapter(Identifier).validate_python(person_id, strict=True)
        if not display_name.strip(): raise ValueError('display name is required')
        with self._transaction(True) as c:
            c.execute('INSERT INTO principals VALUES(?,?)', (person_id, display_name))

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def create_space(self, space_id: str, owner_id: str):
        from pydantic import TypeAdapter
        from .models import Identifier
        TypeAdapter(Identifier).validate_python(space_id, strict=True)
        with self._transaction(True) as c:
            c.execute('INSERT INTO spaces VALUES(?,?,?)', (space_id, owner_id, self._now()))
            c.execute("INSERT INTO memberships VALUES(?,?,'active')", (space_id, owner_id))
            c.execute('INSERT INTO scope_epochs(space_id) VALUES(?)', (space_id,))

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def add_member(self, ctx: Scope, person_id: str):
        with self._transaction(True) as c:
            self._admin(c, ctx)
            c.execute("INSERT INTO memberships VALUES(?,?,'active') ON CONFLICT(space_id,person_id) DO UPDATE SET status='active'", (ctx.space_id, person_id))
            self._bump(c, ctx.space_id, policy=True)
            self._audit(c, ctx, 'add_member', person_id)

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def remove_member(self, ctx: Scope, person_id: str):
        with self._transaction(True) as c:
            self._admin(c, ctx)
            if ctx.actor_id == person_id: raise ConflictError('space owner cannot remove itself')
            c.execute("UPDATE memberships SET status='removed' WHERE space_id=? AND person_id=?", (ctx.space_id, person_id))
            for rule in c.execute("SELECT grant_id,subject_id,audience_json FROM grants WHERE space_id=? AND status='active'", (ctx.space_id,)).fetchall():
                if rule['subject_id'] == person_id or person_id in json.loads(rule['audience_json']):
                    c.execute("UPDATE grants SET status='revoked' WHERE grant_id=?", (rule['grant_id'],))
            c.execute("UPDATE memory_views SET status='invalidated' WHERE memory_id IN (SELECT memory_id FROM memories WHERE space_id=?) AND projection_type<>'private'", (ctx.space_id,))
            self._bump(c, ctx.space_id, policy=True)
            self._audit(c, ctx, 'remove_member', person_id)

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def add_relation(self, ctx: Scope, source_id: str, target_id: str, relation: Identifier):
        if not relation or len(relation) > 80: raise ValueError('relation name is required')
        with self._transaction(True) as c:
            self._admin(c, ctx)
            if not all(self._member(c, ctx.space_id, p) for p in (source_id, target_id)):
                raise AccessDenied('relation members unavailable')
            c.execute('INSERT OR IGNORE INTO relations VALUES(?,?,?,?)', (ctx.space_id, source_id, target_id, relation))
            self._bump(c, ctx.space_id, policy=True)

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def relations_for(self, ctx: Scope, source_id: str):
        with self._transaction() as c:
            if not self._context_ok(c, ctx) or not self._member(c, ctx.space_id, source_id): return []
            return [dict(r) for r in c.execute("SELECT source_id,target_id,relation FROM relations r JOIN memberships m ON m.space_id=r.space_id AND m.person_id=r.target_id WHERE r.space_id=? AND r.source_id=? AND m.status='active' ORDER BY relation,target_id", (ctx.space_id, source_id))]

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def create_conversation(self, ctx: Scope, conversation_id: Identifier):
        with self._transaction(True) as c:
            self._auth(c, ctx)
            c.execute('INSERT INTO conversations VALUES(?,?,?,?)', (conversation_id, ctx.space_id, ctx.actor_id, self._now()))
            for person in ctx.recipients:
                c.execute('INSERT INTO conversation_members VALUES(?,?)', (conversation_id, person))

    @staticmethod
    def _conversation_access(c, ctx, conversation_id):
        row = c.execute('SELECT space_id FROM conversations WHERE conversation_id=?', (conversation_id,)).fetchone()
        if row is None or row['space_id'] != ctx.space_id:
            return False
        participants = {r[0] for r in c.execute('SELECT person_id FROM conversation_members WHERE conversation_id=?', (conversation_id,))}
        return ctx.recipients <= participants

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def add_conversation_participant(self, ctx: Scope, conversation_id: str, person_id: str):
        with self._transaction(True) as c:
            self._auth(c, ctx)
            conv = c.execute('SELECT * FROM conversations WHERE conversation_id=?', (conversation_id,)).fetchone()
            if conv is None or conv['space_id'] != ctx.space_id or conv['owner_id'] != ctx.actor_id or not self._member(c, ctx.space_id, person_id):
                raise AccessDenied('conversation invitation is not authorized')
            c.execute('INSERT OR IGNORE INTO conversation_members VALUES(?,?)', (conversation_id, person_id))
            self._bump(c, ctx.space_id, policy=True)

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def record_observation(self, ctx: Scope, obs: ObservationInput):
        if not obs.is_final or not obs.storage_allowed or ctx.identity_status == 'unknown' or not ctx.audience_confirmed:
            return {'status': 'transient', 'persisted': False, 'observation_id': obs.observation_id}
        with self._transaction(True) as c:
            self._auth(c, ctx)
            conv = c.execute('SELECT * FROM conversations WHERE conversation_id=?', (obs.conversation_id,)).fetchone()
            if conv is None or not self._conversation_access(c, ctx, obs.conversation_id): raise AccessDenied('conversation unavailable')
            head = c.execute('SELECT * FROM observations WHERE observation_id=?', (obs.observation_id,)).fetchone()
            text_hash = digest(obs.text)
            if head:
                if head['space_id'] != ctx.space_id or head['owner_id'] != ctx.actor_id or head['conversation_id'] != obs.conversation_id or head['role'] != obs.role or head['status'] != 'active' or head['expires_at'] <= self._now():
                    raise AccessDenied('observation unavailable')
                existing = c.execute('SELECT * FROM observation_versions WHERE observation_id=? AND revision=?', (obs.observation_id, obs.revision)).fetchone()
                if existing:
                    if obs.revision != head['current_revision']: raise ConflictError('superseded input cannot be replayed')
                    if existing['content_hash'] != text_hash or existing['source_timestamp'] != timestamp(obs.source_timestamp):
                        raise ConflictError('same observation revision cannot change content/time')
                    return {'status': 'already_persisted', 'persisted': True, 'observation_id': obs.observation_id}
                if obs.revision != head['current_revision'] + 1: raise ConflictError('revision must advance by one')
                affected = [(r[0], r[1]) for r in c.execute('SELECT DISTINCT v.memory_id,v.version FROM version_spans v JOIN source_spans s USING(span_id) WHERE s.observation_id=?', (obs.observation_id,))]
                self._invalidate_versions(c, affected)
                c.execute("UPDATE observation_versions SET status='superseded' WHERE observation_id=?", (obs.observation_id,))
                c.execute('UPDATE observations SET current_revision=? WHERE observation_id=?', (obs.revision, obs.observation_id))
            else:
                if obs.revision != 1: raise ConflictError('first revision must be 1')
                speaker = ctx.actor_id if obs.role == 'user' else 'agent:bumi'
                c.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?)', (obs.observation_id, obs.conversation_id, ctx.space_id, ctx.actor_id, speaker, obs.role, encode(sorted(ctx.recipients)), 1, 'active', self._now(), self._now()+self.retention_seconds))
            c.execute("INSERT INTO observation_versions VALUES(?,?,?,?,?, 'active')", (obs.observation_id, obs.revision, obs.text, text_hash, timestamp(obs.source_timestamp)))
            seq = c.execute('SELECT COALESCE(MAX(sequence_no),0)+1 FROM conversation_turns WHERE conversation_id=?', (obs.conversation_id,)).fetchone()[0]
            turn_id = str(uuid4())
            c.execute('INSERT INTO conversation_turns VALUES(?,?,?,?,?,?,?)', (turn_id, obs.conversation_id, seq, obs.observation_id, obs.revision, 0, None))
            self._bump(c, ctx.space_id)
            self._audit(c, ctx, 'record_observation', obs.observation_id)
            return {'status': 'persisted', 'persisted': True, 'observation_id': obs.observation_id, 'turn_id': turn_id}

    def _observation(self, c, obs_id, revision):
        return c.execute('SELECT o.*,v.revision,v.text,v.source_timestamp,v.status AS revision_status FROM observations o JOIN observation_versions v USING(observation_id) WHERE o.observation_id=? AND v.revision=?', (obs_id, revision)).fetchone()

    def _source_live(self, row):
        return row is not None and row['status'] == 'active' and row['revision_status'] == 'active' and row['revision'] == row['current_revision'] and row['expires_at'] > self._now()

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def add_span(self, ctx: Scope, span: SpanInput):
        with self._transaction(True) as c:
            self._auth(c, ctx)
            row = self._observation(c, span.observation_id, span.revision)
            if not self._source_live(row) or row['space_id'] != ctx.space_id or row['owner_id'] != ctx.actor_id:
                raise EvidenceError('source unavailable or not owned by writer')
            if span.end > len(row['text']) or row['text'][span.start:span.end] != span.quote:
                raise EvidenceError('quote does not match Unicode source span')
            existing = c.execute('SELECT span_id FROM source_spans WHERE observation_id=? AND revision=? AND start_offset=? AND end_offset=?', (span.observation_id, span.revision, span.start, span.end)).fetchone()
            if existing: return existing['span_id']
            sid = str(uuid4())
            c.execute('INSERT INTO source_spans VALUES(?,?,?,?,?,?)', (sid, span.observation_id, span.revision, span.start, span.end, digest(span.quote)))
            return sid

    @staticmethod
    def _flat(memory, version):
        return {**json.loads(version['payload_json']), 'subject_ids': json.loads(memory['subjects_json'])}

    def _node(self, c, memory_id, version):
        m = c.execute('SELECT * FROM memories WHERE memory_id=?', (memory_id,)).fetchone()
        v = c.execute('SELECT * FROM memory_versions WHERE memory_id=? AND version=?', (memory_id, version)).fetchone()
        if m is None or v is None: raise EvidenceError('memory evidence unavailable')
        return m, v

    def _lineage(self, c, memory_id, version, attributes, visited=None):
        visited = set() if visited is None else set(visited)
        if len(visited) >= 64: raise EvidenceError('evidence depth exceeds Stage-1 limit')
        if (memory_id, version) in visited: raise EvidenceError('cyclic evidence')
        visited.add((memory_id, version))
        m, v = self._node(c, memory_id, version)
        if m['state'] in ('deleted', 'contested') or m['expires_at'] <= self._now() or v['validity_status'] != 'valid':
            raise EvidenceError('evidence has been withdrawn, corrected or expired')
        flat = self._flat(m, v)
        if not set(attributes) <= set(flat): raise EvidenceError('unknown source attributes')
        subjects = frozenset(json.loads(m['subjects_json']))
        if not all(self._member(c, m['space_id'], p) for p in subjects | {m['owner_id']}):
            raise EvidenceError('source membership unavailable')
        spans = c.execute('SELECT s.* FROM version_spans v JOIN source_spans s USING(span_id) WHERE v.memory_id=? AND v.version=?', (memory_id, version)).fetchall()
        for span in spans:
            row = self._observation(c, span['observation_id'], span['revision'])
            if not self._source_live(row): raise EvidenceError('source observation is no longer valid')
        result = [Requirement(memory_id, m['owner_id'], subjects, frozenset(attributes))]
        parents = c.execute('SELECT * FROM parent_edges WHERE memory_id=? AND version=?', (memory_id, version)).fetchall()
        if not spans and not parents: raise EvidenceError('memory has no evidence')
        for p in parents:
            result.extend(self._lineage(c, p['parent_id'], p['parent_version'], json.loads(p['attributes_json']), visited))
        return result

    def _has_user_source(self, c, memory_id, version, visited=None):
        visited = set() if visited is None else set(visited)
        if (memory_id, version) in visited:
            return False
        visited.add((memory_id, version))
        rows = c.execute('SELECT o.role FROM version_spans v JOIN source_spans s USING(span_id) JOIN observations o USING(observation_id) WHERE v.memory_id=? AND v.version=?', (memory_id, version))
        if any(r['role'] == 'user' for r in rows):
            return True
        return any(self._has_user_source(c, p['parent_id'], p['parent_version'], visited) for p in c.execute('SELECT parent_id,parent_version FROM parent_edges WHERE memory_id=? AND version=?', (memory_id, version)))

    def _validate_draft(self, c, ctx, draft):
        if not all(self._member(c, ctx.space_id, p) for p in draft.subject_ids): raise EvidenceError('unknown subject')
        expires = self._now()+self.retention_seconds
        user_evidence = False
        for sid in set(draft.span_ids):
            span = c.execute('SELECT * FROM source_spans WHERE span_id=?', (sid,)).fetchone()
            if span is None: raise EvidenceError('unknown span')
            row = self._observation(c, span['observation_id'], span['revision'])
            if not self._source_live(row) or row['space_id'] != ctx.space_id or row['owner_id'] != ctx.actor_id:
                raise EvidenceError('foreign or invalid source span')
            expires = min(expires, row['expires_at'])
            user_evidence |= row['role'] == 'user'
        for p in draft.parent_memories:
            m, v = self._node(c, p.memory_id, p.version)
            if m['space_id'] != ctx.space_id or m['owner_id'] != ctx.actor_id or p.version != m['current_version']:
                raise EvidenceError('parent must be current and owned by writer')
            nodes = self._lineage(c, p.memory_id, p.version, p.attributes)
            if any(n.memory_id == draft.memory_id for n in nodes): raise EvidenceError('self-reinforcing evidence cycle')
            if not set().union(*(set(n.subjects) for n in nodes)) <= set(draft.subject_ids):
                raise EvidenceError('derived memory cannot drop involved subjects')
            expires = min(expires, m['expires_at'])
            user_evidence |= self._has_user_source(c, p.memory_id, p.version)
        if not user_evidence and draft.kind != 'record_index':
            if not (draft.kind == 'episode' and draft.payload.get('event_type') == 'assistant_commitment' and draft.assertion_kind == 'assistant_commitment'):
                raise EvidenceError('assistant-only content is only an assistant commitment or source record')
        payload = PAYLOADS[draft.kind].model_validate(draft.payload).model_dump()
        key = encode([sorted(draft.subject_ids), payload['predicate'], payload['context']]) if draft.kind == 'identity' else None
        return encode(payload), key, expires

    def _put_version(self, c, draft, version, payload, start, reason):
        if draft.valid_until is not None and timestamp(draft.valid_until) <= start:
            raise ConflictError('effective validity interval is empty')
        c.execute('INSERT INTO memory_versions VALUES(?,?,?,?,?,?,?,?,?)', (draft.memory_id, version, payload, draft.assertion_kind, 'valid', start, timestamp(draft.valid_until), self._now(), reason))
        for sid in set(draft.span_ids): c.execute('INSERT INTO version_spans VALUES(?,?,?)', (draft.memory_id, version, sid))
        for p in draft.parent_memories:
            c.execute('INSERT INTO parent_edges VALUES(?,?,?,?,?)', (draft.memory_id, version, p.memory_id, p.version, encode(sorted(set(p.attributes)))))

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def create_memory(self, ctx: Scope, draft: MemoryDraft):
        with self._transaction(True) as c:
            self._auth(c, ctx)
            if c.execute('SELECT 1 FROM memories WHERE memory_id=?', (draft.memory_id,)).fetchone(): raise ConflictError('memory ID already exists')
            payload, key, expires = self._validate_draft(c, ctx, draft)
            if key and c.execute("SELECT 1 FROM memories WHERE space_id=? AND owner_id=? AND fact_key=? AND state IN ('active','candidate') AND expires_at>?", (ctx.space_id, ctx.actor_id, key, self._now())).fetchone():
                raise ConflictError('identity field exists; use explicit version update')
            c.execute('INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?)', (draft.memory_id, ctx.space_id, ctx.actor_id, draft.kind, encode(sorted(draft.subject_ids)), key, 1, draft.status, self._now(), expires))
            self._put_version(c, draft, 1, payload, timestamp(draft.valid_from) if draft.valid_from else self._now(), 'create')
            self._bump(c, ctx.space_id)
            self._audit(c, ctx, 'create_memory', draft.memory_id)
            return 1

    def _owned_memory(self, c, ctx, memory_id):
        self._auth(c, ctx)
        m = c.execute('SELECT * FROM memories WHERE memory_id=?', (memory_id,)).fetchone()
        if m is None or m['space_id'] != ctx.space_id or m['owner_id'] != ctx.actor_id or m['state'] == 'deleted' or m['expires_at'] <= self._now():
            raise AccessDenied('memory unavailable')
        return m

    def _invalidate_versions(self, c, seeds):
        affected = set(seeds)
        frontier = set(seeds)
        while frontier:
            children = set()
            for mid, version in frontier:
                children.update((r[0], r[1]) for r in c.execute('SELECT memory_id,version FROM parent_edges WHERE parent_id=? AND parent_version=?', (mid, version)))
            frontier = children - affected
            affected.update(frontier)
        for mid, version in affected:
            c.execute("UPDATE memory_versions SET validity_status='withdrawn' WHERE memory_id=? AND version=?", (mid, version))
            c.execute("UPDATE memory_views SET status='invalidated',view_revision=view_revision+1 WHERE memory_id=? AND source_version=? AND status='active'", (mid, version))
            c.execute("UPDATE memories SET state='contested' WHERE memory_id=? AND current_version=? AND state<>'deleted'", (mid, version))

    def _invalidate(self, c, memory_ids, *, roots):
        seeds = set()
        for mid in memory_ids:
            c.execute("UPDATE memory_views SET status='invalidated',view_revision=view_revision+1 WHERE memory_id=? AND status='active'", (mid,))
            if roots:
                seeds.update((mid, r[0]) for r in c.execute('SELECT version FROM memory_versions WHERE memory_id=?', (mid,)))
            else:
                seeds.update((r[0], r[1]) for r in c.execute('SELECT memory_id,version FROM parent_edges WHERE parent_id=?', (mid,)))
        self._invalidate_versions(c, seeds)

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def revise_memory(self, ctx: Scope, draft: MemoryDraft, *, expected_version: int, reason: str):
        if reason not in ('temporal_update', 'correction'): raise ValueError('explicit revision reason required')
        with self._transaction(True) as c:
            m = self._owned_memory(c, ctx, draft.memory_id)
            if m['current_version'] != expected_version: raise ConflictError('stale expected version')
            if m['kind'] != draft.kind or json.loads(m['subjects_json']) != sorted(draft.subject_ids):
                raise ConflictError('kind/subjects cannot be moved across a memory ID in Stage 1')
            payload, key, expires = self._validate_draft(c, ctx, draft)
            if key != m['fact_key']: raise ConflictError('identity predicate/context cannot change')
            old = c.execute('SELECT * FROM memory_versions WHERE memory_id=? AND version=?', (draft.memory_id, expected_version)).fetchone()
            start = timestamp(draft.valid_from) if draft.valid_from else (self._now() if reason == 'temporal_update' else old['valid_from'])
            if reason == 'temporal_update':
                if start <= old['valid_from']: raise ConflictError('temporal update must advance validity')
                end = min(old['valid_until'], start) if old['valid_until'] is not None else start
                c.execute('UPDATE memory_versions SET valid_until=? WHERE memory_id=? AND version=?', (end, draft.memory_id, expected_version))
            else:
                c.execute("UPDATE memory_versions SET validity_status='corrected' WHERE memory_id=? AND version=?", (draft.memory_id, expected_version))
            self._invalidate(c, [draft.memory_id], roots=False)
            version = expected_version+1
            c.execute('UPDATE memories SET current_version=?,state=?,expires_at=? WHERE memory_id=?', (version, draft.status, min(expires, m['expires_at']), draft.memory_id))
            self._put_version(c, draft, version, payload, start, reason)
            self._bump(c, ctx.space_id)
            self._audit(c, ctx, reason, draft.memory_id)
            return version

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def delete_memory(self, ctx: Scope, memory_id: str):
        """Logical semantic withdrawal. Raw source withdrawal is a distinct operation."""
        with self._transaction(True) as c:
            self._auth(c, ctx)
            m = c.execute('SELECT * FROM memories WHERE memory_id=?', (memory_id,)).fetchone()
            if m is None or m['space_id'] != ctx.space_id or m['owner_id'] != ctx.actor_id:
                raise AccessDenied('memory unavailable')
            if m['state'] == 'deleted':
                return False
            self._invalidate(c, [memory_id], roots=True)
            c.execute("UPDATE memories SET state='deleted' WHERE memory_id=?", (memory_id,))
            self._bump(c, ctx.space_id, deletion=True)
            self._audit(c, ctx, 'delete_memory', memory_id)
            return True

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def withdraw_observation(self, ctx: Scope, observation_id: str):
        with self._transaction(True) as c:
            self._auth(c, ctx)
            o = c.execute('SELECT * FROM observations WHERE observation_id=?', (observation_id,)).fetchone()
            if o is None or o['space_id'] != ctx.space_id or o['owner_id'] != ctx.actor_id: raise AccessDenied('source unavailable')
            if o['status'] == 'withdrawn':
                return False
            c.execute("UPDATE observations SET status='withdrawn' WHERE observation_id=?", (observation_id,))
            c.execute("UPDATE observation_versions SET status='withdrawn' WHERE observation_id=?", (observation_id,))
            affected = [(r[0], r[1]) for r in c.execute('SELECT DISTINCT v.memory_id,v.version FROM version_spans v JOIN source_spans s USING(span_id) WHERE s.observation_id=?', (observation_id,))]
            self._invalidate_versions(c, affected)
            self._bump(c, ctx.space_id, deletion=True)
            self._audit(c, ctx, 'withdraw_observation', observation_id)
            return True

    def _rules(self, c, space):
        now = self._now()
        rows = c.execute("SELECT * FROM grants WHERE space_id=? AND status='active' AND starts_at<=? AND (expires_at IS NULL OR expires_at>?)", (space, now, now)).fetchall()
        return [Rule(r['subject_id'], frozenset(json.loads(r['audience_json'])), r['effect'], r['category'], frozenset(json.loads(r['attributes_json'])), frozenset(json.loads(r['purposes_json'])), r['target_memory_id'], bool(r['block_inference'])) for r in rows]

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def grant(self, ctx: Scope, grant: GrantInput):
        with self._transaction(True) as c:
            self._auth(c, ctx)
            if grant.subject_id != ctx.actor_id: raise AccessDenied('cannot grant on another subject behalf')
            if not all(self._member(c, ctx.space_id, p) for p in grant.audience_ids): raise AccessDenied('grant recipient unavailable')
            if grant.memory_id:
                m = c.execute('SELECT * FROM memories WHERE memory_id=?', (grant.memory_id,)).fetchone()
                if m is None or m['space_id'] != ctx.space_id or ctx.actor_id not in set(json.loads(m['subjects_json'])) | {m['owner_id']}:
                    raise AccessDenied('target memory unavailable')
            gid = str(uuid4())
            c.execute('INSERT INTO grants VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', (gid, ctx.space_id, grant.subject_id, encode(sorted(grant.audience_ids)), grant.effect, grant.category, encode(sorted(set(grant.attributes))), encode(sorted(set(grant.purposes))), grant.memory_id, int(grant.block_inference), timestamp(grant.starts_at) if grant.starts_at else self._now(), timestamp(grant.expires_at), 'active'))
            self._bump(c, ctx.space_id, policy=True)
            self._audit(c, ctx, 'grant_'+grant.effect, gid)
            return gid

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def revoke_grant(self, ctx: Scope, grant_id: str):
        with self._transaction(True) as c:
            self._auth(c, ctx)
            g = c.execute('SELECT * FROM grants WHERE grant_id=?', (grant_id,)).fetchone()
            if g is None or g['space_id'] != ctx.space_id or g['subject_id'] != ctx.actor_id: raise AccessDenied('grant unavailable')
            if g['status'] == 'revoked':
                return False
            c.execute("UPDATE grants SET status='revoked' WHERE grant_id=?", (grant_id,))
            c.execute("UPDATE memory_views SET status='invalidated',view_revision=view_revision+1 WHERE projection_type<>'private' AND memory_id IN (SELECT memory_id FROM memories WHERE space_id=?) AND status='active'", (ctx.space_id,))
            self._bump(c, ctx.space_id, policy=True)
            self._audit(c, ctx, 'revoke_grant', grant_id)
            return True

    @staticmethod
    def _category(projection):
        return {'private': 'private', 'availability': 'coordination.availability', 'shared_activity': 'coordination.shared_activity'}[projection]

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def create_view(self, ctx: Scope, draft: ViewDraft):
        with self._transaction(True) as c:
            self._owned_memory(c, ctx, draft.memory_id)
            m, v = self._node(c, draft.memory_id, draft.source_version)
            flat = self._flat(m, v)
            if not set(draft.attributes) <= set(flat): raise EvidenceError('unknown projection attributes')
            shared = draft.projection_type != 'private'
            if shared:
                allowed = {'availability', 'time', 'subject_ids'} if draft.projection_type == 'availability' else {'activity', 'time', 'event_status', 'subject_ids', 'location'}
                if m['kind'] != 'episode' or not set(draft.attributes) <= allowed:
                    raise AccessDenied('unsupported shared projection')
            elif set(draft.audience_ids) != {ctx.actor_id}:
                raise AccessDenied('private view is owner-only')
            if not all(self._member(c, ctx.space_id, p) for p in draft.audience_ids): raise AccessDenied('audience unavailable')
            requirements = self._lineage(c, draft.memory_id, draft.source_version, draft.attributes)
            if draft.snapshot_kind == 'current' and m['current_version'] != draft.source_version: raise ConflictError('current view must use current source version')
            if m['state'] == 'candidate' and (shared or set(draft.purposes) != {'explicit_memory_review'}): raise AccessDenied('candidate can only be privately reviewed')
            category = self._category(draft.projection_type)
            if shared and any(p != category for p in draft.purposes): raise AccessDenied('sharing purpose must match projection category')
            rules = self._rules(c, ctx.space_id)
            for purpose in draft.purposes:
                if not PolicyEngine.permits(requirements=requirements, rules=rules, recipients=set(draft.audience_ids), purpose=purpose, category=category, shared=shared):
                    raise AccessDenied('projection is not currently authorized')
            projected = {name: flat[name] for name in draft.attributes}
            # Availability is a claim about a plan, not a completed action.
            if m['kind'] == 'episode' and flat.get('event_status') in ('planned', 'tentative') and projected.get('availability') in ('busy', 'free'):
                projected['availability'] = 'possibly_' + projected['availability']
            vid = str(uuid4())
            c.execute('INSERT INTO memory_views VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', (vid, draft.memory_id, draft.source_version, 1, draft.snapshot_kind, draft.projection_type, encode(draft.attributes), encode(sorted(draft.audience_ids)), encode(sorted(draft.purposes)), encode(projected), encode(projected), 'active', self._now()))
            self._bump(c, ctx.space_id)
            self._audit(c, ctx, 'create_view', vid)
            return vid

    def _visible(self, c, ctx, view_id, mode, as_of):
        if not self._context_ok(c, ctx): return None
        view = c.execute('SELECT * FROM memory_views WHERE view_id=?', (view_id,)).fetchone()
        if view is None or view['status'] != 'active' or view['snapshot_kind'] != mode: return None
        try: m, v = self._node(c, view['memory_id'], view['source_version'])
        except EvidenceError: return None
        if m['space_id'] != ctx.space_id or ctx.purpose not in json.loads(view['purposes_json']): return None
        if not ctx.recipients <= set(json.loads(view['audience_json'])): return None
        shared = view['projection_type'] != 'private'
        if not shared and (ctx.actor_id != m['owner_id'] or ctx.recipients != {m['owner_id']}): return None
        if m['state'] == 'candidate' and (shared or ctx.purpose != 'explicit_memory_review'): return None
        if mode == 'current' and m['current_version'] != view['source_version']: return None
        business_time = self._now() if mode == 'current' else as_of
        if business_time is None or business_time < v['valid_from'] or (v['valid_until'] is not None and business_time >= v['valid_until']): return None
        try: requirements = self._lineage(c, m['memory_id'], v['version'], json.loads(view['attributes_json']))
        except EvidenceError: return None
        if not PolicyEngine.permits(requirements=requirements, rules=self._rules(c, ctx.space_id), recipients=ctx.recipients, purpose=ctx.purpose, category=self._category(view['projection_type']), shared=shared): return None
        return VisibleView(view_id=view['view_id'], view_revision=view['view_revision'], kind=m['kind'], snapshot_kind=mode, projected_fields=json.loads(view['projected_json']), display_text=view['display_text'], validity_status=v['validity_status'], assertion_kind=v['assertion_kind'], status=m['state'], valid_from=iso(v['valid_from']), valid_until=iso(v['valid_until']))

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def read_view(self, ctx: Scope, view_id: str, *, temporal_mode='current', as_of=None):
        if temporal_mode not in ('current', 'historical'): raise ValueError('invalid temporal mode')
        point = timestamp(as_of)
        if temporal_mode == 'historical' and point is None: raise ValueError('historical reads require as_of')
        with self._transaction() as c: return self._visible(c, ctx, view_id, temporal_mode, point)

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def list_views(self, ctx: Scope, *, temporal_mode='current', as_of=None):
        if temporal_mode not in ('current', 'historical'): raise ValueError('invalid temporal mode')
        point = timestamp(as_of)
        if temporal_mode == 'historical' and point is None: raise ValueError('historical reads require as_of')
        with self._transaction() as c:
            if not self._context_ok(c, ctx): return []
            ids = [r[0] for r in c.execute("SELECT v.view_id FROM memory_views v JOIN memories m USING(memory_id) WHERE m.space_id=? AND v.status='active' ORDER BY v.created_at,v.view_id", (ctx.space_id,))]
            return [value for vid in ids if (value := self._visible(c, ctx, vid, temporal_mode, point)) is not None]

    @staticmethod
    def _fingerprint(ctx, mode, as_of):
        value = ctx.model_dump()
        value['audience_ids'] = sorted(value['audience_ids'])
        return digest(encode({'context': value, 'mode': mode, 'as_of': iso(timestamp(as_of))}))

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def prepare_read(self, ctx: Scope, *, temporal_mode='current', as_of=None):
        if temporal_mode not in ('current', 'historical'): raise ValueError('invalid temporal mode')
        point = timestamp(as_of)
        if temporal_mode == 'historical' and point is None: raise ValueError('historical reads require as_of')
        with self._transaction() as c:
            self._auth(c, ctx)
            ids = [r[0] for r in c.execute("SELECT v.view_id FROM memory_views v JOIN memories m USING(memory_id) WHERE m.space_id=? AND v.status='active'", (ctx.space_id,))]
            selected = [vid for vid in ids if self._visible(c, ctx, vid, temporal_mode, point) is not None]
            return ReadTicket(context_fingerprint=self._fingerprint(ctx, temporal_mode, as_of), epochs=self._epoch(c, ctx.space_id), view_ids=selected, temporal_mode=temporal_mode, as_of=iso(point))

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def materialize(self, ctx: Scope, ticket: ReadTicket):
        with self._transaction() as c:
            self._auth(c, ctx)
            if ticket.context_fingerprint != self._fingerprint(ctx, ticket.temporal_mode, ticket.as_of) or ticket.epochs != self._epoch(c, ctx.space_id):
                raise StaleRead('context or permission/content generation changed')
            values = [self._visible(c, ctx, vid, ticket.temporal_mode, timestamp(ticket.as_of)) for vid in ticket.view_ids]
            if any(v is None for v in values): raise StaleRead('a source/view is no longer valid')
            return values

    def _raw_visible(self, c, ctx, row, *, context=False):
        if not self._context_ok(c, ctx) or not self._source_live(row) or row['space_id'] != ctx.space_id: return False
        if context:
            if not ctx.recipients <= set(json.loads(row['audience_json'])): return False
        elif ctx.purpose != 'explicit_memory_review' or ctx.actor_id != row['owner_id'] or ctx.recipients != {row['owner_id']}:
            return False
        rules = self._rules(c, ctx.space_id)
        raw_requirement = Requirement(row['observation_id'], row['owner_id'], frozenset({row['owner_id']}), frozenset({'text'}))
        if not PolicyEngine.permits(requirements=[raw_requirement], rules=rules, recipients=ctx.recipients, purpose=ctx.purpose, category='private', shared=False): return False
        # Block whole raw turns conservatively when linked semantic data is denied
        # or withdrawn. This is not automatic text redaction or physical deletion.
        linked = c.execute('SELECT DISTINCT v.memory_id,v.version FROM version_spans v JOIN source_spans s USING(span_id) WHERE s.observation_id=? AND s.revision=?', (row['observation_id'], row['revision'])).fetchall()
        for item in linked:
            try:
                m, v = self._node(c, item['memory_id'], item['version'])
                requirements = self._lineage(c, m['memory_id'], v['version'], list(self._flat(m, v)))
            except EvidenceError: return False
            for category in ('private', 'coordination.availability', 'coordination.shared_activity'):
                if not PolicyEngine.permits(requirements=requirements, rules=rules, recipients=ctx.recipients, purpose=ctx.purpose, category=category, shared=False): return False
        return True

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def read_observation(self, ctx: Scope, observation_id: str, revision: int):
        with self._transaction() as c:
            row = self._observation(c, observation_id, revision)
            if not self._raw_visible(c, ctx, row): return None
            return {'observation_id': observation_id, 'revision': revision, 'text': row['text'], 'speaker_id': row['speaker_id'], 'role': row['role'], 'source_timestamp': iso(row['source_timestamp'])}

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def read_turns(self, ctx: Scope, conversation_id: str, *, limit=20):
        if not 1 <= limit <= 100: raise ValueError('limit must be 1..100')
        with self._transaction() as c:
            if not self._context_ok(c, ctx): return []
            conv = c.execute('SELECT space_id FROM conversations WHERE conversation_id=?', (conversation_id,)).fetchone()
            if conv is None or not self._conversation_access(c, ctx, conversation_id): return []
            output = []
            for turn in c.execute('SELECT * FROM conversation_turns WHERE conversation_id=? ORDER BY sequence_no DESC', (conversation_id,)):
                row = self._observation(c, turn['observation_id'], turn['revision'])
                if self._raw_visible(c, ctx, row, context=True):
                    output.append({'turn_id': turn['turn_id'], 'text': row['text'], 'speaker_id': row['speaker_id'], 'role': row['role'], 'observation_id': row['observation_id'], 'revision': row['revision'], 'context_only': bool(turn['context_only'])})
                    if len(output) >= limit: break
            return list(reversed(output))

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def carryover(self, ctx: Scope, turn_id: str, target_conversation_id: str):
        with self._transaction(True) as c:
            self._auth(c, ctx)
            turn = c.execute('SELECT * FROM conversation_turns WHERE turn_id=?', (turn_id,)).fetchone()
            target = c.execute('SELECT * FROM conversations WHERE conversation_id=?', (target_conversation_id,)).fetchone()
            if turn is None or target is None or not self._conversation_access(c, ctx, target_conversation_id) or not self._conversation_access(c, ctx, turn['conversation_id']): raise AccessDenied('turn or target unavailable')
            row = self._observation(c, turn['observation_id'], turn['revision'])
            if not self._raw_visible(c, ctx, row, context=True): raise AccessDenied('carry-over is not authorized')
            existing = c.execute('SELECT turn_id FROM conversation_turns WHERE conversation_id=? AND source_turn_id=?', (target_conversation_id, turn_id)).fetchone()
            if existing: return existing['turn_id']
            seq = c.execute('SELECT COALESCE(MAX(sequence_no),0)+1 FROM conversation_turns WHERE conversation_id=?', (target_conversation_id,)).fetchone()[0]
            tid = str(uuid4())
            c.execute('INSERT INTO conversation_turns VALUES(?,?,?,?,?,?,?)', (tid, target_conversation_id, seq, turn['observation_id'], turn['revision'], 1, turn_id))
            return tid

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def epochs(self, ctx: Scope):
        with self._transaction() as c:
            self._auth(c, ctx)
            return self._epoch(c, ctx.space_id)
