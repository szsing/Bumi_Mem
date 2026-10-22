# 本次改动（2026-10-21）：接入隔离配置、共同事实索引及查询，个人检索仅返回关系事实并复用查询向量。
"""Graphiti-native L3. Upstream extraction prompts are preserved; retrieval uses RRF.

The local journal records ingestion outcomes, not an alternative memory store.
An interrupted write is uncertain and is never automatically replayed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bumi_mem.isolation_graph import IsolationGraphMixin
from bumi_mem.control_graph import ControlGraphMixin
from bumi_mem.identity import require_pair
from bumi_mem.providers import ProviderConfig, bind_provider_request_options
from bumi_mem.schemas import (
    MemoryWriteUncertainError, RetryableMemoryError, canonical_messages_digest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphitiConfig:
    uri: str
    user: str
    password: str = field(repr=False)
    data_dir: Path
    embed_model: Path
    reranker_model: Path | None = None  # Deprecated; accepted for old configurations.
    database: str = 'neo4j'
    device: str = 'auto'
    embedding_dims: int = 1024
    community_every: int = 5
    max_coroutines: int = 3
    reranker_fp16: bool = True
    registered_isolation: bool = False


def graph_group(user_id: str, agent_id: str) -> str:
    require_pair(user_id, agent_id)
    return 'bumi_' + hashlib.sha256(json.dumps([user_id, agent_id]).encode()).hexdigest()


def rrf_search_config():
    """Same four hybrid routes and limits as before, without model reranking."""
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER
    from graphiti_core.search.search_config import (
        EdgeReranker, NodeReranker, EpisodeReranker, CommunityReranker,
    )
    config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    for field, enum in (
        ('edge_config', EdgeReranker), ('node_config', NodeReranker),
        ('episode_config', EpisodeReranker), ('community_config', CommunityReranker),
    ):
        getattr(config, field).reranker = enum.rrf
    return config


from bumi_mem.admin_memory import AdminMemoryMixin


class GraphitiLongTermMemory(IsolationGraphMixin, ControlGraphMixin, AdminMemoryMixin):
    backend_name = 'graphiti'

    def __init__(self, llm: ProviderConfig, config: GraphitiConfig):
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.config import LLMConfig
        from bumi_mem.graphiti_llm import ValidatedGraphitiClient
        from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
        from graphiti_core.cross_encoder.client import CrossEncoderClient
        from sentence_transformers import SentenceTransformer

        self.config = config
        self._write_lock = asyncio.Lock()
        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.journal = config.data_dir / 'graphiti-journal.sqlite3'
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS ingestions (
                    id TEXT PRIMARY KEY, group_id TEXT NOT NULL, state TEXT NOT NULL,
                    payload TEXT NOT NULL, result TEXT, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS community_jobs (
                    group_id TEXT PRIMARY KEY, built INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0,
                    error TEXT);
                CREATE TABLE IF NOT EXISTS replay_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ingestion_id TEXT NOT NULL, job_id TEXT NOT NULL, event TEXT NOT NULL,
                    at REAL NOT NULL, detail TEXT NOT NULL DEFAULT '');
            ''')
        device = None if config.device == 'auto' else config.device

        class LocalEmbedder(EmbedderClient):
            def __init__(self):
                self.config = EmbedderConfig(embedding_dim=config.embedding_dims)
                self.model = SentenceTransformer(str(config.embed_model), device=device, local_files_only=True)
                if self.model.get_sentence_embedding_dimension() != config.embedding_dims:
                    raise ValueError('Configured embedding dimension does not match local model')
                self.lock = asyncio.Lock()

            async def create_batch(self, input_data_list):
                async with self.lock:
                    return await asyncio.to_thread(self._encode, input_data_list)

            def _encode(self, texts):
                return self.model.encode(texts, normalize_embeddings=True).tolist()

            async def create(self, input_data):
                texts = [input_data] if isinstance(input_data, str) else list(input_data)
                return (await self.create_batch(texts))[0]

        class DisabledReranker(CrossEncoderClient):
            async def rank(self, query, passages):
                # Passing None would construct Graphiti's default API reranker.
                raise RuntimeError('Model reranking is disabled; use the RRF search configuration')

        self.llm_client = ValidatedGraphitiClient(
            config=LLMConfig(model=llm.model, small_model=llm.model,
                             api_key=llm.api_key, base_url=llm.base_url),
            structured_output_mode='json_object', diagnostic_dir=config.data_dir / 'llm-diagnostics')
        # Preserve the configured provider transport mode without changing any
        # upstream extraction prompt, response schema or generation method.
        completions = self.llm_client.client.chat.completions
        completions.create = bind_provider_request_options(completions.create, llm)
        self.client = Graphiti(
            config.uri, config.user, config.password,
            llm_client=self.llm_client,
            embedder=LocalEmbedder(), cross_encoder=DisabledReranker(),
            max_coroutines=config.max_coroutines,
        )
        # Database selection belongs to the Neo4j driver, not query text.
        if config.database != 'neo4j':
            raise ValueError('First release supports the neo4j database; use a separate Neo4j instance for isolation')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.journal, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    async def warmup(self, *, query: str):
        await self.client.build_indices_and_constraints()
        if self.config.registered_isolation:
            await self.isolation_prepare()
        await self.client.embedder.create(query)

    async def add_session(self, *, user_id, agent_id, messages, metadata, prompt=None,
                          replay_authorization=None):
        from graphiti_core.nodes import EpisodeType
        from graphiti_core.utils.bulk_utils import RawEpisode
        group = graph_group(user_id, agent_id)
        key = metadata['consolidation_id']
        if self.config.registered_isolation and user_id not in {u['user_id'] for u in metadata.get('registered_users',[]) if u.get('active',True)}:
            raise RetryableMemoryError('Owner is not registered; Graphiti write never started')
        stable_metadata = {k: v for k, v in metadata.items() if k not in ('consolidation_attempt','registered_users')}
        payload = json.dumps({'messages': messages, 'metadata': stable_metadata}, ensure_ascii=False, sort_keys=True)
        async with self._write_lock:
            with self._db() as db:
                previous = db.execute('SELECT state,payload,result FROM ingestions WHERE id=?', (key,)).fetchone()
                if previous:
                    if previous[1] != payload:
                        raise MemoryWriteUncertainError('Ingestion identity reused with different source data')
                    if previous[0] == 'complete':
                        # Already succeeded with identical input: return the original receipt.
                        return json.loads(previous[2])
                    # Incomplete write. A replay may proceed ONLY with a grant that
                    # the durable store issued for this exact chunk, scope and input.
                    self._assert_replay_grant(replay_authorization, key, user_id, agent_id, metadata, messages)
                    self._audit_replay(key, metadata, 'accepted', 'journal reopened under explicit grant')
                db.execute('INSERT OR REPLACE INTO ingestions VALUES(?,?,?, ?,NULL,?)',
                           (key, group, 'writing', payload, time.time()))
            # ---- pre-flight ---------------------------------------------------
            # Building the raw episode and parsing the reference time never
            # reaches the backend, so a failure here is provably free of graph
            # side effects. The journal row is rolled back and the caller may
            # retry normally.
            try:
                if hasattr(self.client, 'driver'):
                    blocked = await self.control_blocks(group_id=group)
                    messages = [m for m in messages if m.get('source_message_id') not in blocked]
                    await self.control_record_batch(group_id=group, batch_id=key,
                        source_ids=[m['source_message_id'] for m in messages if m.get('source_message_id')])
                    if not messages:
                        receipt = [{'backend':'graphiti','episode_uuids':[],'withdrawn':True}]
                        with self._db() as db:
                            db.execute("UPDATE ingestions SET state='complete',result=? WHERE id=?", (json.dumps(receipt),key))
                        return receipt
                body = '\n'.join(f"{m['role']}: {m['content']}" for m in messages)
                reference = datetime.fromisoformat(messages[0]['observed_at'])
                episode = RawEpisode(name=key, content=body, source=EpisodeType.message,
                                     source_description='Bumi conversation batch',
                                     reference_time=reference)
            except Exception as exc:
                with self._db() as db:
                    db.execute('DELETE FROM ingestions WHERE id=?', (key,))
                self._audit_replay(key, metadata, 'not_sent',
                                   f'pre-flight failed before any backend call: {type(exc).__name__}: {exc}')
                raise RetryableMemoryError(f'Graphiti write never started: {exc}') from exc
            # ---- the write itself ---------------------------------------------
            # Upstream persists the episode BEFORE it calls the extraction LLM
            # (add_nodes_and_edges_bulk precedes _extract_and_dedupe_nodes_bulk in
            # add_episode_bulk), so from here on a failure can always have left a
            # partial graph write behind, including a 429 or a read timeout. The
            # SDK also owns a 4-attempt retry on the LLM call, so adding another
            # automatic retry here would multiply attempts and duplicate episodes.
            try:
                result = await self.client.add_episode_bulk([episode], group_id=group)
            except asyncio.CancelledError:
                # Cancellation is control flow, not a write outcome. Keep the
                # journal so a blind replay stays blocked, then propagate.
                self._audit_replay(key, metadata, 'cancelled',
                                   'write cancelled; incomplete journal retained')
                raise
            except BaseException as exc:
                self._audit_replay(key, metadata, 'uncertain',
                                   f'{type(exc).__name__}: {exc}')
                raise MemoryWriteUncertainError(
                    'Graphiti write outcome uncertain; episode persistence precedes extraction, '
                    'so a partial write cannot be ruled out. Automatic replay disabled') from exc
            # The graph write has landed. A receipt problem does not undo it.
            try:
                episodes = [e.uuid for e in result.episodes]
                if not episodes:
                    raise RuntimeError('Graphiti returned no episode receipt')
                receipt = [{'backend': 'graphiti', 'episode_uuids': episodes,
                            'node_uuids': [n.uuid for n in result.nodes],
                            'edge_uuids': [e.uuid for e in result.edges]}]
            except Exception as exc:
                self._audit_replay(key, metadata, 'uncertain', f'receipt unusable: {exc}')
                raise MemoryWriteUncertainError(
                    'Graphiti write returned no usable receipt; the graph may already '
                    'contain this batch. Automatic replay disabled') from exc
            if self.config.registered_isolation:
                await self.isolation_index(user_id=user_id,agent_id=agent_id,messages=messages,
                    users=metadata.get('registered_users',[]))
            if hasattr(self.client, 'driver'):
                await self.control_refresh_edges(group_id=group,edge_ids=receipt[0]['edge_uuids'],
                    source_ids=[m['source_message_id'] for m in messages if m.get('source_message_id')])
            with self._db() as db:
                db.execute("UPDATE ingestions SET state='complete',result=? WHERE id=?",
                           (json.dumps(receipt), key))
                db.execute('INSERT OR IGNORE INTO community_jobs(group_id) VALUES(?)', (group,))
            if replay_authorization:
                self._audit_replay(key, metadata, 'succeeded', 'granted replay produced a receipt')
            return receipt

    def _assert_replay_grant(self, grant, key, user_id, agent_id, metadata, messages) -> None:
        """Refuse to re-enter an incomplete write without a valid single-use grant."""
        if not grant:
            raise MemoryWriteUncertainError(
                'Prior Graphiti ingestion is incomplete; inspect the graph and issue an explicit replay')
        if grant.get('chunk_id') != key:
            raise MemoryWriteUncertainError('Replay grant belongs to a different chunk')
        if (grant.get('user_id'), grant.get('agent_id')) != (user_id, agent_id):
            raise MemoryWriteUncertainError('Replay grant belongs to another user/agent scope')
        generation = metadata.get('memory_generation')
        if generation is not None and grant.get('generation') != generation:
            raise MemoryWriteUncertainError('Replay grant belongs to a superseded memory generation')
        expected = canonical_messages_digest(messages)
        if grant.get('digest') != expected:
            raise MemoryWriteUncertainError('Replay grant input does not match the stored chunk')

    def _audit_replay(self, key, metadata, event, detail) -> None:
        with self._db() as db:
            db.execute('INSERT INTO replay_audit(ingestion_id,job_id,event,at,detail) VALUES(?,?,?,?,?)',
                       (key, str(metadata.get('job_id', '')), event, time.time(), detail[:500]))

    def replay_audit(self, ingestion_id: str) -> list[dict]:
        """Read back the journal's own replay trail (grants accepted/refused/outcome)."""
        with self._db() as db:
            rows = db.execute('SELECT event,at,detail FROM replay_audit '
                              'WHERE ingestion_id=? ORDER BY audit_id', (ingestion_id,)).fetchall()
        return [{'event': row[0], 'at': row[1], 'detail': row[2]} for row in rows]

    async def search(self, *, user_id, agent_id, query, **_legacy_options):
        group = graph_group(user_id, agent_id)
        isolated = self.config.registered_isolation or _legacy_options.get('registered_isolation',False)
        users = _legacy_options.get('registered_users',[])
        if isolated and user_id not in {u['user_id'] for u in users if u.get('active',True)}:
            return []
        config = rrf_search_config()
        if isolated:
            config.node_config = config.episode_config = config.community_config = None
        query_vector = await self.client.embedder.create(query) if isolated else None
        result = await self.client.search_(query, group_ids=[group], config=config,
            **({'query_vector':query_vector} if isolated else {}))
        rows = []
        overrides = await self._admin_recall_overrides(group)
        suppressed, has_blocks = (await self.control_suppressed(group_id=group)
                                  if hasattr(self.client, 'driver') else (set(), False))
        fresh_edges = await self.control_fresh_edges(group_id=group) if has_blocks else set()
        blocked_sources = (await self.control_blocks(group_id=group)
                           if has_blocks and any(o.get('control_relearned') for o in overrides) else set())
        changed = {o['id']: o for o in overrides}
        # Derived summaries and original episodes can repeat deleted/edited facts.
        # Until rebuilt against manual revisions, expose facts plus explicit edits only.
        for override in overrides:
            fresh = override.get('control_relearned') and not (set(override.get('control_source_ids') or []) & blocked_sources)
            if not override['deleted'] and (not has_blocks or fresh):
                rows.append(dict(id='edge:'+override['id'],memory='[管理员修订] '+override['content'],
                    user_id=user_id,agent_id=agent_id,fact_status='admin_corrected',
                    metadata={'backend':'graphiti','kind':'edge','admin_version':override['version']}))
        foreign = []
        for kind, values in [('edge', result.edges), ('node', result.nodes),
                             ('episode', result.episodes), ('community', result.communities)]:
            if isolated and kind!='edge':
                continue
            for item in values:
                if item.group_id == group and overrides and (kind != 'edge' or item.uuid in changed):
                    continue
                if item.group_id != group:
                    # Never publish another namespace's content into this user's
                    # context. One bad item must not discard a whole recall, but
                    # a recurring leak must not stay silent either.
                    foreign.append((kind, getattr(item, 'uuid', None), item.group_id))
                    continue
                raw = item.model_dump(mode='json')
                if has_blocks and (kind != 'edge' or (item.uuid not in fresh_edges and (not raw.get('episodes') or set(raw['episodes']) & suppressed))):
                    continue
                text = raw.get('fact') or raw.get('summary') or raw.get('content') or raw.get('name')
                if not text:
                    continue
                times = {k: raw[k] for k in ('valid_at', 'invalid_at', 'expired_at', 'created_at') if raw.get(k)}
                historical = any(datetime.fromisoformat(raw[k].replace('Z', '+00:00')) <= datetime.now(timezone.utc)
                                 for k in ('invalid_at', 'expired_at') if raw.get(k))
                text = f'[{kind}] ' + text
                if times:
                    text += '\n[Graphiti time: ' + json.dumps(times, ensure_ascii=False) + ']'
                rows.append(dict(id=f'{kind}:{item.uuid}', memory=text, user_id=user_id,
                                 agent_id=agent_id, fact_status='historical_version' if historical else 'graphiti_record',
                                 metadata={'backend': 'graphiti', 'kind': kind, **times,
                                           'episode_uuids': raw.get('episodes', [])}))
        if foreign:
            logger.warning('Graphiti returned %d item(s) outside group %s (expected %s): %s',
                           len(foreign), foreign[0][2], group,
                           ', '.join(f'{kind}:{uuid}' for kind, uuid, _ in foreign[:10]))
            self._audit_replay(group, {'job_id': f'search:{agent_id}'}, 'foreign_namespace',
                               f'{len(foreign)} item(s) from foreign group_id={foreign[0][2]}; dropped')
        if isolated:
            rows += await self.isolation_shared_search(user_id=user_id,agent_id=agent_id,query=query,users=users,query_vector=query_vector)
        return rows

    async def build_due_communities(self):
        if self.config.community_every <= 0:
            return
        with self._db() as db:
            groups = db.execute('SELECT group_id,built FROM community_jobs WHERE next_at<=?', (time.time(),)).fetchall()
        for group, built in groups:
            with self._db() as db:
                count = db.execute("""SELECT COUNT(*) FROM (
                    SELECT COALESCE(json_extract(payload,'$.metadata.job_id'),id)
                    FROM ingestions WHERE group_id=? AND state='complete'
                    GROUP BY COALESCE(json_extract(payload,'$.metadata.job_id'),id)
                    HAVING COUNT(*)=MAX(COALESCE(json_extract(payload,'$.metadata.chunk_count'),1))
                )""", (group,)).fetchone()[0]
            if count - built < self.config.community_every:
                continue
            try:
                # Serialize graph mutation; foreground search does not take
                # this lock. The durable watermark coalesces pending rebuilds.
                async with self._write_lock:
                    from bumi_mem.graphiti_community import community_driver
                    scoped_driver = community_driver(self.client.driver,
                        self.config.data_dir / 'community-clustering')
                    await self.client.build_communities(group_ids=[group], driver=scoped_driver)
                with self._db() as db:
                    db.execute('UPDATE community_jobs SET built=?,attempts=0,next_at=0,error=NULL WHERE group_id=?', (count, group))
            except Exception as exc:
                with self._db() as db:
                    db.execute('UPDATE community_jobs SET attempts=attempts+1,next_at=?,error=? WHERE group_id=?',
                               (time.time() + 60, str(exc), group))

    async def debug(self, *, user_id, agent_id):
        group = graph_group(user_id, agent_id)
        with self._db() as db:
            writes = db.execute('SELECT id,state,result FROM ingestions WHERE group_id=? ORDER BY created_at', (group,)).fetchall()
            community = db.execute('SELECT built,attempts,next_at,error FROM community_jobs WHERE group_id=?', (group,)).fetchone()
        return {'group_id': group, 'ingestions': writes, 'community': community}

    async def get_all(self, **kwargs):
        raise NotImplementedError('Use scoped graph debug/search rather than a flat memory listing')

    async def get_memory(self, **kwargs):
        raise NotImplementedError('Flat Mem0 record lookup is unavailable; inspect Graphiti objects')

    async def remember(self, **kwargs):
        raise NotImplementedError('Explicit remember is disabled; use automatic conversation ingestion')

    async def delete_all(self, *, user_id, agent_id):
        group = graph_group(user_id, agent_id)
        async with self._write_lock:
            await self.client.driver.execute_query('MATCH (n {group_id: $group}) DETACH DELETE n', group=group)
            await self._admin_query('MATCH (f:BumiSharedFact {agent_id:$agent,owner_id:$user}) SET f.deleted=true,f.version=f.version+1 RETURN count(f) AS n',agent=agent_id,user=user_id)
            with self._db() as db:
                db.execute('DELETE FROM ingestions WHERE group_id=?', (group,))
                db.execute('DELETE FROM community_jobs WHERE group_id=?', (group,))

    async def close(self):
        try:
            await self.client.close()
        finally:
            if hasattr(self, 'llm_client'):
                await self.llm_client.client.close()
