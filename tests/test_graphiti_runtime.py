"""Graphiti migration contracts without models, network or Neo4j."""
import asyncio
import json
import threading
import types
from datetime import datetime, timezone

import pytest

from bumi_mem.async_memory_access import AsyncMemoryAccess
from bumi_mem.graphiti_memory import GraphitiLongTermMemory, GraphitiConfig, graph_group
from bumi_mem.schemas import MemoryWriteUncertainError
from test_bumi_memory import make_engine, chat_turn


def graph_engine(path, *, rounds=3, l1=20, tokens=8000):
    engine, store, memory, chat, summary = make_engine(path, max_messages=l1)
    engine.config.semantic_state_enabled = False
    engine.consolidator.config.backend = 'graphiti'
    engine.consolidator.config.extraction_turns = rounds
    engine.consolidator.config.extraction_tokens = tokens
    # The effective L1 window is real configuration, not an implicit default.
    engine.consolidator.config.l1_max_messages = l1
    return engine, store, memory, chat, summary


def test_extraction_frequency_does_not_shorten_l1(tmp_path):
    engine, store, memory, _, summary = graph_engine(tmp_path)
    for i in range(2):
        chat_turn(engine, f'我叫小星{i}')
    assert not engine.memory_jobs(user_id='u1', agent_id='a1')
    chat_turn(engine, '第三轮')
    assert len(engine.memory_jobs(user_id='u1', agent_id='a1')) == 1
    assert len(store.list_shared_messages('a1')) == 6
    chat_turn(engine, '第四轮')
    assert len(engine.context_builder.select_l1(store.list_shared_messages('a1'))) == 8
    assert store.semantic_context('u1', 'a1') == []
    assert engine.worker.run_once()
    assert summary.calls == []
    assert memory.add_calls[0]['prompt'] is None


def test_l1_limit_does_not_trigger_graph_extraction(tmp_path):
    engine, store, *_ = graph_engine(tmp_path, rounds=10, l1=2)
    for _ in range(3):
        chat_turn(engine)
    assert not engine.memory_jobs(user_id='u1', agent_id='a1')
    assert len(engine.context_builder.select_l1(store.list_shared_messages('a1'))) == 2


def test_pending_rounds_survive_restart(tmp_path):
    first, *_ = graph_engine(tmp_path, rounds=3)
    chat_turn(first, '第一轮')
    chat_turn(first, '第二轮')
    first.close()
    second, store, *_ = graph_engine(tmp_path, rounds=3)
    assert not second.memory_jobs(user_id='u1', agent_id='a1')
    chat_turn(second, '第三轮')
    assert len(second.memory_jobs(user_id='u1', agent_id='a1')) == 1
    assert len(store.list_shared_messages('a1')) == 6


def test_token_boundary_is_independent_and_no_semantic_promise(tmp_path):
    engine, store, *_ = graph_engine(tmp_path, rounds=10, tokens=5)
    chat_turn(engine, '记住我叫小星')
    jobs = engine.memory_jobs(user_id='u1', agent_id='a1')
    assert jobs and jobs[0].reason == 'extraction_token_limit'
    assert not store.semantic_context('u1', 'a1')


def test_long_graph_input_is_not_truncated(tmp_path):
    from bumi_mem.token_counter import ApproxTokenCounter
    engine, *_ = graph_engine(tmp_path)
    engine.consolidator.memory_token_counter = ApproxTokenCounter()
    session = engine.get_or_create_session(user_id='u1', agent_id='a1')
    source = engine.record_user_message(session.session_id, '恐龙' * 900, user_id='u1', agent_id='a1')
    fragments = engine.consolidator._bounded_graph_messages([source], 500)
    assert ''.join(m.content for m in fragments) == source.content
    assert {m.message_id for m in fragments} == {source.message_id}
    chunks = engine.consolidator._local_chunks(fragments, token_limit=500)
    assert len(chunks) > 1
    assert all(sum(engine.consolidator.memory_token_counter.count_text(m.to_extraction_message()['content']) + 4 for m in c) <= 500 for c in chunks)


def test_read_does_not_wait_behind_write_and_clients_close_on_owner_loop():
    entered, release = threading.Event(), threading.Event()
    class Backend:
        async def add_session(self, **kwargs):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(.01)
            return []
        async def search(self, **kwargs):
            return [threading.get_ident()]
        async def close(self):
            self.closed_on = threading.get_ident()
    backend = Backend()
    access = AsyncMemoryAccess(lambda: backend, read_timeout=1)
    writer = threading.Thread(target=lambda: access.add_session())
    try:
        writer.start()
        assert entered.wait(2)
        owner = access.search()[0]
        assert owner != threading.get_ident()
        release.set()
        writer.join(2)
    finally:
        release.set()
        access.close()
    assert backend.closed_on == owner
    assert not access._thread.is_alive()


def adapter(tmp_path, client):
    """Inject a graph driver, leaving ingestion/journal/search logic real."""
    result = GraphitiLongTermMemory.__new__(GraphitiLongTermMemory)
    result.config = GraphitiConfig('bolt://test', 'neo4j', 'secret', tmp_path, tmp_path, tmp_path)
    result.journal = tmp_path / 'journal.db'
    result.client = client
    result._write_lock = asyncio.Lock()
    async def no_overrides(group):
        return []
    result._admin_recall_overrides = no_overrides
    with result._db() as db:
        db.executescript('''CREATE TABLE ingestions(id TEXT PRIMARY KEY,group_id TEXT,state TEXT,payload TEXT,result TEXT,created_at REAL);
            CREATE TABLE community_jobs(group_id TEXT PRIMARY KEY,built INTEGER DEFAULT 0,attempts INTEGER DEFAULT 0,next_at REAL DEFAULT 0,error TEXT);
            CREATE TABLE replay_audit(audit_id INTEGER PRIMARY KEY AUTOINCREMENT,ingestion_id TEXT NOT NULL,job_id TEXT NOT NULL,
                event TEXT NOT NULL,at REAL NOT NULL,detail TEXT NOT NULL DEFAULT '');''')
    return result


@pytest.fixture
def raw_episode_modules(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'graphiti_core.nodes', types.SimpleNamespace(EpisodeType=types.SimpleNamespace(message='message')))
    monkeypatch.setitem(sys.modules, 'graphiti_core.utils.bulk_utils', types.SimpleNamespace(RawEpisode=lambda **kw: types.SimpleNamespace(**kw)))


def test_upstream_bulk_receipt_recovery_and_uncertain_no_replay(tmp_path, raw_episode_modules):
    class Client:
        calls = 0
        async def add_episode_bulk(self, episodes, **kwargs):
            self.calls += 1
            assert list(kwargs) == ['group_id']  # No custom prompts, types or overrides.
            assert len(episodes) == 1 and episodes[0].content.startswith('user:')
            return types.SimpleNamespace(episodes=[types.SimpleNamespace(uuid='ep')], nodes=[], edges=[])
    async def run():
        client = Client()
        memory = adapter(tmp_path, client)
        args = dict(user_id='u', agent_id='a', messages=[dict(role='user', content='我喜欢恐龙', observed_at='2026-10-15T08:00:00+08:00')], metadata=dict(consolidation_id='batch1', job_id='job1'))
        first = await memory.add_session(**args)
        again = await memory.add_session(**args)
        assert first == again and client.calls == 1
        with memory._db() as db:
            db.execute("UPDATE ingestions SET state='writing'")
        with pytest.raises(MemoryWriteUncertainError):
            await memory.add_session(**args)
        assert client.calls == 1
    asyncio.run(run())


def test_combined_search_keeps_all_kinds_and_temporal_state(tmp_path):
    group = graph_group('u', 'a')
    class Item:
        group_id = group
        def __init__(self, uuid, **values):
            self.uuid, self.values = uuid, values
        def model_dump(self, **kwargs):
            return self.values
    class Client:
        async def search_(self, query, **kwargs):
            assert kwargs['group_ids'] == [group]
            assert kwargs['config'].edge_config.reranker.name == 'rrf'
            return types.SimpleNamespace(edges=[Item('e', fact='以前喜欢红色', invalid_at='2025-01-01T00:00:00Z')],
                nodes=[Item('n', summary='小星')], episodes=[Item('p', content='一段对话')], communities=[Item('c', summary='共同兴趣')])
    async def run():
        memory = adapter(tmp_path, Client())
        rows = await memory.search(user_id='u', agent_id='a', query='颜色', top_k=1, threshold=.99)
        assert len(rows) == 4
        assert rows[0]['fact_status'] == 'historical_version'
        assert {r['metadata']['kind'] for r in rows} == {'edge', 'node', 'episode', 'community'}
    asyncio.run(run())


def test_community_retry_does_not_reingest_and_watermark_survives_restart(tmp_path):
    class Client:
        driver = types.SimpleNamespace(graph_operations_interface=None)
        calls = 0
        fail = True
        async def build_communities(self, **kwargs):
            assert kwargs['driver'] is not self.driver
            assert self.driver.graph_operations_interface is None
            self.calls += 1
            if self.fail:
                raise RuntimeError('offline')
    async def run():
        client = Client()
        memory = adapter(tmp_path, client)
        with memory._db() as db:
            for i in range(5):
                db.execute('INSERT INTO ingestions VALUES(?,?,?,?,?,?)', (str(i),'g','complete',json.dumps({'metadata': {'job_id': str(i)}}),'[]',i))
            db.execute("INSERT INTO community_jobs(group_id) VALUES('g')")
        await memory.build_due_communities()
        with memory._db() as db:
            assert db.execute('SELECT built,attempts FROM community_jobs').fetchone() == (0,1)
            db.execute('UPDATE community_jobs SET next_at=0')
        client.fail = False
        await memory.build_due_communities()
        await memory.build_due_communities()
        assert client.calls == 2
        with memory._db() as db:
            assert db.execute('SELECT built FROM community_jobs').fetchone()[0] == 5
    asyncio.run(run())

def test_community_waits_for_all_chunks_of_a_batch(tmp_path):
    class Client:
        driver = types.SimpleNamespace(graph_operations_interface=None)
        calls = 0
        async def build_communities(self, **kwargs):
            self.calls += 1
    async def run():
        client = Client()
        memory = adapter(tmp_path, client)
        with memory._db() as db:
            for i in range(4):
                db.execute('INSERT INTO ingestions VALUES(?,?,?,?,?,?)',
                    (str(i),'g','complete',json.dumps({'metadata': {'job_id': str(i)}}),'[]',i))
            db.execute('INSERT INTO ingestions VALUES(?,?,?,?,?,?)',
                ('part1','g','complete',json.dumps({'metadata': {'job_id': 'split', 'chunk_count': 2}}),'[]',5))
            db.execute("INSERT INTO community_jobs(group_id) VALUES('g')")
        await memory.build_due_communities()
        assert client.calls == 0
        with memory._db() as db:
            db.execute('INSERT INTO ingestions VALUES(?,?,?,?,?,?)',
                ('part2','g','complete',json.dumps({'metadata': {'job_id': 'split', 'chunk_count': 2}}),'[]',6))
        await memory.build_due_communities()
        assert client.calls == 1
    asyncio.run(run())


def test_deepseek_transport_options_preserve_async_call():
    from bumi_mem.providers import ProviderConfig, bind_provider_request_options
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        return {'ok': True}
    config = ProviderConfig('deepseek', 'deepseek-flash', 'https://example.test', 'test', thinking_mode='disabled')
    bound = bind_provider_request_options(create, config)
    assert asyncio.run(bound(messages=[], response_format={'type': 'json_object'})) == {'ok': True}
    assert calls[0]['extra_body'] == {'thinking': {'type': 'disabled'}}
    assert calls[0]['response_format'] == {'type': 'json_object'}


# ---------------------------------------------------------------------------
# Explicit replay authorization: per-chunk, scope-bound, single-use.
# ---------------------------------------------------------------------------

from bumi_mem.long_term import FakeLongTermMemory
from bumi_mem.schemas import canonical_messages_digest


class WriteThenFail(FakeLongTermMemory):
    """Mirrors a lost response: the write may already have landed.

    Fails only the FIRST attempt. That is what makes a granted replay able to
    succeed later, and lets the tests distinguish "retry refused" from "retry
    ran and failed again". The counter is per-instance so tests stay isolated.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def add_session(self, **kwargs):
        self.attempts += 1
        if self.attempts > 1:
            return super().add_session(**kwargs)
        super().add_session(**kwargs)
        raise OSError('response lost after the write may have landed')


def uncertain_engine(tmp_path, long_term=None):
    """One completed turn is enough to queue a batch whose write then fails."""
    engine, store, memory, _, _ = make_engine(
        tmp_path, long_term=long_term if long_term is not None else WriteThenFail())
    engine.config.semantic_state_enabled = False
    engine.consolidator.config.backend = 'graphiti'
    engine.consolidator.config.extraction_turns = 1
    engine.consolidator.config.extraction_tokens = 8000
    engine.consolidator.config.l1_max_messages = engine.config.max_messages
    return engine, store, memory


def make_uncertain(tmp_path):
    """Drive one real batch until its write outcome is unknown."""
    engine, store, memory = uncertain_engine(tmp_path)
    chat_turn(engine, '我叫小星，我最喜欢三角龙。')
    job = engine.memory_jobs(user_id='u1', agent_id='a1')[0]
    assert engine.worker.run_once()
    assert store.get_job(job.job_id).status == 'failed'
    chunks = store.chunks(job.job_id)
    assert chunks and all(c['status'] == 'uncertain' for c in chunks)
    return engine, store, memory, job, chunks


def test_failed_write_plain_retry_refused_then_granted_replay_succeeds(tmp_path):
    engine, store, memory, job, chunks = make_uncertain(tmp_path)
    writes_before = len(memory.add_calls)
    # (a) first failure left the outcome unknown
    assert store.get_job(job.job_id).status == 'failed'
    assert store.replay_authorization(chunks[0]['chunk_id']) is None
    # (b) an ordinary retry cannot silently re-run an uncertain write
    with pytest.raises(ValueError, match='unknown'):
        engine.retry_memory_job(job.job_id, user_id='u1', agent_id='a1')
    assert len(memory.add_calls) == writes_before
    # (c) explicit authorization is issued and persisted with the reset
    engine.retry_memory_job(job.job_id, user_id='u1', agent_id='a1', allow_replay=True)
    for chunk in chunks:
        grant = store.replay_authorization(chunk['chunk_id'])
        assert grant['status'] == 'issued'
        assert (grant['user_id'], grant['agent_id']) == ('u1', 'a1')
        assert grant['generation'] == store.get_job(job.job_id).generation
        # Granting authorizes the replay; it does not itself rewrite the chunk.
        assert store.chunks(job.job_id)[0]['status'] == 'pending'
    # (d) the granted replay executes for real and succeeds
    assert engine.worker.run_once()
    assert store.get_job(job.job_id).status == 'succeeded'
    assert len(memory.add_calls) == writes_before + 1
    assert memory.add_calls[-1]['replay_authorization'] is not None
    for chunk in chunks:
        assert store.chunks(job.job_id)[0]['status'] == 'succeeded'
        assert store.replay_authorization(chunk['chunk_id'])['status'] == 'consumed'
    events = [row['event'] for row in store.replay_audit(job_id=job.job_id)]
    assert 'granted' in events and 'claimed' in events and 'consumed' in events


def test_replay_grant_is_single_use(tmp_path):
    engine, store, memory, job, chunks = make_uncertain(tmp_path)
    engine.retry_memory_job(job.job_id, user_id='u1', agent_id='a1', allow_replay=True)
    chunk_id = chunks[0]['chunk_id']
    digest = store.chunk_input_digest(json.loads(chunks[0]['messages_json']))
    grant = store.take_replay_authorization(chunk_id=chunk_id, user_id='u1', agent_id='a1',
                                            generation=store.get_job(job.job_id).generation,
                                            input_digest=digest)
    assert grant['chunk_id'] == chunk_id
    # The same authorization cannot be claimed a second time.
    with pytest.raises(MemoryWriteUncertainError, match='already executing'):
        store.take_replay_authorization(chunk_id=chunk_id, user_id='u1', agent_id='a1',
                                        generation=store.get_job(job.job_id).generation,
                                        input_digest=digest)
    store.record_replay_outcome(chunk_id=chunk_id, success=True)
    with pytest.raises(MemoryWriteUncertainError):
        store.take_replay_authorization(chunk_id=chunk_id, user_id='u1', agent_id='a1',
                                        generation=store.get_job(job.job_id).generation,
                                        input_digest=digest)


def test_replay_failure_burns_grant_and_needs_a_new_one(tmp_path):
    class FailAgain(WriteThenFail):
        """Fails the initial write AND the first granted replay."""

        def __init__(self) -> None:
            super().__init__()
            self.replay_attempts = 0

        def add_session(self, **kwargs):
            if kwargs.get('replay_authorization') is not None:
                self.replay_attempts += 1
                if self.replay_attempts == 1:
                    raise OSError('replay failed again')
            return super().add_session(**kwargs)

    engine, store, memory = uncertain_engine(tmp_path, long_term=FailAgain())
    chat_turn(engine, '我叫小星')
    job = engine.memory_jobs(user_id='u1', agent_id='a1')[0]
    assert engine.worker.run_once()
    chunk_id = store.chunks(job.job_id)[0]['chunk_id']
    engine.retry_memory_job(job.job_id, user_id='u1', agent_id='a1', allow_replay=True)
    assert engine.worker.run_once()
    # The replay failed: still unknown, and the grant is burnt rather than reusable.
    assert store.get_job(job.job_id).status == 'failed'
    grant = store.replay_authorization(chunk_id)
    assert grant['status'] == 'executing'
    assert grant['attempt'] == 1 and grant['last_error']
    with pytest.raises(MemoryWriteUncertainError):
        store.take_replay_authorization(chunk_id=chunk_id, user_id='u1', agent_id='a1',
                                        generation=store.get_job(job.job_id).generation,
                                        input_digest=store.chunk_input_digest(
                                            json.loads(store.chunks(job.job_id)[0]['messages_json'])))
    events = [row['event'] for row in store.replay_audit(job_id=job.job_id)]
    assert events.count('granted') == 1 and 'failed' in events
    # A fresh authorization is required before another attempt.
    FailAgain.calls = 0
    engine.retry_memory_job(job.job_id, user_id='u1', agent_id='a1', allow_replay=True)
    assert store.replay_authorization(chunk_id)['status'] == 'issued'
    assert engine.worker.run_once()
    assert store.get_job(job.job_id).status == 'succeeded'


def test_replay_grant_cannot_cross_scope_or_generation(tmp_path):
    engine, store, memory, job, chunks = make_uncertain(tmp_path)
    chunk_id = chunks[0]['chunk_id']
    digest = store.chunk_input_digest(json.loads(chunks[0]['messages_json']))
    generation = store.get_job(job.job_id).generation
    engine.retry_memory_job(job.job_id, user_id='u1', agent_id='a1', allow_replay=True)
    with pytest.raises(MemoryWriteUncertainError, match='another user/agent scope'):
        store.take_replay_authorization(chunk_id=chunk_id, user_id='u2', agent_id='a1',
                                        generation=generation, input_digest=digest)
    with pytest.raises(MemoryWriteUncertainError, match='superseded memory generation'):
        store.take_replay_authorization(chunk_id=chunk_id, user_id='u1', agent_id='a1',
                                        generation=generation + 1, input_digest=digest)
    with pytest.raises(MemoryWriteUncertainError, match='input changed'):
        store.take_replay_authorization(chunk_id=chunk_id, user_id='u1', agent_id='a1',
                                        generation=generation, input_digest='deadbeef')
    # The refusals are recorded, and the grant is still unused and claimable.
    refused = [r for r in store.replay_audit(job_id=job.job_id) if r['event'] == 'refused']
    assert len(refused) == 3
    assert store.replay_authorization(chunk_id)['status'] == 'issued'
    assert store.take_replay_authorization(chunk_id=chunk_id, user_id='u1', agent_id='a1',
                                          generation=generation, input_digest=digest)


def test_replay_authorization_survives_restart(tmp_path):
    from bumi_mem.session_store import SQLiteSessionStore
    engine, store, memory, job, chunks = make_uncertain(tmp_path)
    chunk_id = chunks[0]['chunk_id']
    engine.retry_memory_job(job.job_id, user_id='u1', agent_id='a1', allow_replay=True)
    digest = store.chunk_input_digest(json.loads(chunks[0]['messages_json']))
    engine.close()
    # A brand-new store over the same file still enforces single use.
    reopened = SQLiteSessionStore(tmp_path / 'bumi.sqlite3')
    assert reopened.replay_authorization(chunk_id)['status'] == 'issued'
    grant = reopened.take_replay_authorization(chunk_id=chunk_id, user_id='u1', agent_id='a1',
                                              generation=reopened.get_job(job.job_id).generation,
                                              input_digest=digest)
    assert grant['attempt'] == 1
    reopened.record_replay_outcome(chunk_id=chunk_id, success=False, detail='process died')
    assert reopened.replay_authorization(chunk_id)['status'] == 'executing'
    with pytest.raises(MemoryWriteUncertainError):
        reopened.take_replay_authorization(chunk_id=chunk_id, user_id='u1', agent_id='a1',
                                          generation=reopened.get_job(job.job_id).generation,
                                          input_digest=digest)


def test_journal_refuses_incomplete_write_without_grant(tmp_path, raw_episode_modules):
    class Client:
        calls = 0

        async def add_episode_bulk(self, episodes, **kwargs):
            self.calls += 1
            return types.SimpleNamespace(episodes=[types.SimpleNamespace(uuid='ep')], nodes=[], edges=[])

    async def run():
        client = Client()
        memory = adapter(tmp_path, client)
        messages = [dict(role='user', content='我喜欢恐龙', observed_at='2026-10-15T08:00:00+08:00')]
        meta = dict(consolidation_id='batch1', job_id='job1', memory_generation=3)
        await memory.add_session(user_id='u', agent_id='a', messages=messages, metadata=meta)
        assert client.calls == 1
        with memory._db() as db:                     # simulate an interrupted write
            db.execute("UPDATE ingestions SET state='writing'")
        with pytest.raises(MemoryWriteUncertainError, match='incomplete'):
            await memory.add_session(user_id='u', agent_id='a', messages=messages, metadata=meta)
        assert client.calls == 1                     # no blind re-entry
    asyncio.run(run())


def test_journal_accepts_only_matching_grant_and_audits(tmp_path, raw_episode_modules):
    class Client:
        calls = 0

        async def add_episode_bulk(self, episodes, **kwargs):
            self.calls += 1
            return types.SimpleNamespace(episodes=[types.SimpleNamespace(uuid='ep')], nodes=[], edges=[])

    async def run():
        client = Client()
        memory = adapter(tmp_path, client)
        messages = [dict(role='user', content='我喜欢恐龙', observed_at='2026-10-15T08:00:00+08:00')]
        meta = dict(consolidation_id='batch1', job_id='job1', memory_generation=3)
        digest = canonical_messages_digest(messages)
        good = dict(chunk_id='batch1', user_id='u', agent_id='a', generation=3, digest=digest)
        await memory.add_session(user_id='u', agent_id='a', messages=messages, metadata=meta)
        with memory._db() as db:
            db.execute("UPDATE ingestions SET state='writing'")
        # Wrong chunk, wrong scope, wrong generation and wrong input are all refused.
        for bad in ({'digest': digest, 'user_id': 'u', 'agent_id': 'a', 'generation': 3},
                    {**good, 'chunk_id': 'other'},
                    {**good, 'user_id': 'someone_else'},
                    {**good, 'generation': 4},
                    {**good, 'digest': 'deadbeef'}):
            with pytest.raises(MemoryWriteUncertainError):
                await memory.add_session(user_id='u', agent_id='a', messages=messages,
                                         metadata=meta, replay_authorization=bad)
        assert client.calls == 1
        # A matching grant reopens the write exactly once.
        receipt = await memory.add_session(user_id='u', agent_id='a', messages=messages,
                                           metadata=meta, replay_authorization=good)
        assert client.calls == 2 and receipt[0]['episode_uuids'] == ['ep']
        events = [row['event'] for row in memory.replay_audit('batch1')]
        assert 'accepted' in events and 'succeeded' in events
        # Once complete with identical input, the original receipt is returned.
        again = await memory.add_session(user_id='u', agent_id='a', messages=messages,
                                         metadata=meta, replay_authorization=good)
        assert client.calls == 2 and again == receipt
    asyncio.run(run())


# ---------------------------------------------------------------------------
# Failure classification: "provably no write yet" vs "may have written".
# Upstream add_episode_bulk persists the episode BEFORE calling the extraction
# LLM, so a 429/timeout during extraction can never be assumed side-effect free.
# ---------------------------------------------------------------------------

from bumi_mem.schemas import RetryableMemoryError


def test_preflight_failure_is_retryable_and_rolls_back_the_journal(tmp_path, raw_episode_modules):
    class Client:
        calls = 0

        async def add_episode_bulk(self, episodes, **kwargs):
            self.calls += 1
            return types.SimpleNamespace(episodes=[types.SimpleNamespace(uuid='ep')], nodes=[], edges=[])

    async def run():
        client = Client()
        memory = adapter(tmp_path, client)
        # No observed_at: the reference time cannot be parsed, so nothing was sent.
        with pytest.raises(RetryableMemoryError):
            await memory.add_session(user_id='u', agent_id='a',
                                     messages=[dict(role='user', content='x')],
                                     metadata=dict(consolidation_id='k', job_id='j'))
        assert client.calls == 0
        with memory._db() as db:
            rows = db.execute('SELECT COUNT(*) FROM ingestions').fetchone()[0]
        assert rows == 0, 'a write that never started must not be left as an incomplete journal row'
        assert 'not_sent' in [r['event'] for r in memory.replay_audit('k')]
        # Because the journal was rolled back, a normal retry is allowed.
        good = [dict(role='user', content='x', observed_at='2026-10-15T08:00:00+08:00')]
        receipt = await memory.add_session(user_id='u', agent_id='a', messages=good,
                                           metadata=dict(consolidation_id='k', job_id='j'))
        assert client.calls == 1 and receipt[0]['episode_uuids'] == ['ep']
    asyncio.run(run())


@pytest.mark.parametrize('failure', [
    RuntimeError('429 Too Many Requests'),
    TimeoutError('read timeout'),
    ConnectionError('connection reset'),
])
def test_partial_write_failure_is_never_auto_replayed(tmp_path, raw_episode_modules, failure):
    class Client:
        calls = 0

        async def add_episode_bulk(self, episodes, **kwargs):
            self.calls += 1
            # The episode was already persisted upstream when this fires.
            raise failure

    async def run():
        client = Client()
        memory = adapter(tmp_path, client)
        messages = [dict(role='user', content='我喜欢恐龙', observed_at='2026-10-15T08:00:00+08:00')]
        meta = dict(consolidation_id='k', job_id='j', memory_generation=1)
        with pytest.raises(MemoryWriteUncertainError):
            await memory.add_session(user_id='u', agent_id='a', messages=messages, metadata=meta)
        assert client.calls == 1
        # Journal keeps the incomplete row, so a blind retry is refused.
        with memory._db() as db:
            state = db.execute('SELECT state FROM ingestions WHERE id=?', ('k',)).fetchone()[0]
        assert state == 'writing'
        with pytest.raises(MemoryWriteUncertainError, match='incomplete'):
            await memory.add_session(user_id='u', agent_id='a', messages=messages, metadata=meta)
        assert client.calls == 1
        assert 'uncertain' in [r['event'] for r in memory.replay_audit('k')]
    asyncio.run(run())


def test_cancellation_keeps_incomplete_journal_and_does_not_become_uncertain(tmp_path, raw_episode_modules):
    started = threading.Event()

    class Client:
        async def add_episode_bulk(self, episodes, **kwargs):
            started.set()
            raise asyncio.CancelledError()

    async def run():
        memory = adapter(tmp_path, Client())
        messages = [dict(role='user', content='x', observed_at='2026-10-15T08:00:00+08:00')]
        meta = dict(consolidation_id='k', job_id='j', memory_generation=1)
        # Cancellation is control flow: it must propagate as CancelledError,
        # never be swallowed and relabelled as an uncertain write.
        with pytest.raises(asyncio.CancelledError):
            await memory.add_session(user_id='u', agent_id='a', messages=messages, metadata=meta)
        with memory._db() as db:
            state = db.execute('SELECT state FROM ingestions WHERE id=?', ('k',)).fetchone()[0]
        assert state == 'writing', 'an interrupted write must stay blocked from blind replay'
        assert 'cancelled' in [r['event'] for r in memory.replay_audit('k')]
        # Still refused without an explicit grant.
        with pytest.raises(MemoryWriteUncertainError):
            await memory.add_session(user_id='u', agent_id='a', messages=messages, metadata=meta)
    asyncio.run(run())


def test_no_extra_retry_layer_around_the_sdk(tmp_path, raw_episode_modules):
    """The SDK already retries the LLM call 4x; a wrapper must not multiply that."""
    class Client:
        calls = 0

        async def add_episode_bulk(self, episodes, **kwargs):
            self.calls += 1
            raise RuntimeError('429')

    async def run():
        client = Client()
        memory = adapter(tmp_path, client)
        messages = [dict(role='user', content='x', observed_at='2026-10-15T08:00:00+08:00')]
        meta = dict(consolidation_id='k', job_id='j', memory_generation=1)
        with pytest.raises(MemoryWriteUncertainError):
            await memory.add_session(user_id='u', agent_id='a', messages=messages, metadata=meta)
        assert client.calls == 1, 'one caller attempt must map to exactly one SDK attempt'
    asyncio.run(run())


def test_retryable_preflight_error_returns_chunk_to_pending(tmp_path):
    """End to end: a pre-write failure lets the consolidator retry normally."""
    class NoTimestamp(FakeLongTermMemory):
        def add_session(self, **kwargs):
            raise RetryableMemoryError('graph backend not initialised')

    engine, store, memory = uncertain_engine(tmp_path, long_term=NoTimestamp())
    chat_turn(engine, '我叫小星')
    job = engine.memory_jobs(user_id='u1', agent_id='a1')[0]
    assert engine.worker.run_once()
    # Retryable: the chunk goes back to pending and the job stays retryable.
    assert store.chunks(job.job_id)[0]['status'] == 'pending'
    assert store.get_job(job.job_id).last_error
    assert not store.replay_authorization(store.chunks(job.job_id)[0]['chunk_id'])


# ---------------------------------------------------------------------------
# Phase 4 review items.
# ---------------------------------------------------------------------------

def test_per_message_time_and_source_reach_raw_episode_content(tmp_path, raw_episode_modules, caplog):
    """Capture what is actually handed to RawEpisode and check per-message provenance."""
    captured = {}

    class Client:
        async def add_episode_bulk(self, episodes, **kwargs):
            captured['episodes'] = episodes
            return types.SimpleNamespace(episodes=[types.SimpleNamespace(uuid='ep')], nodes=[], edges=[])

    from bumi_mem.schemas import MessageRecord
    records = [
        MessageRecord(message_id='m1', session_id='s', sequence_no=1, role='user',
                      content='我叫小星', token_count=1, created_at='2026-10-15T10:00:00+08:00',
                      user_id='u', agent_id='a'),
        MessageRecord(message_id='m2', session_id='s', sequence_no=2, role='assistant',
                      content='你好小星', token_count=1, created_at='2026-10-15T10:00:05+08:00',
                      user_id='u', agent_id='a'),
        MessageRecord(message_id='m3', session_id='s', sequence_no=3, role='user',
                      content='我喜欢三角龙', token_count=1, created_at='2026-10-15T10:01:00+08:00',
                      user_id='u', agent_id='a'),
    ]
    messages = [r.to_extraction_message() for r in records]

    async def run():
        memory = adapter(tmp_path, Client())
        await memory.add_session(user_id='u', agent_id='a', messages=messages,
                                 metadata=dict(consolidation_id='k', job_id='j'))
        content = captured['episodes'][0].content
        # Every message's own timestamp and source id must be present, so a
        # multi-turn batch is not flattened onto a single reference time.
        for record in records:
            assert record.created_at in content, f'missing time for {record.message_id}'
            assert record.message_id in content, f'missing source id for {record.message_id}'
        # Ordering is preserved end to end.
        assert content.index('m1') < content.index('m2') < content.index('m3')
        # Sanity: this is NOT merely "role: content".
        assert content != '\n'.join(f"{m['role']}: {m['content'].split(chr(10),1)[-1]}" for m in messages)
    asyncio.run(run())


def test_l1_max_messages_is_real_configuration_not_a_hidden_default(tmp_path):
    """A larger L1 window must take effect, and its absence must not default silently."""
    engine, store, *_ = graph_engine(tmp_path, rounds=10, l1=30)
    assert engine.consolidator.config.l1_max_messages == 30
    # Submit explicitly: the snapshot carries the configured window.
    receipt = engine.consolidator.submit(
        engine.get_or_create_session(user_id='u1', agent_id='a1').session_id, reason='test')
    job = store.get_job(receipt.job_id)
    assert json.loads(job.payload_json)['settings']['l1_max_messages'] == 30
    # A Graphiti submission with no configured window is refused, not guessed.
    engine.consolidator.config.l1_max_messages = None
    engine.consolidator.config.carryover_messages = 3
    with pytest.raises(ValueError, match='l1_max_messages'):
        engine.consolidator.submit(
            engine.get_or_create_session(user_id='u2', agent_id='a1').session_id, reason='test')


def test_smaller_carryover_does_not_shrink_the_l1_window(tmp_path):
    """carryover below the window is raised to the window; the window is the floor."""
    engine, store, *_ = graph_engine(tmp_path, rounds=2, l1=30)
    engine.consolidator.config.carryover_messages = 4
    for _ in range(6):
        chat_turn(engine)
    receipt = engine.memory_jobs(user_id='u1', agent_id='a1')[0]
    new_sid = store.get_job(receipt.job_id).new_session_id
    assert len(store.list_shared_messages('a1')) >= 4
    assert store.stats(new_sid).context_only_messages >= 4


def test_foreign_group_results_are_dropped_and_warned_not_fatal(tmp_path, caplog):
    group = graph_group('u', 'a')

    class Item:
        def __init__(self, uuid, gid, **values):
            self.uuid, self.group_id, self.values = uuid, gid, values

        def model_dump(self, **kwargs):
            return self.values

    class Client:
        async def search_(self, query, **kwargs):
            return types.SimpleNamespace(
                edges=[Item('e1', group, fact='我喜歡三角龍'), Item('leak', 'someone_else_group', fact='别人的秘密')],
                nodes=[Item('n1', group, summary='小星')],
                episodes=[Item('p1', 'another_group', content='别人的对话')],
                communities=[Item('c1', group, summary='共同兴趣')])

    async def run():
        memory = adapter(tmp_path, Client())
        import logging
        with caplog.at_level(logging.WARNING):
            rows = await memory.search(user_id='u', agent_id='a', query='恐龙')
        # Legitimate results survive; nothing from a foreign namespace is published.
        assert {r['id'] for r in rows} == {'edge:e1', 'node:n1', 'community:c1'}
        assert all('别人的' not in r['memory'] for r in rows)
        assert any('outside group' in r.message for r in caplog.records), 'a leak must not be silent'
        # And it is recorded for later inspection.
        assert 'foreign_namespace' in [e['event'] for e in memory.replay_audit(group)]
    asyncio.run(run())


def test_rrf_keeps_routes_and_limits_without_mutating_upstream():
    from bumi_mem.graphiti_memory import rrf_search_config
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER as original
    config = rrf_search_config()
    assert config.limit == original.limit == 10
    for field in ('edge_config', 'node_config', 'episode_config', 'community_config'):
        assert getattr(config, field).search_methods == getattr(original, field).search_methods
        assert getattr(config, field).reranker.name == 'rrf'
        assert getattr(original, field).reranker.name == 'cross_encoder'



def test_recall_refs_are_replaced_in_one_write_transaction(tmp_path):
    """Graphiti mode refreshes this user's recall refs without a second write pass."""
    engine, store, memory, _, _ = make_engine(tmp_path)
    engine.config.semantic_state_enabled = False
    engine.consolidator.config.backend = 'graphiti'
    engine.consolidator.config.extraction_turns = 99
    engine.consolidator.config.l1_max_messages = engine.config.max_messages
    memory.search_results = [{'id': 'dino', 'memory': '小星最喜欢三角龙'}]
    result = engine.process_context(user_id='u1', agent_id='a1', query='我喜欢什么动物',
                                      previous_ai_reply=None)
    assert result['recall_status'] == 'ok'
    selected = [m.message_id for m in store.list_shared_messages('a1')]
    refs = store.shared_memory_refs('a1', selected_message_ids=selected)
    assert [r['id'] for r in refs] == ['dino']  # refs expose the payload row plus ref_id
    # A later turn that recalls nothing must retire the stale reference rather
    # than leaving retired memory visible as fresh context.
    memory.search_results = []
    engine.process_context(user_id='u1', agent_id='a1', query='别的',
                           previous_ai_reply='我喜欢三角龙')
    assert store.shared_memory_refs('a1', selected_message_ids=[m.message_id for m in store.list_shared_messages('a1')]) == []
    # Deletion protection is unchanged: a pending forget-all suppresses refs.
    store.request_forget(user_id='u1', agent_id='a1', max_attempts=1)
    with store._connect(write=True) as conn:
        conn.execute('UPDATE memory_scopes SET delete_pending=1 WHERE user_id=? AND agent_id=?', ('u1', 'a1'))
    memory.search_results = [{'id': 'dino', 'memory': '小星最喜欢三角龙'}]
    engine.process_context(user_id='u1', agent_id='a1', query='还记得三角龙吗',
                           previous_ai_reply='三角龙')
    assert store.shared_memory_refs('a1', selected_message_ids=[m.message_id for m in store.list_shared_messages('a1')]) == []


def test_graphiti_availability_recovers_without_consuming_write_jobs():
    from bumi_mem.schemas import RetryableMemoryError
    class Backend:
        fail = True
        probes = 0
        async def warmup(self, **kwargs):
            self.probes += 1
            if self.fail:
                raise RetryableMemoryError('database unavailable')
        async def close(self):
            pass
    backend = Backend()
    access = AsyncMemoryAccess(lambda: backend, degrade_cooloff_seconds=60)
    try:
        assert access.status == 'unchecked'
        assert access.ensure_available() is False
        assert access.degraded and backend.probes == 1
        assert access.ensure_available() is False and backend.probes == 1
        backend.fail = False
        access.degrade_cooloff_seconds = 0
        assert access.ensure_available() is True
        assert access.status == 'ready' and backend.probes == 2
    finally:
        access.close()


def test_rrf_warmup_has_no_model_reranker(tmp_path):
    calls = []
    class Client:
        async def build_indices_and_constraints(self):
            calls.append('indices')
        embedder = types.SimpleNamespace()
    async def embed(query):
        calls.append('embedding')
    client = Client()
    client.embedder.create = embed
    memory = adapter(tmp_path, client)
    asyncio.run(memory.warmup(query='warmup'))
    assert calls == ['indices', 'embedding']
