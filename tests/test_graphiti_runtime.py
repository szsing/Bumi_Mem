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
    with result._db() as db:
        db.executescript('''CREATE TABLE ingestions(id TEXT PRIMARY KEY,group_id TEXT,state TEXT,payload TEXT,result TEXT,created_at REAL);
            CREATE TABLE community_jobs(group_id TEXT PRIMARY KEY,built INTEGER DEFAULT 0,attempts INTEGER DEFAULT 0,next_at REAL DEFAULT 0,error TEXT);''')
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
            assert kwargs == {'group_ids': [group]}
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
        calls = 0
        fail = True
        async def build_communities(self, **kwargs):
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
