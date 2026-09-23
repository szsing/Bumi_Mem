from __future__ import annotations

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from bumi_mem.api import create_app
from bumi_mem.engine import BumiMemoryConfig
from bumi_mem.long_term import FakeLongTermMemory
from bumi_mem.schemas import MemoryReadUnavailable
from bumi_mem.service_settings import ServiceSettings, build_service_engine
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.turn_store import TurnConflict
from test_bumi_memory import make_engine


KEY = 'test-service-token-not-a-real-key-12345678'
AUTH = {'Authorization': 'Bearer ' + KEY}


@pytest.fixture
def harness(tmp_path):
    config = ServiceSettings(data_dir=tmp_path, embed_model_path=tmp_path/'not-downloaded',
                             api_key=KEY, memory_api_key='test-memory-only', agent_id='bot')
    engine, store, memory, chat, summary = make_engine(tmp_path)
    engine.chat_provider = None  # no answering model, even in tests
    # Deterministic worker: start warmup normally; manually drain jobs when desired.
    engine.worker.start = lambda: None
    config = replace(config, data_dir=tmp_path)
    app = create_app(config, engine_factory=lambda _: engine)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, engine, store, memory, chat, summary, app, config



def test_restart_replays_saved_prompt_and_accepts_original_reply(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    first = engine.prepare_prompt(user_id='A',agent_id='bot',query='original',idempotency_key='r1')
    engine.close()
    restarted, _, _, _, _ = make_engine(tmp_path,long_term=memory)
    assert restarted.prepare_prompt(user_id='A',agent_id='bot',query='original',idempotency_key='r1') == first
    assert len(memory.search_calls) == 1
    result = restarted.record_reply(turn_id=first['turn_id'],agent_id='bot',reply='original answer')
    assert result['reply_saved']
    restarted.close()
    again, _, _, _, _ = make_engine(tmp_path)
    assert again.record_reply(turn_id=first['turn_id'],agent_id='bot',reply='original answer') == result
    assert len(again.store.list_shared_messages('bot')) == 2

def test_preparing_turn_recovers_after_crash_between_reserve_and_prompt(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    row = store.reserve_turn(user_id='A',agent_id='bot',query='q',token_count=5,idempotency_key='r',
                             ttl_seconds=300,settings=engine.consolidator._settings(),max_messages=20,max_l1_tokens=8000)
    engine.close()
    new, _, _, _, _ = make_engine(tmp_path)
    result = new.prepare_prompt(user_id='A',agent_id='bot',query='q',idempotency_key='r')
    assert result['turn_id'] == row['turn_id']
    assert len(store.list_shared_messages('bot')) == 1

def test_pending_turn_blocks_manual_rotation_and_wrong_agent_finish(tmp_path):
    engine, _, _, _, _ = make_engine(tmp_path)
    turn = engine.prepare_prompt(user_id='A',agent_id='bot',query='q')
    for call in (lambda:engine.end_session(user_id='B',agent_id='bot'),
                 lambda:engine.discard_session(user_id='B',agent_id='bot')):
        with pytest.raises(TurnConflict,match='agent_busy'):
            call()
    from bumi_mem.turn_store import TurnNotFound
    with pytest.raises(TurnNotFound):
        engine.record_reply(turn_id=turn['turn_id'],agent_id='other',reply='bad')

def test_v3_database_upgrade_preserves_shared_window(tmp_path):
    store = SQLiteSessionStore(tmp_path/'legacy.sqlite3')
    session = store.create_session('A','bot')
    store.append_message(session.session_id,role='user',content='keep',token_count=5)
    with sqlite3.connect(store.path) as c:
        c.execute('DROP TABLE interaction_turns')
        c.execute('PRAGMA user_version=3')
    upgraded = SQLiteSessionStore(store.path)
    assert upgraded.list_shared_messages('bot') == store.list_shared_messages('bot')
    with sqlite3.connect(store.path) as c:
        assert c.execute('PRAGMA user_version').fetchone()[0] == 8
        assert c.execute('SELECT COUNT(*) FROM interaction_turns').fetchone()[0] == 0

def test_config_validation_and_missing_local_model_do_not_download(tmp_path):
    config = ServiceSettings(data_dir=tmp_path,embed_model_path=tmp_path/'missing',
                             api_key=KEY,memory_api_key='memory-secret')
    assert 'memory-secret' not in repr(config) and KEY not in repr(config)
    with pytest.raises(ValueError,match='existing local model'):
        build_service_engine(config)
    for changes in ({'api_key':'short'},{'read_timeout':0},{'embed_device':'bad'},
                    {'memory_api_key':''},{'read_timeout':float('nan')}):
        with pytest.raises(ValueError):
            replace(config,**changes)

def test_startup_warmup_failure_never_serves_ready(tmp_path):
    config = ServiceSettings(data_dir=tmp_path,embed_model_path=tmp_path/'missing',api_key=KEY,memory_api_key='fake')
    engine, _, _, _, _ = make_engine(tmp_path)
    def fail():
        raise RuntimeError('warmup failed')
    engine.start = fail
    app = create_app(config,engine_factory=lambda _:engine)
    with pytest.raises(RuntimeError,match='warmup failed'):
        with TestClient(app):
            pass
    assert not app.state.ready and engine._closed

def test_second_http_process_cannot_open_same_data_directory(harness):
    _, _, _, _, _, _, _, config = harness
    # A second lifespan obtains a separate OS lock handle, even in this test process.
    app = create_app(config,engine_factory=lambda _:pytest.fail('factory must not run'))
    with pytest.raises(RuntimeError,match='already owns'):
        with TestClient(app):
            pass

def test_factory_constructs_only_memory_provider_and_defers_sdk(tmp_path, monkeypatch):
    import bumi_mem.providers as providers
    import bumi_mem.long_term as ltm
    from bumi_mem.providers import FakeTextProvider
    created, native = [], []
    backend = FakeLongTermMemory()
    def summary(config):
        created.append(config)
        return FakeTextProvider('memory')
    def memory_factory(config, adapter):
        native.append((config,adapter))
        return backend
    monkeypatch.setattr(providers,'OpenAICompatibleProvider',summary)
    monkeypatch.setattr(ltm,'Mem0LongTermMemory',memory_factory)
    model = tmp_path/'model'; model.mkdir()
    config = ServiceSettings(data_dir=tmp_path/'data',embed_model_path=model,api_key=KEY,
                             memory_api_key='memory-secret',memory_model='memory-model')
    engine = build_service_engine(config)
    try:
        assert engine.chat_provider is None
        assert len(created) == 1 and created[0].api_key == 'memory-secret'
        assert created[0].model == 'memory-model'
        assert native == []
        engine.start()
        assert len(native) == 1 and native[0][1].embed_model == model
        assert backend.warmup_calls
    finally:
        engine.close()
