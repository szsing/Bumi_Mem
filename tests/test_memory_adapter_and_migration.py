from pathlib import Path
import sqlite3
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from bumi_mem.long_term import Mem0AdapterConfig, Mem0LongTermMemory
from bumi_mem.memory_access import SerialMemoryAccess
from bumi_mem.providers import ProviderConfig
from bumi_mem.schemas import MemoryWriteUncertainError, RetryableMemoryError
from bumi_mem.session_store import SQLiteSessionStore
from test_bumi_memory import make_engine, chat_turn, submit


@pytest.fixture
def sdk(monkeypatch):
    class LLMError(Exception):
        pass
    class SDK:
        rows = {}
        config = None
        last_add = None
        failure = None
        lose_write = False
        partial_delete = False
        @classmethod
        def from_config(cls, config):
            cls.config = config
            return cls()
        def add(self, messages, **kwargs):
            SDK.last_add = kwargs
            if SDK.failure:
                raise SDK.failure
            event = {'id':'m1','memory':'fact','event':'ADD'}
            if not SDK.lose_write:
                SDK.rows['m1'] = {'id':'m1','memory':'fact','user_id':kwargs['user_id'],
                                  'agent_id':kwargs['agent_id'],'metadata':kwargs['metadata']}
            return {'results':[event]}
        def get(self, mid):
            return SDK.rows.get(mid)
        def delete_all(self, **kwargs):
            if not SDK.partial_delete:
                SDK.rows.clear()
        def get_all(self, **kwargs):
            return {'results':list(SDK.rows.values())}
        def close(self):
            pass
    mod = types.ModuleType('mem0')
    mod.Memory = SDK
    errors = types.ModuleType('mem0.exceptions')
    errors.LLMError = LLMError
    monkeypatch.setitem(sys.modules, 'mem0', mod)
    monkeypatch.setitem(sys.modules, 'mem0.exceptions', errors)
    return SDK, LLMError


def adapter(tmp_path):
    return Mem0LongTermMemory(ProviderConfig('fake','fake-model','http://fake','fake'),
                              Mem0AdapterConfig(data_dir=tmp_path/'l3', embed_model=tmp_path/'model'))


def write(a, generation=1):
    return a.add_session(user_id='u', agent_id='a', messages=[{'role':'user','content':'fact'}],
                         metadata={'memory_generation':generation,'consolidation_id':'stable-chunk'}, prompt='rules')


def test_adapter_changes_extraction_cache_namespace_after_forget(tmp_path, sdk):
    raw, _ = sdk
    a = adapter(tmp_path)
    write(a, 3)
    assert raw.last_add['run_id'] == 'bumi-generation-3'
    write(a, 4)
    assert raw.last_add['run_id'] == 'bumi-generation-4'
    assert raw.last_add['infer'] is True


def test_adapter_llm_error_is_retryable_before_write(tmp_path, sdk):
    raw, error = sdk
    a = adapter(tmp_path)
    raw.failure = error('model unavailable')
    with pytest.raises(RetryableMemoryError):
        write(a)
    assert not raw.rows


def test_adapter_rejects_silent_insert_failure(tmp_path, sdk):
    raw, _ = sdk
    a = adapter(tmp_path)
    raw.lose_write = True
    with pytest.raises(MemoryWriteUncertainError, match='verification'):
        write(a)


def test_adapter_does_not_acknowledge_partial_delete(tmp_path, sdk):
    raw, _ = sdk
    a = adapter(tmp_path)
    write(a)
    raw.partial_delete = True
    with pytest.raises(RuntimeError, match='remaining'):
        a.delete_all(user_id='u', agent_id='a')


def test_adapter_factory_not_constructed_on_callers_thread(tmp_path, sdk):
    raw, _ = sdk
    creator = []
    access = SerialMemoryAccess(lambda: (creator.append(threading.get_ident()), adapter(tmp_path))[1], read_timeout=1)
    try:
        access.get_all(user_id='u', agent_id='a')
        assert creator and creator[0] != threading.get_ident()
    finally:
        access.close()


LEGACY_SCHEMA = '''
CREATE TABLE sessions (
 session_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,agent_id TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('active','processing','failed','consolidated','discarded')),
 started_at TEXT NOT NULL,closed_at TEXT,close_reason TEXT,summary TEXT,summary_provider TEXT,
 summary_status TEXT NOT NULL DEFAULT 'pending',summary_error TEXT,
 consolidation_status TEXT NOT NULL DEFAULT 'pending',consolidation_attempts INTEGER NOT NULL DEFAULT 0,
 last_error TEXT,created_at TEXT NOT NULL);
CREATE UNIQUE INDEX one_open_session_per_pair ON sessions(user_id,agent_id)
 WHERE status IN ('active','processing','failed');
CREATE TABLE messages (
 message_id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
 sequence_no INTEGER NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,token_count INTEGER NOT NULL,
 context_only INTEGER NOT NULL DEFAULT 0,source_session_id TEXT,created_at TEXT NOT NULL,consolidated_at TEXT,
 UNIQUE(session_id,sequence_no));
INSERT INTO sessions(session_id,user_id,agent_id,status,started_at,created_at)
 VALUES('old','u1','a1','processing','2026-09-04','2026-09-04');
INSERT INTO messages(message_id,session_id,sequence_no,role,content,token_count,created_at)
 VALUES('old-message','old',1,'user','legacy fact',5,'2026-09-04');
'''


def test_v01_schema_migrates_without_losing_messages(tmp_path):
    with sqlite3.connect(tmp_path/'bumi.sqlite3') as conn:
        conn.executescript(LEGACY_SCHEMA)
    engine, store, memory, _, _ = make_engine(tmp_path)
    active = store.get_open_session('u1','a1')
    assert active.session_id != 'old'
    assert store.list_messages('old')[0].content == 'legacy fact'
    assert store.list_messages(active.session_id)[0].context_only
    assert store.get_job('session_old').status == 'queued'
    engine.worker.run_once()
    assert store.get_job('session_old').status == 'succeeded'
    assert memory.memories[('u1','a1')][0]['memory'] == 'legacy fact'


@pytest.mark.parametrize("old_status", ["success", "processing", "failed"])
def test_legacy_partial_write_is_not_automatically_replayed(tmp_path, old_status):
    with sqlite3.connect(tmp_path/'bumi.sqlite3') as conn:
        conn.executescript(LEGACY_SCHEMA)
    store = SQLiteSessionStore(tmp_path/'bumi.sqlite3')
    with store._connect(write=True) as conn:
        conn.execute('INSERT INTO consolidation_runs(consolidation_id,session_id,attempt,chunk_index,chunk_count,provider,status) '
                     "VALUES('old-run','old',1,0,1,'deepseek',?)", (old_status,))
    engine, store, memory, _, _ = make_engine(tmp_path)
    assert store.get_job('session_old').status == 'failed'
    assert not engine.worker.run_once()
    assert chat_turn(engine).answer
    assert not memory.add_calls
    with pytest.raises(ValueError, match='unknown'):
        engine.retry_memory_job('session_old', user_id='u1', agent_id='a1')


def test_submission_rolls_back_rotation_if_queue_insert_fails(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    result = chat_turn(engine)
    with store._connect(write=True) as conn:
        conn.execute("CREATE TRIGGER reject_job BEFORE INSERT ON memory_jobs BEGIN SELECT RAISE(ABORT,'queue unavailable'); END")
    with pytest.raises(sqlite3.IntegrityError):
        submit(engine)
    assert store.get_session(result.session_id).status == 'active'
    assert store.get_open_session('u1','a1').session_id == result.session_id
    assert store.stats(result.session_id).effective_messages == 2
    assert store.list_jobs(user_id='u1', agent_id='a1') == []


def test_read_result_is_dropped_even_if_delete_completes_during_read(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    def stale_read(**kwargs):
        job = store.request_forget(user_id='u1',agent_id='a1')
        owner = 'deleter'
        claimed = store.claim_job(owner, lease_seconds=30)
        store.finish_job(claimed.job_id,owner)
        store.release_lease(owner)
        return [{'memory':'stale result'}]
    engine.long_term.search = stale_read
    assert engine.search_long_term(user_id='u1',agent_id='a1',query='q') == []


def test_cli_build_defers_native_sdk_construction(tmp_path, monkeypatch):
    import bumi_mem.cli as cli
    from bumi_mem.long_term import FakeLongTermMemory
    from bumi_mem.providers import FakeTextProvider, ProviderConfig
    from bumi_mem.token_counter import FixedTokenCounter
    monkeypatch.setattr(sys, 'argv', ['bumi', '--chat-provider', 'deepseek', '--memory-provider', 'deepseek'])
    args = cli.parse_args()
    args.db_path = str(tmp_path/'cli.sqlite3')
    monkeypatch.setattr(cli, 'configure_runtime', lambda: None)
    monkeypatch.setattr(cli, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(cli, 'MODEL_ROOT', tmp_path/'models')
    monkeypatch.setattr(cli, '_provider_config', lambda *a, **k: ProviderConfig('fake','fake-model','http://fake','fake'))
    monkeypatch.setattr(cli, '_token_counter', lambda _: FixedTokenCounter(1))
    monkeypatch.setattr(cli, 'OpenAICompatibleProvider', lambda _: FakeTextProvider())
    construction = []
    monkeypatch.setattr(cli, 'Mem0LongTermMemory', lambda *a, **k: (construction.append(threading.get_ident()), FakeLongTermMemory())[1])
    engine = cli.build_engine(args)
    try:
        assert construction == []
        engine.long_term.get_all(user_id='u', agent_id='a')
        assert construction[0] != threading.get_ident()
    finally:
        engine.close()



def test_newer_schema_is_not_silently_downgraded(tmp_path):
    path = tmp_path / 'future.sqlite3'
    with sqlite3.connect(path) as conn:
        conn.execute('PRAGMA user_version=999')
    with pytest.raises(RuntimeError, match='newer'):
        SQLiteSessionStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 999



def test_engine_rejects_native_sdk_constructed_outside_owner_thread(tmp_path):
    unsafe = object.__new__(Mem0LongTermMemory)
    with pytest.raises(ValueError, match='SerialMemoryAccess'):
        make_engine(tmp_path, long_term=unsafe)
