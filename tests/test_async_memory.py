from __future__ import annotations

import json
import sqlite3
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from bumi_mem.cli import run_interactive
from bumi_mem.long_term import FakeLongTermMemory
from bumi_mem.memory_access import SerialMemoryAccess
from bumi_mem.providers import FakeTextProvider
from bumi_mem.schemas import MemoryReadUnavailable, RetryableMemoryError, ProviderConfigurationError
from bumi_mem.session_store import SQLiteSessionStore
from test_bumi_memory import make_engine, chat_turn, submit


def wait_until(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), 'condition did not become true before deadline'


class SlowWrite(FakeLongTermMemory):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def add_session(self, **kwargs):
        self.entered.set()
        if not self.release.wait(5):
            raise RetryableMemoryError('test timed out before writing')
        return super().add_session(**kwargs)


def test_real_background_write_does_not_block_next_chat(tmp_path):
    memory = SlowWrite()
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory, auto_start=True)
    chat_turn(engine, '上一段')
    receipt = submit(engine)
    try:
        assert memory.entered.wait(2)
        start = time.monotonic()
        result = chat_turn(engine, '继续聊')
        assert time.monotonic() - start < 1.0
        assert result.answer and result.active_session_id == receipt.new_session_id
        assert result.recall_status == 'unavailable'
        assert result.recalled == []
        assert any(m['content'].split('\n', 1)[-1] == '上一段' for m in result.prompt_messages)
        assert store.get_job(receipt.job_id).status == 'running'
    finally:
        memory.release.set()
    wait_until(lambda: store.get_job(receipt.job_id).status == 'succeeded')


def test_slow_summary_also_does_not_block_next_chat(tmp_path):
    class SlowSummary(FakeTextProvider):
        def generate(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().generate(*args, **kwargs)
    entered, release = threading.Event(), threading.Event()
    engine, store, _, _, _ = make_engine(tmp_path, summary=SlowSummary(), auto_start=True)
    chat_turn(engine)
    receipt = submit(engine)
    try:
        assert entered.wait(2)
        start = time.monotonic()
        assert chat_turn(engine).answer
        assert time.monotonic() - start < 1.0
    finally:
        release.set()
    wait_until(lambda: store.get_job(receipt.job_id).status == 'succeeded')


def test_retry_budget_and_manual_retry_preserve_chat(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    memory.fail_add_count = 3
    for _ in range(3):
        assert engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'failed'
    assert store.get_job(receipt.job_id).attempts == 3
    assert not engine.worker.run_once()
    assert chat_turn(engine).answer
    engine.retry_memory_job(receipt.job_id, user_id='u1', agent_id='a1')
    engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'succeeded'
    assert store.get_job(receipt.job_id).attempts == 4


def test_backoff_does_not_hot_loop(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    engine.consolidator.config.retry_base_seconds = 30
    chat_turn(engine)
    receipt = submit(engine)
    memory.fail_add_count = 1
    engine.worker.run_once()
    job = store.get_job(receipt.job_id)
    assert job.status == 'retry_wait' and job.available_at > time.time() + 20
    assert not engine.worker.run_once()


def test_partial_success_skips_successful_chunk_on_retry(tmp_path):
    class FailSecond(FakeLongTermMemory):
        def __init__(self):
            super().__init__()
            self.n = 0
        def add_session(self, **kwargs):
            self.n += 1
            if self.n == 2:
                raise RetryableMemoryError('second chunk failed before writing')
            return super().add_session(**kwargs)
    engine, store, memory, _, summary = make_engine(tmp_path, long_term=FailSecond(),
                                                   memory_provider='local', local_chunk_tokens=10)
    chat_turn(engine, 'first')
    chat_turn(engine, 'second')
    receipt = submit(engine)
    engine.worker.run_once()
    assert store.chunks(receipt.job_id)[0]['status'] == 'succeeded'
    engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'succeeded'
    assert [m['memory'] for m in memory.memories[('u1','a1')]] == ['first', 'second']
    assert len(summary.calls) == 1


def test_queued_job_recovers_on_next_engine_start(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    chat_turn(engine, 'restart me')
    receipt = submit(engine)
    engine.close()
    restarted, store2, memory2, _, _ = make_engine(tmp_path, auto_start=True)
    wait_until(lambda: store2.get_job(receipt.job_id).status == 'succeeded')
    assert len(memory2.add_calls) == 1
    assert store2.get_open_session('u1','a1').session_id == receipt.new_session_id


def test_expired_lease_recovers_before_write(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    job = store.claim_job('dead-process', lease_seconds=30)
    with store._connect(write=True) as conn:
        conn.execute('UPDATE memory_worker_lease SET expires_at=0')
        conn.execute('UPDATE memory_jobs SET lease_until=0 WHERE job_id=?', (job.job_id,))
    engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'succeeded'
    assert len(memory.add_calls) == 1


def test_crash_during_write_requires_explicit_replay(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    job = store.claim_job('dead-process', lease_seconds=30)
    source = json.loads(job.payload_json)['messages'][0]
    store.ensure_chunks(job.job_id, job.owner, [[{'role': source['role'], 'content': source['content']}]])
    chunk = store.chunks(job.job_id)[0]
    store.set_chunk(job.job_id, job.owner, chunk['chunk_id'], status='writing')
    with store._connect(write=True) as conn:
        conn.execute('UPDATE memory_worker_lease SET expires_at=0')
        conn.execute('UPDATE memory_jobs SET lease_until=0 WHERE job_id=?', (job.job_id,))
    assert not engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'failed'
    assert not memory.add_calls
    with pytest.raises(ValueError, match='unknown'):
        engine.retry_memory_job(job.job_id, user_id='u1', agent_id='a1')
    engine.retry_memory_job(job.job_id, user_id='u1', agent_id='a1', allow_replay=True)
    engine.worker.run_once()
    assert store.get_job(job.job_id).status == 'succeeded'
    assert len(memory.add_calls) == 1
    replayed = memory.add_calls[0]['messages'][0]
    assert replayed['source_message_id'] == source['message_id']
    assert replayed['observed_at'] == source['created_at']
    assert replayed['content'].split('\n', 1)[1] == source['content']


def test_unknown_write_exception_is_not_blindly_retried(tmp_path):
    class WriteThenFail(FakeLongTermMemory):
        def add_session(self, **kwargs):
            super().add_session(**kwargs)
            raise OSError('response lost after commit')
    engine, store, memory, _, _ = make_engine(tmp_path, long_term=WriteThenFail())
    chat_turn(engine)
    receipt = submit(engine)
    engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'failed'
    assert store.chunks(receipt.job_id)[0]['status'] == 'uncertain'
    assert not engine.worker.run_once()
    assert len(memory.memories[('u1','a1')]) == 1


def test_second_worker_cannot_claim_live_owner_work(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    first = store.claim_job('worker-one', lease_seconds=30)
    assert first.job_id == receipt.job_id
    other = SQLiteSessionStore(store.path)
    assert other.claim_job('worker-two', lease_seconds=30) is None
    assert other.get_job(first.job_id).status == 'running'


def test_heartbeat_keeps_slow_job_owned(tmp_path):
    memory = SlowWrite()
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory)
    engine.worker.lease_seconds = 0.3
    chat_turn(engine)
    receipt = submit(engine)
    engine.start()
    try:
        assert memory.entered.wait(2)
        time.sleep(0.7)
        assert store.get_job(receipt.job_id).lease_until > time.time()
        assert store.claim_job('other-worker', lease_seconds=1) is None
    finally:
        memory.release.set()
    wait_until(lambda: store.get_job(receipt.job_id).status == 'succeeded')


def test_duplicate_submit_is_atomic_and_rotates_once(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    result = chat_turn(engine)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: engine.consolidator.submit(result.session_id, reason='test'), range(4)))
    assert len({r.job_id for r in receipts}) == 1
    assert len({r.new_session_id for r in receipts}) == 1
    assert len(store.list_jobs(user_id='u1', agent_id='a1')) == 1


def test_retry_is_scoped_to_user_and_agent(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path, max_attempts=1)
    chat_turn(engine)
    receipt = submit(engine)
    memory.fail_add_count = 1
    engine.worker.run_once()
    with pytest.raises(KeyError):
        engine.retry_memory_job(receipt.job_id, user_id='u2', agent_id='a1')
    assert store.get_job(receipt.job_id).status == 'failed'


def test_forget_cancels_queued_writes_and_excludes_old_l1(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine, 'old queued fact')
    old = submit(engine)
    chat_turn(engine, 'old active fact')
    deletion = engine.forget_all(user_id='u1', agent_id='a1')
    assert store.get_job(old.job_id).status == 'cancelled'
    assert store.reads_blocked('u1','a1')
    future = submit(engine)
    assert future.status == 'succeeded'  # all source messages predating forget were excluded
    engine.worker.run_once()
    assert store.get_job(deletion.job_id).status == 'succeeded'
    assert not memory.add_calls
    chat_turn(engine, 'new authorized fact')
    new = submit(engine)
    engine.worker.run_once()
    assert store.get_job(new.job_id).status == 'succeeded'
    assert memory.memories[('u1','a1')][0]['memory'] == 'new authorized fact'
    with pytest.raises(ValueError):
        engine.retry_memory_job(old.job_id, user_id='u1', agent_id='a1', allow_replay=True)


def test_forget_during_inflight_write_deletes_after_old_write(tmp_path):
    memory = SlowWrite()
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory, auto_start=True)
    chat_turn(engine, 'old fact')
    old = submit(engine)
    try:
        assert memory.entered.wait(2)
        deletion = engine.forget_all(user_id='u1', agent_id='a1')
        assert engine.get_memories(user_id='u1', agent_id='a1') == []
        assert chat_turn(engine, 'new question').recall_status == 'deleting'
    finally:
        memory.release.set()
    wait_until(lambda: store.get_job(deletion.job_id).status == 'succeeded')
    assert store.get_job(old.job_id).status == 'cancelled'
    assert memory.get_all(user_id='u1', agent_id='a1') == []


def test_failed_delete_keeps_reads_suppressed_until_manual_retry(tmp_path):
    class FailDelete(FakeLongTermMemory):
        fail = True
        def delete_all(self, **kwargs):
            if self.fail:
                raise RuntimeError('delete unavailable')
            return super().delete_all(**kwargs)
    memory = FailDelete()
    memory.memories[('u1','a1')] = [{'memory':'secret'}]
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory, max_attempts=1)
    deletion = engine.forget_all(user_id='u1', agent_id='a1')
    engine.worker.run_once()
    assert store.get_job(deletion.job_id).status == 'failed'
    assert engine.get_memories(user_id='u1', agent_id='a1') == []
    assert chat_turn(engine).answer
    memory.fail = False
    engine.retry_memory_job(deletion.job_id, user_id='u1', agent_id='a1')
    engine.worker.run_once()
    assert not store.reads_blocked('u1','a1')
    assert memory.get_all(user_id='u1', agent_id='a1') == []


def test_new_generation_write_waits_until_delete_succeeds(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    deletion = engine.forget_all(user_id='u1', agent_id='a1')
    chat_turn(engine, 'new fact')
    new = submit(engine)
    engine.worker.run_once()
    assert store.get_job(deletion.job_id).status == 'succeeded'
    assert store.get_job(new.job_id).status == 'queued'
    engine.worker.run_once()
    assert memory.get_all(user_id='u1', agent_id='a1')[0]['memory'] == 'new fact'


def test_sdk_factory_calls_and_close_share_one_thread():
    calls, closed = [], threading.Event()
    class ThreadBound:
        def __init__(self):
            self.db = sqlite3.connect(':memory:')
            calls.append(threading.get_ident())
        def search(self, **kwargs):
            self.db.execute('SELECT 1')
            calls.append(threading.get_ident())
            return []
        def add_session(self, **kwargs):
            self.db.execute('SELECT 1')
            calls.append(threading.get_ident())
            return []
        def close(self):
            self.db.close()
            calls.append(threading.get_ident())
            closed.set()
    access = SerialMemoryAccess(ThreadBound, read_timeout=1)
    access.search()
    access.add_session()
    access.close()
    assert closed.wait(2)
    assert len(set(calls)) == 1
    assert calls[0] != threading.get_ident()


def test_expired_reads_are_cancelled_not_replayed_after_write():
    memory = SlowWrite()
    access = SerialMemoryAccess(lambda: memory, read_timeout=0.01)
    errors = []
    def write():
        try:
            access.add_session(user_id='u', agent_id='a', messages=[{'role':'user','content':'x'}], metadata={}, prompt='')
        except Exception as e:
            errors.append(e)
    thread = threading.Thread(target=write)
    thread.start()
    try:
        assert memory.entered.wait(2)
        for _ in range(5):
            with pytest.raises(MemoryReadUnavailable):
                access.search(user_id='u', agent_id='a', query='q', top_k=5, threshold=0.1)
    finally:
        memory.release.set()
        thread.join(2)
        access.close()
    assert not errors
    assert memory.search_calls == []


def test_search_failure_degrades_without_aborting_answer(tmp_path):
    class BrokenSearch(FakeLongTermMemory):
        def search(self, **kwargs):
            raise OSError('search unavailable')
    engine, _, _, chat, _ = make_engine(tmp_path, long_term=BrokenSearch())
    result = chat_turn(engine)
    assert result.answer and result.recall_status == 'unavailable'
    assert len(chat.calls) == 1


def test_provider_change_does_not_silently_reprocess_job(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    engine.close()
    changed, _, memory, _, _ = make_engine(tmp_path, memory_provider='local')
    changed.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'failed'
    assert 'differs' in store.get_job(receipt.job_id).last_error
    assert not memory.add_calls


def cli_args():
    return Namespace(debug=True, user_id='u1', agent_id='a1', chat_provider='fake', memory_provider='fake',
                     max_messages=20, max_l1_tokens=8000)


def test_cli_quit_save_is_durable_submission_not_completion(tmp_path, monkeypatch, capsys):
    engine, store, memory, _, _ = make_engine(tmp_path)
    inputs = iter(['hello', '/quit', 'save'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    run_interactive(engine, cli_args())
    jobs = store.list_jobs(user_id='u1', agent_id='a1')
    assert jobs[0].status == 'queued'
    assert not memory.add_calls
    output = capsys.readouterr().out
    assert '下次启动恢复' in output
    assert '已保存并退出' not in output


def test_cli_eof_keeps_unsent_l1_and_closes_worker(tmp_path, monkeypatch):
    engine, store, _, _, _ = make_engine(tmp_path)
    chat_turn(engine, 'unsent')
    monkeypatch.setattr('builtins.input', lambda _: (_ for _ in ()).throw(EOFError()))
    run_interactive(engine, cli_args())
    assert store.stats(store.get_open_session('u1','a1').session_id).effective_messages == 2
    assert store.list_jobs(user_id='u1', agent_id='a1') == []
    assert engine._closed


def test_cli_jobs_and_new_report_receipts(tmp_path, monkeypatch, capsys):
    engine, _, _, _, _ = make_engine(tmp_path)
    inputs = iter(['hello', '/new', '/jobs', '/quit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    run_interactive(engine, cli_args())
    output = capsys.readouterr().out
    assert '长期记忆仍待后台完成' in output
    assert 'kind=consolidate' in output


def test_close_is_bounded_while_sdk_write_is_inflight(tmp_path):
    memory = SlowWrite()
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory, auto_start=True)
    chat_turn(engine)
    receipt = submit(engine)
    try:
        assert memory.entered.wait(2)
        start = time.monotonic()
        engine.close()
        assert time.monotonic() - start < 1.0
        assert store.get_job(receipt.job_id).status in ('running', 'succeeded')
    finally:
        memory.release.set()
    wait_until(lambda: not engine.worker._thread.is_alive())


def test_discard_current_session_does_not_cancel_previous_job(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine, 'keep previous')
    old = submit(engine)
    chat_turn(engine, 'discard current')
    engine.discard_session(user_id='u1', agent_id='a1')
    engine.worker.run_once()
    assert store.get_job(old.job_id).status == 'succeeded'
    assert memory.memories[('u1','a1')][0]['memory'] == 'keep previous'


def test_delete_suppression_survives_restart(tmp_path):
    memory = FakeLongTermMemory()
    memory.memories[('u1','a1')] = [{'memory':'old'}]
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory)
    deletion = engine.forget_all(user_id='u1', agent_id='a1')
    engine.close()
    other, _, _, _, _ = make_engine(tmp_path, long_term=memory)
    assert other.get_memories(user_id='u1', agent_id='a1') == []
    other.worker.run_once()
    assert store.get_job(deletion.job_id).status == 'succeeded'
    assert memory.memories.get(('u1','a1'), []) == []



def test_warmup_can_use_a_longer_timeout_than_foreground_reads():
    class SlowMemory(FakeLongTermMemory):
        def warmup(self, **kwargs):
            time.sleep(0.05)
            return super().warmup(**kwargs)

        def search(self, **kwargs):
            time.sleep(0.05)
            return super().search(**kwargs)

    access = SerialMemoryAccess(lambda: SlowMemory(), read_timeout=0.01)
    try:
        assert access.warmup(timeout=1.0, query='warmup') is None
        with pytest.raises(MemoryReadUnavailable):
            access.search(
                user_id='u1',
                agent_id='a1',
                query='q',
                top_k=1,
                threshold=0.1,
            )
    finally:
        access.close()


def test_engine_start_warms_backend_once_without_a_user_scope(tmp_path):
    engine, _, memory, _, _ = make_engine(tmp_path)
    try:
        elapsed_ms = engine.start()
        assert elapsed_ms >= 0
        assert memory.warmup_calls == ['记忆系统预热']
        assert memory.search_calls == []
        assert engine.start() == elapsed_ms
        assert memory.warmup_calls == ['记忆系统预热']
    finally:
        engine.close()


def test_factory_failure_marks_degraded_and_short_circuits_during_cooloff():
    calls = []

    def factory():
        calls.append(1)
        raise ConnectionError('connection refused')

    access = SerialMemoryAccess(factory, read_timeout=0.05, degrade_cooloff_seconds=60.0)
    assert access.status == 'unchecked'
    with pytest.raises(RetryableMemoryError):
        access.add_session(user_id='u', agent_id='a', messages=[], metadata={}, prompt='')
    assert access.status == 'degraded'
    assert len(calls) == 1
    # 冷却期内再调用：立即短路，不会再碰 factory。
    with pytest.raises(RetryableMemoryError):
        access.add_session(user_id='u', agent_id='a', messages=[], metadata={}, prompt='')
    assert len(calls) == 1
    access.close()


def test_recovers_after_cooloff_when_probe_succeeds():
    attempts = []

    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError('connection refused')
        return FakeLongTermMemory()

    access = SerialMemoryAccess(factory, read_timeout=0.5, degrade_cooloff_seconds=0.02)
    with pytest.raises(RetryableMemoryError):
        access.add_session(user_id='u', agent_id='a', messages=[], metadata={}, prompt='')
    assert access.status == 'degraded'
    time.sleep(0.05)
    access.add_session(user_id='u', agent_id='a', messages=[], metadata={}, prompt='')
    assert access.status == 'ready'
    assert len(attempts) == 2
    access.close()


def test_permanent_config_error_does_not_enter_degraded():
    def factory():
        raise ProviderConfigurationError('bad config')

    access = SerialMemoryAccess(factory, read_timeout=0.05)
    with pytest.raises(RetryableMemoryError):
        access.add_session(user_id='u', agent_id='a', messages=[], metadata={}, prompt='')
    assert access.status == 'unchecked'
    access.close()


def test_unknown_error_from_search_does_not_flip_circuit_but_still_raises():
    class Weird(FakeLongTermMemory):
        def search(self, **kwargs):
            raise KeyError('weird')

    access = SerialMemoryAccess(lambda: Weird(), read_timeout=0.5)
    with pytest.raises(KeyError):
        access.search(user_id='u', agent_id='a', query='q', top_k=1, threshold=0.1)
    assert access.status == 'unchecked'
    access.close()


def test_unknown_error_from_warmup_is_treated_as_degraded():
    class Weird(FakeLongTermMemory):
        def warmup(self, **kwargs):
            raise KeyError('weird')

    access = SerialMemoryAccess(lambda: Weird(), read_timeout=0.5)
    with pytest.raises(KeyError):
        access.warmup(timeout=1.0, query='q')
    assert access.status == 'degraded'
    access.close()


def test_ensure_available_probes_once_when_unchecked_and_becomes_ready():
    access = SerialMemoryAccess(lambda: FakeLongTermMemory(), read_timeout=0.5, degrade_probe_timeout=0.5)
    assert access.status == 'unchecked'
    assert access.ensure_available() is True
    assert access.status == 'ready'
    access.close()


def test_ensure_available_short_circuits_during_cooloff_without_touching_backend():
    calls = []

    def factory():
        calls.append(1)
        raise OSError('down')

    access = SerialMemoryAccess(factory, read_timeout=0.05, degrade_cooloff_seconds=60.0)
    assert access.ensure_available() is False
    assert access.status == 'degraded'
    assert len(calls) == 1
    assert access.ensure_available() is False
    assert len(calls) == 1
    access.close()


def test_ensure_available_raises_identified_permanent_error_instead_of_swallowing():
    def factory():
        raise ProviderConfigurationError('bad config')

    access = SerialMemoryAccess(factory, read_timeout=0.05)
    with pytest.raises(RetryableMemoryError):
        access.ensure_available()
    access.close()


def test_concurrent_calls_only_one_probe_reaches_backend_after_cooloff():
    entered, release = threading.Event(), threading.Event()
    hits = []

    class Gate(FakeLongTermMemory):
        def search(self, **kwargs):
            hits.append(1)
            entered.set()
            assert release.wait(2)
            return super().search(**kwargs)

    memory = Gate()
    attempts = {'n': 0}

    def factory():
        attempts['n'] += 1
        if attempts['n'] == 1:
            raise ConnectionError('still down')
        return memory

    access = SerialMemoryAccess(factory, read_timeout=0.5, degrade_cooloff_seconds=0.02)
    with pytest.raises(RetryableMemoryError):
        access.search(user_id='u', agent_id='a', query='q', top_k=1, threshold=0.1)
    assert access.status == 'degraded'
    time.sleep(0.05)
    errors = []

    def call():
        try:
            access.search(user_id='u', agent_id='a', query='q', top_k=1, threshold=0.1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(5)]
    for t in threads:
        t.start()
    assert entered.wait(2)
    time.sleep(0.05)
    release.set()
    for t in threads:
        t.join(2)
    assert len(hits) == 1
    assert len(errors) == 4
    access.close()


def test_concurrent_calls_only_one_probe_reaches_backend_when_unchecked():
    entered, release = threading.Event(), threading.Event()
    hits = []

    class Gate(FakeLongTermMemory):
        def search(self, **kwargs):
            hits.append(1)
            entered.set()
            assert release.wait(2)
            return super().search(**kwargs)

    access = SerialMemoryAccess(lambda: Gate(), read_timeout=0.5)
    assert access.status == 'unchecked'
    errors = []

    def call():
        try:
            access.search(user_id='u', agent_id='a', query='q', top_k=1, threshold=0.1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(5)]
    for t in threads:
        t.start()
    assert entered.wait(2)
    time.sleep(0.05)
    release.set()
    for t in threads:
        t.join(2)
    assert len(hits) == 1
    assert len(errors) == 4
    access.close()


def test_stale_success_cannot_overwrite_a_fresher_failure(monkeypatch):
    # 熔断在 status != 'ready' 时会用一把锁保证只有一个请求去探测，所以要复现"旧成功
    # 覆盖新故障"必须先让状态落在 ready（稳态，两边都不碰那把锁），才会有 A/B 真正并发、
    # 结果被乱序应用的空间——这正是 Codex 描述的生产场景：大量并发读写都在正常流动时，
    # 一个更早派发、更晚被调度去"应用结果"的成功调用，不能盖掉一个更晚派发、先被调度、
    # 真正失败了的调用。
    class Backend(FakeLongTermMemory):
        def search(self, **kwargs):
            raise ConnectionError('search failed')

    access = SerialMemoryAccess(lambda: Backend(), read_timeout=0.5)
    access.add_session(user_id='u', agent_id='a', messages=[], metadata={}, prompt='')  # seq=0，推进到 ready
    assert access.status == 'ready'

    reached_a, release_a = threading.Event(), threading.Event()
    original_record = access._record_outcome

    def patched(seq, *args, **kwargs):
        if seq == 1:  # A：seq=1，写操作，会成功
            reached_a.set()
            assert release_a.wait(2)
        return original_record(seq, *args, **kwargs)

    monkeypatch.setattr(access, '_record_outcome', patched)

    results = {}

    def call_a():
        results['a'] = access.add_session(user_id='u', agent_id='a', messages=[], metadata={}, prompt='')

    thread_a = threading.Thread(target=call_a)
    thread_a.start()
    assert reached_a.wait(2)  # A 已经在后端真正执行完（成功），正卡在"应用结果"这一步
    with pytest.raises(ConnectionError):
        access.search(user_id='u', agent_id='a', query='q', top_k=1, threshold=0.1)  # B，seq=2，失败
    assert access.status == 'degraded'  # B 的失败已经生效
    release_a.set()
    thread_a.join(2)
    assert 'a' in results  # A 本身确实成功了（没有抛异常），只是它的状态更新被拒绝
    assert access.status == 'degraded'  # 旧的成功（seq=0）不能覆盖新的故障（seq=1）
    access.close()


def test_fail_fast_outcome_does_not_suppress_an_earlier_pending_failure(monkeypatch):
    # A（seq=1，真实的连接失败，真正反映可用性）先派发，但调用方线程被人为卡住，还没
    # 应用结果；B（seq=2，参数错误，fail-fast，不反映可用性）后派发、先被调度、应用完毕。
    # fail-fast 的 B 不该占用 seq 位，否则 A 后面应用 seq=1 时会被 "1 <= 2" 误判成过期丢弃，
    # 一次真实的连接故障就这样消失，熔断永远停在 ready，worker 也不会再去探测恢复。
    class Backend(FakeLongTermMemory):
        def search(self, **kwargs):
            raise ConnectionError('search failed')
        def get_all(self, **kwargs):
            raise TypeError('bad params')

    access = SerialMemoryAccess(lambda: Backend(), read_timeout=0.5)
    access.add_session(user_id='u', agent_id='a', messages=[], metadata={}, prompt='')  # seq=0，推进到 ready
    assert access.status == 'ready'

    reached_a, release_a = threading.Event(), threading.Event()
    original_record = access._record_outcome

    def patched(seq, *args, **kwargs):
        if seq == 1:  # A：真实的连接失败
            reached_a.set()
            assert release_a.wait(2)
        return original_record(seq, *args, **kwargs)

    monkeypatch.setattr(access, '_record_outcome', patched)

    results = {}

    def call_a():
        try:
            access.search(user_id='u', agent_id='a', query='q', top_k=1, threshold=0.1)
        except Exception as exc:
            results['a'] = exc

    thread_a = threading.Thread(target=call_a)
    thread_a.start()
    assert reached_a.wait(2)  # A 已经在后端真正失败，正卡在"应用结果"这一步
    with pytest.raises(TypeError):
        access.get_all(user_id='u', agent_id='a')  # B，seq=2，fail-fast
    assert access.status == 'ready'  # fail-fast 不该改变状态，也不该占用 seq 位
    release_a.set()
    thread_a.join(2)
    assert isinstance(results.get('a'), ConnectionError)
    assert access.status == 'degraded'  # A 的真实故障最终必须生效，不能被 B 吞掉
    access.close()


def test_cuda_accelerator_and_oom_errors_are_fail_fast():
    import torch
    from bumi_mem.memory_access import _is_fail_fast

    try:
        raise torch.AcceleratorError('boom')
    except Exception as exc:
        assert _is_fail_fast(exc, lenient_unknown=True) is True
    try:
        raise torch.cuda.OutOfMemoryError('boom')
    except Exception as exc:
        assert _is_fail_fast(exc, lenient_unknown=True) is True


def test_cause_chain_names_includes_location_but_not_message():
    from bumi_mem.memory_access import _cause_chain_names

    try:
        raise RuntimeError('do not leak this api-key-abc123')
    except Exception as exc:
        chain = _cause_chain_names(exc)
    assert chain.startswith('RuntimeError@test_async_memory.py:')
    assert 'api-key-abc123' not in chain
