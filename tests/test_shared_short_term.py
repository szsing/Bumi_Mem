from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from bumi_mem.context import ContextBuilder
from bumi_mem.long_term import FakeLongTermMemory
from bumi_mem.schemas import MemoryScopeError
from bumi_mem.session_store import SQLiteSessionStore
from test_bumi_memory import make_engine


def chat(engine, user, text, agent='bot'):
    return engine.chat(user_id=user, agent_id=agent, text=text)


def drain(engine):
    for _ in range(30):
        if not engine.worker.run_once():
            break


def test_b_sees_a_and_agent_speech_but_only_b_private_l3(tmp_path):
    engine, store, backend, provider, _ = make_engine(tmp_path)
    engine.remember(user_id='A', agent_id='bot', contents=['A_PRIVATE_L3'])
    engine.remember(user_id='B', agent_id='bot', contents=['B_PRIVATE_L3'])
    provider.responses = ['刚才回答了A']
    chat(engine, 'A', '我刚说了一句话')
    b = chat(engine, 'B', '你刚刚说什么')
    prompt = b.prompt_messages
    assert 'B_PRIVATE_L3' in prompt[0]['content']
    refs = json.loads(prompt[0]['content'].split('[共享 L3 引用 / shared memory references]\n', 1)[1].split('\n', 1)[0])
    assert any(r['owner_user_id'] == 'A' and r['memory'] == 'A_PRIVATE_L3' for r in refs)
    assert [r['memory'] for r in b.recalled] == ['B_PRIVATE_L3']
    assert 'A_PRIVATE_L3' not in prompt[0]['content'].split('[共享 L3 引用 / shared memory references]', 1)[0]
    assert [m['role'] for m in prompt[1:]] == ['user', 'assistant', 'user']
    assert '我刚说了一句话' in prompt[1]['content']
    assert '刚才回答了A' in prompt[2]['content']
    assert '"user_id": "A"' in prompt[1]['content']
    assert '"user_id"' not in prompt[2]['content'].split("\n", 1)[0]
    assert '"speaker_role": "assistant"' in prompt[2]['content']
    assert '"user_id": "B"' in prompt[3]['content']
    assert all('"agent_id": "bot"' in row['content'] for row in prompt[1:])
    assert backend.search_calls[-1]['user_id'] == 'B'
    assert engine.session_stats(user_id='A', agent_id='bot').effective_messages == 4
    assert engine.session_stats(user_id='B', agent_id='bot').effective_messages == 4
    assert [(m.user_id, m.agent_id) for m in store.list_shared_messages('bot')] == [('A','bot'),('','bot'),('B','bot'),('','bot')]


def test_agent_boundary_is_not_shared(tmp_path):
    engine, _, _, _, _ = make_engine(tmp_path)
    chat(engine, 'A', 'OTHER_AGENT', agent='other')
    result = chat(engine, 'B', '这里刚才说什么')
    assert 'OTHER_AGENT' not in str(result.prompt_messages)


def test_twenty_message_boundary_is_global_and_owner_jobs_are_separate(tmp_path):
    engine, store, backend, _, _ = make_engine(tmp_path)
    for i in range(10):
        result = chat(engine, 'A' if i % 2 == 0 else 'B', f'FACT_{i}')
        assert bool(result.auto_consolidation) == (i == 9)
    jobs = [store.list_jobs(user_id=u, agent_id='bot')[0] for u in ['A','B']]
    assert all(j.status == 'queued' for j in jobs)
    for job in jobs:
        rows = json.loads(job.payload_json)['messages']
        assert len(rows) == 10
        assert all(m['user_id'] == (job.user_id if m['role'] == 'user' else '') and m['agent_id'] == 'bot' for m in rows)
    tail = store.list_shared_messages('bot')
    assert len(tail) == 4 and all(m.context_only for m in tail)
    assert [m.user_id for m in tail] == ['A','','B','']
    assert engine.session_stats(user_id='B', agent_id='bot').effective_messages == 0
    drain(engine)
    assert all(store.get_job(j.job_id).status == 'succeeded' for j in jobs)
    for call in backend.add_calls:
        expected = range(0,10,2) if call['user_id'] == 'A' else range(1,10,2)
        assert [m['content'].split('\n', 1)[1] for m in call['messages'] if m['role']=='user'] == [f'FACT_{i}' for i in expected]


def test_carryover_shared_after_restart_and_not_reextracted(tmp_path):
    engine, store, backend, _, _ = make_engine(tmp_path, max_messages=4)
    chat(engine, 'A', 'OLD_A')
    chat(engine, 'B', 'OLD_B')
    drain(engine)
    engine.close()
    restored, store2, _, _, _ = make_engine(tmp_path, max_messages=4, long_term=backend)
    new = chat(restored, 'C', 'NEW_C')
    assert 'OLD_A' in str(new.prompt_messages) and 'OLD_B' in str(new.prompt_messages)
    assert '"user_id": "A"' in str(new.prompt_messages)
    restored.end_session(user_id='C', agent_id='bot')
    drain(restored)
    calls = [call for call in backend.add_calls if call['user_id']=='C']
    assert len(calls) == 1
    assert [m['content'].split('\n', 1)[1] for m in calls[0]['messages'] if m['role']=='user'] == ['NEW_C']
    assert len([c for c in backend.add_calls if c['user_id']=='A']) == 1
    assert len([c for c in backend.add_calls if c['user_id']=='B']) == 1


def test_manual_rotation_enqueues_other_users_even_if_caller_has_no_turn(tmp_path):
    engine, store, backend, _, _ = make_engine(tmp_path)
    chat(engine, 'A', 'A_ONLY')
    receipt = engine.end_session(user_id='B', agent_id='bot')
    assert store.get_job(receipt.job_id).user_id == 'B'
    assert store.list_jobs(user_id='A', agent_id='bot')[0].status == 'queued'
    drain(engine)
    assert [call['user_id'] for call in backend.add_calls] == ['A']


def test_rotation_transaction_rolls_back_all_owners_on_error(tmp_path, monkeypatch):
    engine, store, _, _, _ = make_engine(tmp_path)
    chat(engine, 'A', 'a'); chat(engine, 'B', 'b')
    before = store.list_shared_messages('bot')
    original = store._submit
    count = 0
    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('injected')
        return original(*args, **kwargs)
    monkeypatch.setattr(store, '_submit', fail_second)
    with pytest.raises(RuntimeError, match='injected'):
        engine.end_session(user_id='B', agent_id='bot')
    assert store.list_shared_messages('bot') == before
    assert all(store.get_open_session(u,'bot').status == 'active' for u in ['A','B'])
    assert not store.list_jobs(user_id='A',agent_id='bot')
    assert not store.list_jobs(user_id='B',agent_id='bot')


def test_delete_one_users_l3_does_not_delete_other_users_job(tmp_path):
    engine, store, backend, _, _ = make_engine(tmp_path)
    chat(engine,'A','A_OLD'); chat(engine,'B','B_KEEP')
    engine.forget_all(user_id='A',agent_id='bot')
    assert 'A_OLD' in str([m.content for m in store.list_shared_messages('bot')])  # L3-only delete
    engine.end_session(user_id='B',agent_id='bot')
    drain(engine)
    assert not [c for c in backend.add_calls if c['user_id']=='A']
    assert [c for c in backend.add_calls if c['user_id']=='B']


def test_global_discard_only_clears_this_agents_live_window(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    chat(engine,'A','keep queued')
    job = engine.end_session(user_id='A',agent_id='bot')
    chat(engine,'A','a'); chat(engine,'B','b'); chat(engine,'A','other',agent='other')
    engine.discard_session(user_id='B',agent_id='bot')
    assert store.list_shared_messages('bot') == []
    assert len(store.list_shared_messages('other')) == 2
    assert store.get_job(job.job_id).status == 'queued'
    assert store.get_open_session('A','bot') is None
    assert 'a' not in [m.content for m in SQLiteSessionStore(store.path).list_shared_messages('bot')]


def test_same_agent_turns_are_serialized_across_users(tmp_path):
    engine, store, _, provider, _ = make_engine(tmp_path)
    entered, release, b_attempt = threading.Event(), threading.Event(), threading.Event()
    def generate(messages, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return 'reply'
    provider.generate = generate
    def second():
        b_attempt.set()
        return chat(engine,'B','b')
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(chat, engine, 'A', 'a')
        assert entered.wait(3)
        b = pool.submit(second)
        assert b_attempt.wait(3)
        try:
            assert engine._pair_lock('A','bot') is engine._pair_lock('B','bot')
            assert [m.user_id for m in store.list_shared_messages('bot')] == ['A']
        finally:
            release.set()
        a.result(); b.result()
    assert [(m.user_id,m.role) for m in store.list_shared_messages('bot')] == [
        ('A','user'),('','assistant'),('B','user'),('','assistant')]


def test_migrate_v2_backfills_speaker_and_only_imports_live_window_once(tmp_path):
    path = tmp_path/'old.sqlite3'
    store = SQLiteSessionStore(path)
    a = store.create_session('A','bot')
    store.append_message(a.session_id,role='user',content='a',token_count=1)
    store.append_message(a.session_id,role='assistant',content='reply a',token_count=1)
    b = store.create_session('B','bot')
    store.append_message(b.session_id,role='user',content='b',token_count=1)
    dead = store.create_session('dead','other')
    store.append_message(dead.session_id,role='user',content='discarded',token_count=1)
    store.discard_and_create(dead.session_id)
    # Reproduce v2 on-disk messages (no identity columns or shared table).
    with sqlite3.connect(path) as c:
        c.execute('DROP TABLE shared_l1')
        c.execute('ALTER TABLE messages DROP COLUMN user_id')
        c.execute('ALTER TABLE messages DROP COLUMN agent_id')
        c.execute('PRAGMA user_version=2')
    migrated = SQLiteSessionStore(path)
    assert [(m.user_id,m.content) for m in migrated.list_shared_messages('bot')] == [('A','a'),('','reply a'),('B','b')]
    assert migrated.list_shared_messages('other') == []
    reopened = SQLiteSessionStore(path)
    assert reopened.list_shared_messages('bot') == migrated.list_shared_messages('bot')
    with sqlite3.connect(path) as c:
        assert c.execute('PRAGMA user_version').fetchone()[0] == 8


def test_tampered_owner_snapshot_is_rejected_before_llm_or_write(tmp_path):
    engine, store, backend, _, summary = make_engine(tmp_path)
    chat(engine,'A','A')
    job = engine.end_session(user_id='A',agent_id='bot')
    payload = json.loads(store.get_job(job.job_id).payload_json)
    payload['messages'][0]['user_id'] = 'B'
    with sqlite3.connect(store.path) as c:
        c.execute('UPDATE memory_jobs SET payload_json=? WHERE job_id=?',(json.dumps(payload),job.job_id))
    drain(engine)
    assert store.get_job(job.job_id).status == 'failed'
    assert backend.add_calls == [] and summary.calls == []


def test_shared_l1_does_not_relax_explicit_foreign_session_or_memory_access(tmp_path):
    engine, _, _, _, _ = make_engine(tmp_path)
    a = chat(engine,'A','a')
    with pytest.raises(MemoryScopeError):
        engine.record_user_message(a.session_id,'fake',user_id='B',agent_id='bot')
    saved = engine.remember(user_id='A',agent_id='bot',contents=['private'])
    assert engine.read_memory(user_id='B',agent_id='bot',memory_id=saved.records[0].memory_id) is None


def test_global_token_limit_rotates_even_when_each_users_stream_is_small(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path, max_messages=100, max_tokens=20)
    first = chat(engine, 'A', 'a')
    assert first.auto_consolidation is None  # 2 * (fixed counter 1 + overhead 4)
    second = chat(engine, 'B', 'b')
    assert second.auto_consolidation is not None
    assert store.get_job(second.auto_consolidation.job_id).reason == 'token_limit'
    assert store.list_jobs(user_id='A', agent_id='bot')[0].reason == 'token_limit'


def test_no_recall_still_shares_short_context(tmp_path):
    engine, _, backend, _, _ = make_engine(tmp_path)
    chat(engine,'A','public speech')
    before = len(backend.search_calls)
    result = engine.chat(user_id='B',agent_id='bot',text='what did you say',recall_enabled=False)
    assert 'public speech' in str(result.prompt_messages)
    assert result.recalled == [] and result.recall_status == 'disabled'
    assert len(backend.search_calls) == before


def test_v2_queued_snapshot_without_speaker_fields_still_executes(tmp_path):
    engine, store, backend, _, _ = make_engine(tmp_path)
    chat(engine, 'A', 'old task')
    job = engine.end_session(user_id='A',agent_id='bot')
    payload = json.loads(store.get_job(job.job_id).payload_json)
    for row in payload['messages']:
        row.pop('user_id'); row.pop('agent_id')
    with sqlite3.connect(store.path) as c:
        c.execute('UPDATE memory_jobs SET payload_json=? WHERE job_id=?',(json.dumps(payload),job.job_id))
    drain(engine)
    assert store.get_job(job.job_id).status == 'succeeded'
    assert backend.add_calls[0]['user_id'] == 'A'


def test_rotating_empty_window_does_not_duplicate_shared_carryover(tmp_path):
    engine, store, backend, _, _ = make_engine(tmp_path)
    chat(engine,'A','one')
    engine.end_session(user_id='A',agent_id='bot')
    for user in ['B','C','D']:
        engine.end_session(user_id=user,agent_id='bot')
    rows = store.list_shared_messages('bot')
    assert len(rows) == 2 and len({m.message_id for m in rows}) == 2
    assert all(m.user_id == ('A' if m.role == 'user' else '') and m.context_only for m in rows)
    drain(engine)
    assert len(backend.add_calls) == 1


def test_cli_switch_reuses_shared_window_before_model_call(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from bumi_mem.cli import run_interactive
    engine, _, backend, provider, _ = make_engine(tmp_path)
    args = Namespace(user_id='A',agent_id='bot',debug=False,chat_provider='fake',
                     memory_provider='fake',max_messages=20,max_l1_tokens=8000)
    commands = iter(['A刚才说的话','/user B','你刚刚说什么','/quit','discard'])
    monkeypatch.setattr('builtins.input', lambda _: next(commands))
    run_interactive(engine,args)
    assert len(provider.calls) == 2
    assert 'A刚才说的话' in str(provider.calls[1])
    assert '"user_id": "A"' in str(provider.calls[1])
    assert '"user_id": "B"' in str(provider.calls[1])
    assert [c['user_id'] for c in backend.search_calls] == ['A','B']
    assert '共享 agent=bot' in capsys.readouterr().out
