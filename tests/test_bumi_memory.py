from __future__ import annotations

from pathlib import Path
import json
import pytest

from bumi_mem.consolidator import ConsolidatorConfig, SessionConsolidator
from bumi_mem.engine import BumiMemory, BumiMemoryConfig
from bumi_mem.long_term import FakeLongTermMemory
from bumi_mem.providers import FakeTextProvider
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import FixedTokenCounter

ENGINES = []


def make_engine(tmp_path: Path, *, max_messages=20, max_tokens=8000,
                memory_provider='deepseek', local_chunk_tokens=3500,
                long_term=None, summary=None, auto_start=False, max_attempts=3):
    store = SQLiteSessionStore(tmp_path / 'bumi.sqlite3')
    long_term = long_term if long_term is not None else FakeLongTermMemory()
    chat = FakeTextProvider('chat')
    summary = summary if summary is not None else FakeTextProvider('memory', ['测试会话摘要'] * 50)
    counter = FixedTokenCounter(1)
    consolidator = SessionConsolidator(store=store, long_term=long_term, summary_provider=summary,
                                       memory_provider_name=memory_provider, memory_token_counter=counter,
                                       config=ConsolidatorConfig(local_chunk_tokens=local_chunk_tokens,
                                                                retry_base_seconds=0, max_attempts=max_attempts))
    engine = BumiMemory(store=store, long_term=long_term, chat_provider=chat, consolidator=consolidator,
                        chat_token_counter=counter, auto_start=auto_start,
                        config=BumiMemoryConfig(max_messages=max_messages, max_l1_tokens=max_tokens))
    ENGINES.append(engine)
    return engine, store, long_term, chat, summary


def chat_turn(engine, text='你好', user='u1', agent='a1'):
    return engine.chat(user_id=user, agent_id=agent, text=text)


def submit(engine):
    return engine.end_session(user_id='u1', agent_id='a1')


def test_first_nineteen_messages_do_not_write_l3(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    for _ in range(9):
        chat_turn(engine)
    session = engine.get_or_create_session(user_id='u1', agent_id='a1')
    engine.record_user_message(session.session_id, '第十九条', user_id=session.user_id, agent_id=session.agent_id)
    assert store.stats(session.session_id).effective_messages == 19
    assert memory.add_calls == []
    assert engine.memory_jobs(user_id='u1', agent_id='a1') == []


def test_twentieth_message_submits_and_rotates_before_l3_write(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    for _ in range(10):
        result = chat_turn(engine)
    receipt = result.auto_consolidation
    assert receipt.accepted and receipt.status == 'queued'
    assert memory.add_calls == []
    assert store.get_session(receipt.session_id).status == 'processing'
    assert store.get_session(receipt.new_session_id).status == 'active'
    assert store.stats(receipt.new_session_id).context_only_messages == 4
    assert engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'succeeded'
    assert len(memory.add_calls) == 1


def test_token_limit_submits_after_answer(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path, max_tokens=10)
    result = chat_turn(engine)
    assert result.answer
    assert result.auto_consolidation.accepted
    assert store.get_job(result.auto_consolidation.job_id).reason == 'token_limit'
    assert not memory.add_calls


def test_manual_new_returns_durable_receipt(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    assert receipt.accepted and not hasattr(receipt, 'success')
    assert not memory.add_calls
    assert store.get_open_session('u1', 'a1').session_id == receipt.new_session_id


def test_current_query_only_is_used_for_l3_search(tmp_path):
    engine, _, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine, '之前的话')
    chat_turn(engine, '当前的问题')
    assert memory.search_calls[-1]['query'] == '当前的问题'


def test_current_user_message_appears_once_in_prompt(tmp_path):
    engine, _, _, _, _ = make_engine(tmp_path)
    result = chat_turn(engine, '唯一的输入')
    assert sum(m['content'].split('\n', 1)[-1] == '唯一的输入' for m in result.prompt_messages) == 1


def test_l3_failure_keeps_l1_and_reusable_l2(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    memory.fail_add_count = 1
    engine.worker.run_once()
    assert store.stats(receipt.session_id).effective_messages == 2
    assert store.get_session(receipt.session_id).summary_status == 'success'
    assert store.get_job(receipt.job_id).status == 'retry_wait'
    assert store.get_open_session('u1', 'a1').session_id == receipt.new_session_id


def test_retry_reuses_summary_and_stable_chunk_identity(tmp_path):
    engine, store, memory, _, summary = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    memory.fail_add_count = 1
    engine.worker.run_once()
    engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'succeeded'
    assert len(summary.calls) == 1
    assert len(memory.add_calls) == 2
    assert memory.add_calls[0]['metadata']['consolidation_id'] == memory.add_calls[1]['metadata']['consolidation_id']
    assert memory.add_calls[1]['metadata']['consolidation_attempt'] == 2


def test_l2_failure_l3_success_uses_fallback(tmp_path):
    engine, store, _, _, summary = make_engine(tmp_path)
    summary.error = RuntimeError('summary unavailable')
    chat_turn(engine)
    receipt = submit(engine)
    engine.worker.run_once()
    session = store.get_session(receipt.session_id)
    assert session.summary_status == 'fallback'
    assert '1 条用户消息' in session.summary
    assert store.get_job(receipt.job_id).status == 'succeeded'


def test_failed_job_does_not_block_new_chat(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path, max_attempts=1)
    chat_turn(engine)
    receipt = submit(engine)
    memory.fail_add_count = 10
    engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'failed'
    assert chat_turn(engine, '继续聊').answer
    assert store.get_open_session('u1', 'a1').status == 'active'


def test_restart_restores_active_session(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    result = chat_turn(engine, '重启前')
    engine.close()
    engine2, store2, _, _, _ = make_engine(tmp_path)
    assert store2.get_open_session('u1', 'a1').session_id == result.session_id
    assert store2.stats(result.session_id).effective_messages == 2


def test_carryover_is_context_only_and_not_reconsolidated(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine, '旧消息')
    receipt = submit(engine)
    assert all(m.context_only for m in store.list_messages(receipt.new_session_id))
    result = chat_turn(engine, '新消息')
    assert any(m['content'].split('\n', 1)[-1] == '旧消息' for m in result.prompt_messages)
    submit(engine)
    engine.worker.run_once()
    engine.worker.run_once()
    assert [m['content'].split('\n', 1)[1] for m in memory.add_calls[1]['messages'] if m['role'] == 'user'] == ['新消息']


def test_discard_never_writes_l3_and_has_no_carryover(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    current = engine.discard_session(user_id='u1', agent_id='a1')
    assert store.stats(current.session_id).context_only_messages == 0
    assert engine.memory_jobs(user_id='u1', agent_id='a1') == []
    assert not memory.add_calls


def test_user_agent_pairs_are_isolated(tmp_path):
    engine, store, _, _, _ = make_engine(tmp_path)
    ids = {chat_turn(engine, user=u, agent=a).session_id for u, a in [('u1','a1'),('u1','a2'),('u2','a1')]}
    assert len(ids) == 3


def test_forget_all_only_targets_current_pair(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    memory.memories[('u1','a1')] = [{'memory': 'private-one'}]
    memory.memories[('u1','a2')] = [{'memory': 'private-two'}]
    job = engine.forget_all(user_id='u1', agent_id='a1')
    assert engine.get_memories(user_id='u1', agent_id='a1') == []
    assert store.get_job(job.job_id).status == 'queued'
    engine.worker.run_once()
    assert store.get_job(job.job_id).status == 'succeeded'
    assert memory.memories[('u1','a2')] == [{'memory': 'private-two'}]


def test_answer_callback_runs_before_automatic_submission(tmp_path):
    engine, _, memory, _, _ = make_engine(tmp_path, max_messages=2)
    seen = []
    result = engine.chat(user_id='u1', agent_id='a1', text='hi', on_answer=lambda _: seen.append(len(memory.add_calls)))
    assert seen == [0] and result.auto_consolidation.accepted
    engine.worker.run_once()
    assert len(memory.add_calls) == 1


def test_l1_current_statement_is_later_than_conflicting_l3(tmp_path):
    engine, _, memory, _, _ = make_engine(tmp_path)
    memory.search_results = [{'memory': '用户喜欢详细回答'}]
    result = chat_turn(engine, '今天请简短回答')
    assert '用户喜欢详细回答' in result.prompt_messages[0]['content']
    assert result.prompt_messages[-1]['content'].split('\n', 1)[-1] == '今天请简短回答'


def test_new_on_empty_session_rotates_without_l3_call(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    receipt = submit(engine)
    assert receipt.status == 'succeeded'
    assert receipt.session_id != receipt.new_session_id
    assert not engine.worker.run_once()
    assert not memory.add_calls


def test_summary_receives_complete_frozen_session(tmp_path):
    engine, _, _, _, summary = make_engine(tmp_path)
    for index in range(3):
        chat_turn(engine, f'完整消息{index}')
    submit(engine)
    chat_turn(engine, '后续消息')
    engine.worker.run_once()
    transcript = summary.calls[0][-1]['content']
    assert all(f'完整消息{i}' in transcript for i in range(3))
    assert '后续消息' not in transcript


def test_l3_metadata_has_stable_session_and_chunk_identity(tmp_path):
    engine, _, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    engine.worker.run_once()
    meta = memory.add_calls[0]['metadata']
    assert meta['session_id'] == receipt.session_id
    assert meta['job_id'] == receipt.job_id
    assert meta['consolidation_id'].endswith(':chunk:0')
    assert meta['chunk_count'] == 1


def test_failed_l3_chunk_is_recorded_for_audit(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    memory.fail_add_count = 1
    engine.worker.run_once()
    rows = store.list_consolidation_runs(receipt.session_id)
    assert rows[0]['status'] == 'pending'
    assert 'fake Mem0 failure' in rows[0]['last_error']


def test_search_scope_contains_both_user_and_agent(tmp_path):
    engine, _, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine, user='alice', agent='bumi')
    assert memory.search_calls[-1]['user_id'] == 'alice'
    assert memory.search_calls[-1]['agent_id'] == 'bumi'
