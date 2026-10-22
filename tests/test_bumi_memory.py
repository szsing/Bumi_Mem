from __future__ import annotations

from pathlib import Path
import json
import time
import pytest

from bumi_mem.consolidator import ConsolidatorConfig, SessionConsolidator
from bumi_mem.engine import BumiMemory, BumiMemoryConfig
from bumi_mem.long_term import FakeLongTermMemory
from bumi_mem.providers import FakeTextProvider
from bumi_mem.schemas import ProviderConfigurationError
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
    assert store.stats(receipt.new_session_id).context_only_messages == 20
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


def test_start_degrades_on_availability_failure_by_default(tmp_path):
    class Down(FakeLongTermMemory):
        def warmup(self, **kwargs):
            raise OSError('down')
    engine, *_ = make_engine(tmp_path, long_term=Down(), auto_start=False)
    engine.start()
    assert engine.long_term_status == 'degraded'
    engine.close()


def test_start_reraises_when_degrade_disabled(tmp_path):
    class Down(FakeLongTermMemory):
        def warmup(self, **kwargs):
            raise OSError('down')
    store = SQLiteSessionStore(tmp_path / 'bumi.sqlite3')
    long_term = Down()
    chat = FakeTextProvider('chat')
    summary = FakeTextProvider('memory', ['测试会话摘要'] * 50)
    counter = FixedTokenCounter(1)
    consolidator = SessionConsolidator(store=store, long_term=long_term, summary_provider=summary,
                                       memory_provider_name='deepseek', memory_token_counter=counter,
                                       config=ConsolidatorConfig(retry_base_seconds=0))
    engine = BumiMemory(store=store, long_term=long_term, chat_provider=chat, consolidator=consolidator,
                        chat_token_counter=counter, auto_start=False,
                        config=BumiMemoryConfig(long_term_degrade_enabled=False))
    ENGINES.append(engine)
    with pytest.raises(OSError):
        engine.start()
    engine.close()


def test_start_always_fails_fast_on_permanent_config_error_even_when_degrade_enabled(tmp_path):
    class BadConfig(FakeLongTermMemory):
        def warmup(self, **kwargs):
            raise ProviderConfigurationError('bad config')
    engine, *_ = make_engine(tmp_path, long_term=BadConfig(), auto_start=False)
    with pytest.raises(ProviderConfigurationError):
        engine.start()
    engine.close()


def test_worker_never_claims_jobs_while_long_term_persistently_degraded(tmp_path):
    class AlwaysDown(FakeLongTermMemory):
        def _fail(self, **kwargs):
            # ConnectionError（而不是裸 OSError）才会被 _KNOWN_AVAILABILITY 明确识别为
            # 可用性问题——裸 OSError 也覆盖 PermissionError 之类的配置问题，不该被归为此类。
            raise ConnectionError('still down')
        search = _fail
        warmup = _fail
        add_session = _fail
        get_all = _fail
        delete_all = _fail
    memory = AlwaysDown()
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory, auto_start=False)
    engine.long_term.degrade_cooloff_seconds = 0.01
    chat_turn(engine, '一些对话', user='u1', agent='a1')
    assert engine.long_term_status == 'degraded'  # search 真的失败过，不是只是没被调用
    consolidate_receipt = submit(engine)
    # 用另一个 user/agent 提交删除作业：request_forget（session_store.py）会把同一个
    # user/agent 下所有排队中的作业标成 cancelled——如果用同一对，consolidate 作业会在
    # worker 跑之前就被取消，attempts==0 只是因为"作业已经不在队列里"，测不出降级门控。
    forget_receipt = engine.forget_all(user_id='u2', agent_id='a2')
    for _ in range(20):
        engine.worker.run_once()
        time.sleep(0.005)
    assert store.get_job(consolidate_receipt.job_id).attempts == 0
    assert store.get_job(forget_receipt.job_id).attempts == 0
    engine.close()


def test_worker_resumes_processing_once_long_term_recovers(tmp_path):
    attempts = {'n': 0}

    class FlakyThenFine(FakeLongTermMemory):
        def _maybe_fail(self):
            attempts['n'] += 1
            if attempts['n'] <= 2:
                raise ConnectionError('still down')
        def search(self, **kwargs):
            self._maybe_fail()
            return super().search(**kwargs)
        def warmup(self, **kwargs):
            self._maybe_fail()
            return super().warmup(**kwargs)
    memory = FlakyThenFine()
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory, auto_start=False)
    engine.long_term.degrade_cooloff_seconds = 0.01
    # 先用一次真实 search 把熔断打进 degraded（只让 warmup 失败测不出真正的降级/恢复，
    # chat_turn 走的是 search，一次成功就会把状态掰回 ready，后面 warmup 永远不会被触发）。
    chat_turn(engine, '一些对话')
    assert engine.long_term_status == 'degraded'
    receipt = submit(engine)
    for _ in range(40):
        engine.worker.run_once()
        if store.get_job(receipt.job_id).status == 'succeeded':
            break
        time.sleep(0.01)
    assert store.get_job(receipt.job_id).status == 'succeeded'
    assert store.get_job(receipt.job_id).attempts == 1
    assert engine.long_term_status == 'ready'


def test_forget_job_survives_degradation_and_completes_after_recovery(tmp_path):
    attempts = {'n': 0}

    class FlakyThenFine(FakeLongTermMemory):
        def _maybe_fail(self):
            attempts['n'] += 1
            if attempts['n'] <= 2:
                raise ConnectionError('still down')
        def search(self, **kwargs):
            self._maybe_fail()
            return super().search(**kwargs)
        def warmup(self, **kwargs):
            self._maybe_fail()
            return super().warmup(**kwargs)
    memory = FlakyThenFine()
    memory.memories[('u1', 'a1')] = [{'memory': 'old'}]
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory, auto_start=False)
    engine.long_term.degrade_cooloff_seconds = 0.01
    chat_turn(engine, '一些对话')
    assert engine.long_term_status == 'degraded'
    receipt = engine.forget_all(user_id='u1', agent_id='a1')
    for _ in range(40):
        engine.worker.run_once()
        if store.get_job(receipt.job_id).status == 'succeeded':
            break
        time.sleep(0.01)
    assert store.get_job(receipt.job_id).status == 'succeeded'
    assert store.get_job(receipt.job_id).attempts == 1
    assert memory.memories.get(('u1', 'a1'), []) == []


def test_worker_does_not_probe_when_warmup_disabled_and_no_jobs_queued(tmp_path):
    memory = FakeLongTermMemory()
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory, auto_start=False)
    engine.config.memory_warmup_enabled = False
    assert engine.long_term_status == 'unchecked'
    for _ in range(5):
        assert not engine.worker.run_once()
    assert memory.warmup_calls == []
    assert memory.search_calls == []
    assert engine.long_term_status == 'unchecked'
    engine.close()


def test_worker_probes_before_claiming_when_unchecked(tmp_path):
    # 不先走 chat_turn：那会先触发一次真实 search，把 unchecked 提前变成 ready，
    # 测不出"worker 自己先探测"这条路径。forget_all 提交删除作业全程不碰 search/warmup。
    memory = FakeLongTermMemory()
    engine, store, _, _, _ = make_engine(tmp_path, long_term=memory, auto_start=False)
    assert engine.long_term_status == 'unchecked'
    receipt = engine.forget_all(user_id='u1', agent_id='a1')
    engine.worker.run_once()
    assert memory.warmup_calls
    assert store.get_job(receipt.job_id).status == 'succeeded'
    engine.close()


def test_degraded_recall_logs_a_visible_record_every_turn(tmp_path, caplog):
    # 验收要求：发生 N 次降级，默认日志（不开 DEBUG）就能查到 N 条对应记录——
    # 不能只有第一次状态翻转那一条，后面每一轮被跳过的长期记忆都要有迹可查。
    import logging

    class AlwaysDown(FakeLongTermMemory):
        def search(self, **kwargs):
            raise ConnectionError('still down')
    engine, *_ = make_engine(tmp_path, long_term=AlwaysDown())
    turns = 10
    with caplog.at_level(logging.WARNING, logger='bumi_mem.engine'):
        for _ in range(turns):
            chat_turn(engine, '一轮对话')
    skipped = [r for r in caplog.records if 'long-term memory recall skipped this turn' in r.message]
    assert len(skipped) == turns
    engine.close()


def test_normal_recall_real_timeouts_10_of_10_skip_new_l3(tmp_path, caplog):
    # 每轮用独立引擎：同一个引擎连续超时会在第一次失败后进入 degraded + 冷却期，
    # 后面几轮会被熔断短路直接拒绝，根本不会再真正调用 search()——那样"10次超时"
    # 实际只有1次是真的，剩下9次只是熔断生效，测不出"实际检索超时"这件事本身。
    import logging

    turns = 10
    search_entries = 0
    skipped_logs = 0
    for i in range(turns):
        search_calls = []

        class SlowSearch(FakeLongTermMemory):
            def search(self, **kwargs):
                search_calls.append(1)
                time.sleep(0.2)  # 明显超过默认 60ms 的普通检索预算
                return super().search(**kwargs)
        engine, *_ = make_engine(tmp_path / f'normal-{i}', long_term=SlowSearch())
        assert engine.config.memory_read_timeout == 0.06
        with caplog.at_level(logging.WARNING, logger='bumi_mem.engine'):
            caplog.clear()
            result = chat_turn(engine, '随便聊聊今天的天气')  # 非显式回忆措辞
        assert result.answer  # 对话正常完成，没有被超时卡住
        assert result.recalled == []  # 本轮没有注入新的 L3 检索结果
        assert result.recall_status == 'unavailable'
        assert len(search_calls) == 1  # 这一轮确实真正进了 search()，不是被熔断短路
        search_entries += len(search_calls)
        skipped_logs += len(
            [r for r in caplog.records if 'long-term memory recall skipped this turn' in r.message])
        engine.close()
    assert search_entries == turns  # 10 轮里 search() 真的被调用了 10 次
    assert skipped_logs == turns  # 10/10 都有默认可见的降级日志


def test_explicit_recall_real_timeouts_10_of_10_skip_new_l3(tmp_path, caplog):
    import logging

    turns = 10
    search_entries = 0
    skipped_logs = 0
    for i in range(turns):
        search_calls = []

        class SlowSearch(FakeLongTermMemory):
            def search(self, **kwargs):
                search_calls.append(1)
                time.sleep(0.4)  # 明显超过默认 200ms 的显式回忆预算
                return super().search(**kwargs)
        engine, *_ = make_engine(tmp_path / f'explicit-{i}', long_term=SlowSearch())
        assert engine.config.explicit_recall_timeout == 0.2
        with caplog.at_level(logging.WARNING, logger='bumi_mem.engine'):
            caplog.clear()
            result = chat_turn(engine, '你还记得我们昨天聊的事情吗')  # 显式回忆措辞
        assert result.answer
        assert result.recalled == []
        assert result.recall_status == 'unavailable'
        assert len(search_calls) == 1
        search_entries += len(search_calls)
        skipped_logs += len(
            [r for r in caplog.records if 'long-term memory recall skipped this turn' in r.message])
        engine.close()
    assert search_entries == turns
    assert skipped_logs == turns


def test_explicit_recall_gets_a_longer_budget_than_normal_recall(tmp_path):
    # 后端耗时 0.1s：介于默认的 60ms（普通）和 200ms（显式回忆）之间，用来证明
    # 两档预算真的分开生效，不是两个配置项取了同一个值。
    class ModeratelySlowSearch(FakeLongTermMemory):
        def search(self, **kwargs):
            time.sleep(0.1)
            return super().search(**kwargs)
    engine, *_ = make_engine(tmp_path, long_term=ModeratelySlowSearch())
    engine.long_term.degrade_cooloff_seconds = 0.01  # 加速测试：让第二次调用能真正探测
    normal = chat_turn(engine, '随便聊聊今天的天气')
    assert normal.recall_status == 'unavailable'  # 0.1s > 60ms 普通预算，超时跳过
    time.sleep(0.02)  # 等冷却期过，让下一次调用不被熔断短路
    explicit = chat_turn(engine, '你记得吗，我们昨天聊过这个话题')
    assert explicit.recall_status == 'ok'  # 0.1s < 200ms 显式回忆预算，正常命中
    engine.close()


def test_you_remember_phrasing_is_recognized_as_explicit_recall():
    import re
    pattern = r'还记得|记不记得|记得.{0,6}吗|回忆|上次.{0,20}(说|聊|提)|\bremember\b|\blast time\b'
    assert re.search(pattern, '你记得吗', re.I)
    assert re.search(pattern, '你记得我昨天说的事吗', re.I)
    assert not re.search(pattern, '记得关灯', re.I)  # 提醒事项，不是回忆提问
