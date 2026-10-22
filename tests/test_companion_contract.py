"""End-to-end offline acceptance of the companion-memory contract (no model calls)."""
import json
import sqlite3
import pytest

from test_bumi_memory import make_engine, chat_turn
from bumi_mem.identity import GUEST_USER_ID
from bumi_mem.schemas import IdentityRequiredError
from bumi_mem.turn_store import TurnConflict
from bumi_mem.semantic_state import SemanticScopeError


def exchange(engine, user='A', query='你好', previous=None, key=None, **kwargs):
    return engine.process_context(user_id=user, agent_id='bot', query=query,
                                  previous_ai_reply=previous, idempotency_key=key, **kwargs)


def text(response):
    return '\n'.join(m['content'] for m in response['messages'])


def test_guest_has_shared_references_but_never_personal_backend_access(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    engine.remember(user_id='A', agent_id='bot', contents=['A_PRIVATE_BEFORE_PUBLICATION'])
    exchange(engine, 'A', '请回忆一下')
    calls = len(memory.search_calls)
    guest = exchange(engine, GUEST_USER_ID, '刚才说了什么', previous='刚才的话')
    assert 'A_PRIVATE_BEFORE_PUBLICATION' in text(guest)
    assert 'unconfirmed' in text(guest)
    assert guest['recall_status'] == 'disabled'
    assert len(memory.search_calls) == calls
    for action in [lambda: engine.remember(user_id=GUEST_USER_ID, agent_id='bot', contents=['no']),
                   lambda: engine.get_memories(user_id=GUEST_USER_ID, agent_id='bot'),
                   lambda: engine.read_memory(user_id=GUEST_USER_ID, agent_id='bot', memory_id='x')]:
        with pytest.raises(IdentityRequiredError):
            action()
    exchange(engine, GUEST_USER_ID, None, previous='收到')
    engine.end_session(user_id=GUEST_USER_ID, agent_id='bot')
    while engine.worker.run_once():
        pass
    assert all(c['user_id'] != GUEST_USER_ID for c in memory.add_calls)
    assert store.semantic_context('A', 'bot') == []


def test_shared_reference_does_not_renew_or_count_as_speech_and_expires(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path, max_messages=4)
    engine.consolidator.config.carryover_messages = 0
    engine.remember(user_id='A', agent_id='bot', contents=['WINDOW_ONLY_FACT'])
    exchange(engine, 'A', 'first')
    assert len(store.list_shared_messages('bot')) == 1
    exchange(engine, 'B', 'second', previous='one')
    third = exchange(engine, GUEST_USER_ID, 'third', previous='two')
    assert 'WINDOW_ONLY_FACT' not in text(third)
    while engine.worker.run_once():
        pass
    assert all('WINDOW_ONLY_FACT' not in m['content'] for c in memory.add_calls for m in c['messages'])


def test_repeat_recall_does_not_move_anchor_or_duplicate_on_score_change(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    memory.search_results = [{'id': 'm1', 'memory': 'same', 'score': .3}]
    exchange(engine, 'A', 'one')
    refs = store.shared_memory_refs('bot', selected_message_ids=[m.message_id for m in store.list_shared_messages('bot')])
    memory.search_results[0]['score'] = .8
    exchange(engine, 'A', 'two', previous='yes')
    newer = store.shared_memory_refs('bot', selected_message_ids=[m.message_id for m in store.list_shared_messages('bot')])
    assert len(newer) == 1
    assert newer[0]['anchor_message_id'] == refs[0]['anchor_message_id']


def test_explicit_event_time_is_source_and_retries_validate_it(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    kwargs = dict(query_occurred_at='2026-09-21T00:05:00+08:00')
    first = exchange(engine, 'A', '昨天我去了一趟游乐场', key='t1', **kwargs)
    assert '2026-09-21T00:05:00+08:00' in text(first)
    assert first == exchange(engine, 'A', '昨天我去了一趟游乐场', key='t1', **kwargs)
    with pytest.raises(TurnConflict):
        exchange(engine, 'A', '昨天我去了一趟游乐场', key='t1', query_occurred_at='2026-09-22T00:05:00+08:00')
    exchange(engine, 'A', None, previous='知道了', previous_reply_occurred_at='2026-09-21T00:06:00+08:00')
    receipt = engine.end_session(user_id='A', agent_id='bot')
    engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'succeeded'
    assert '2026-09-20' in memory.add_calls[0]['messages'][0]['content']
    assert memory.add_calls[0]['metadata']['source_messages'][0]['observed_at'] == kwargs['query_occurred_at']


def test_core_is_persisted_before_reply_survives_restart_and_read_failure(tmp_path, monkeypatch):
    engine, store, memory, _, _ = make_engine(tmp_path)
    first = exchange(engine, 'A', '以后叫我林老师', key='name')
    assert first['core_write']['status'] == 'persisted'
    assert exchange(engine, 'A', '以后叫我林老师', key='name') == first
    exchange(engine, 'A', None, previous='好的')
    engine.close()
    engine2, store2, memory2, _, _ = make_engine(tmp_path)
    monkeypatch.setattr(memory2, 'search', lambda **kwargs: (_ for _ in ()).throw(RuntimeError('offline')))
    result = exchange(engine2, 'A', '聊聊别的')
    assert '林老师' in result['messages'][0]['content']
    assert result['recall_status'] == 'unavailable'
    assert len(store2.semantic_context('A', 'bot')) == 1


def test_failed_core_update_keeps_unrelated_rules_and_reports_failure(tmp_path, monkeypatch):
    engine, store, memory, _, _ = make_engine(tmp_path)
    exchange(engine, 'A', '请不要评价我的外貌')
    monkeypatch.setattr(store, 'apply_semantic_turn', lambda **kwargs: (_ for _ in ()).throw(RuntimeError('write failed')))
    failed = exchange(engine, 'A', '以后叫我林老师', previous='好的')
    assert failed['core_write']['status'] == 'failed'
    assert 'appearance' in failed['messages'][0]['content']
    assert any(m['content'].endswith('以后叫我林老师') for m in failed['messages'])
    assert len(store.semantic_context('A', 'bot')) == 1


def test_clear_correction_supersedes_old_reference_without_erasing_history(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    memory.search_results = [{'id': 'old', 'memory': '[用户偏好] 我喜欢咖啡'}]
    exchange(engine, 'A', '我喜欢咖啡')
    corrected = exchange(engine, 'A', '我不喜欢咖啡', previous='好的')
    assert corrected['core_write']['status'] == 'persisted'
    assert 'old' not in json.dumps(store.shared_memory_refs('bot', selected_message_ids=[m.message_id for m in store.list_shared_messages('bot')]))
    with store._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM semantic_versions').fetchone()[0] == 2
    assert store.semantic_context('A', 'bot')[0]['value'] == 'dislike'


def test_delete_revokes_other_users_cached_prompt_and_excludes_delayed_reply(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    engine.remember(user_id='A', agent_id='bot', contents=['TO_BE_FORGOTTEN'])
    exchange(engine, 'A', 'hi')
    exchange(engine, 'B', 'what', previous='hello', key='b')
    engine.forget_all(user_id='A', agent_id='bot')
    with pytest.raises(TurnConflict, match='shared_memory_context_changed'):
        exchange(engine, 'B', 'what', previous='hello', key='b')
    exchange(engine, GUEST_USER_ID, 'continue', previous='TO_BE_FORGOTTEN')
    with store._connect() as conn:
        assert conn.execute("SELECT memory_excluded FROM messages WHERE role='assistant' AND content='TO_BE_FORGOTTEN'").fetchone()[0] == 1
    # Actual spoken words remain in L1 by contract; stale retrieved copies do not.
    refs = store.shared_memory_refs('bot', selected_message_ids=[m.message_id for m in store.list_shared_messages('bot')])
    assert refs == []


def test_targeted_delete_suppresses_source_core_old_jobs_and_future_read(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    exchange(engine, 'A', '请叫我小林')
    source = next(m for m in store.list_shared_messages('bot') if m.role == 'user')
    row = {'id': 'source-backed', 'memory': '用户希望叫小林', 'user_id': 'A', 'agent_id': 'bot',
           'metadata': {'source': 'l1_session', 'source_message_ids': [source.message_id]}}
    memory.memories[('A', 'bot')] = [row]
    exchange(engine, 'A', None, previous='好的')
    pending = engine.end_session(user_id='A', agent_id='bot')
    receipt = engine.forget_memory(user_id='A', agent_id='bot', memory_id=row['id'])
    assert receipt['status'] == 'deleted'
    assert store.get_job(pending.job_id).status == 'cancelled'
    assert store.semantic_context('A', 'bot') == []
    assert engine.get_memories(user_id='A', agent_id='bot') == []
    assert engine.read_memory(user_id='A', agent_id='bot', memory_id=row['id']) is None
    assert engine.search_long_term(user_id='A', agent_id='bot', query='叫什么') == []
    with pytest.raises(SemanticScopeError, match='withdrawn'):
        store.apply_semantic_turn('A', 'bot', source.message_id, source.content, source.created_at, 0)


def test_guest_shared_bucket_never_consolidates_even_legacy_queued_job(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    chat_turn(engine, user=GUEST_USER_ID, agent='bot', text='我喜欢绿茶')
    receipt = engine.end_session(user_id=GUEST_USER_ID, agent_id='bot')
    assert receipt.status == 'succeeded'
    assert store.get_job(receipt.job_id).payload_json.find('我喜欢绿茶') < 0
    assert not engine.worker.run_once()
    assert memory.add_calls == []


def test_actual_robot_promise_has_assistant_provenance_and_no_execution(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    exchange(engine, 'A', '我们约定每天练习')
    exchange(engine, 'A', '继续', previous='我答应你陪你练习')
    rows = store.semantic_context('A', 'bot')
    assert len(rows) == 1
    assert rows[0]['category'] == 'robot_promise'
    assert rows[0]['source_role'] == 'assistant'
    assert rows[0]['status'] == 'promised'
    assert rows[0]['source_text'] == '我答应你陪你练习'


def test_full_deletion_removes_core_and_retry_cannot_recreate_from_old_source(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    exchange(engine, 'A', '别再叫我宝宝，我不喜欢')
    assert store.semantic_context('A', 'bot')
    engine.forget_all(user_id='A', agent_id='bot')
    assert store.semantic_context('A', 'bot') == []
    engine.worker.run_once()
    assert store.semantic_context('A', 'bot') == []


def test_promise_flush_retry_remains_idempotent(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    exchange(engine, 'A', 'hi')
    result = exchange(engine, 'A', 'next', previous='我答应你陪你练习', key='promise-reply')
    epoch = store.context_epoch('bot')
    assert exchange(engine, 'A', 'next', previous='我答应你陪你练习', key='promise-reply') == result
    assert store.context_epoch('bot') == epoch
    assert len(store.semantic_context('A', 'bot')) == 1


def test_history_question_can_read_superseded_versions_but_not_deleted_data(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    exchange(engine, 'A', '以后叫我小林')
    exchange(engine, 'A', '以后叫我林老师', previous='好的')
    history = exchange(engine, 'A', '以前你叫我什么', previous='好的')
    assert '小林' in history['messages'][0]['content']
    assert 'historical_version' in history['messages'][0]['content']
    memory.search_results = [{'id': 'old', 'memory': '[用户偏好] 我喜欢咖啡'}]
    exchange(engine, 'A', '我不喜欢咖啡', previous='好的')
    assert engine.search_long_term(user_id='A', agent_id='bot', query='推荐喝什么') == []
    old = engine.search_long_term(user_id='A', agent_id='bot', query='我以前喜欢什么')
    assert old[0]['fact_status'] == 'historical_superseded'


def test_timestamp_validation_is_offset_aware_and_rejects_orphan_fields():
    from bumi_mem.api import ContextRequest
    from pydantic import ValidationError
    for body in [dict(user_id='A', query='hi', query_occurred_at='2026-09-21T01:00:00'),
                 dict(user_id='A', query=None, query_occurred_at='2026-09-21T01:00:00+08:00'),
                 dict(user_id='A', query='hi', previous_reply_occurred_at='2026-09-21T01:00:00+08:00')]:
        with pytest.raises(ValidationError):
            ContextRequest(**body)
    assert ContextRequest(user_id='A', query='hi', query_occurred_at='2026-09-21T00:00:00Z').query_occurred_at.endswith('+00:00')


def test_explicit_recall_uses_separate_bounded_timeout(tmp_path, monkeypatch):
    engine, store, memory, _, _ = make_engine(tmp_path)
    timeouts = []
    monkeypatch.setattr(engine.long_term, 'search_with_timeout', lambda *, timeout, **kwargs: timeouts.append(timeout) or [])
    exchange(engine, 'A', '你还记得上次我们聊了什么吗')
    assert timeouts == [engine.config.explicit_recall_timeout]
    exchange(engine, 'A', '天气不错', previous='暂时想不起来')
    assert len(timeouts) == 1


def test_targeted_delete_removes_new_id_generated_from_withdrawn_sources(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    exchange(engine, 'A', 'source fact')
    source = store.list_shared_messages('bot')[0]
    metadata = {'source': 'l1_session', 'source_message_ids': [source.message_id]}
    old = {'id': 'old', 'memory': 'fact', 'user_id': 'A', 'agent_id': 'bot', 'metadata': metadata}
    memory.memories[('A', 'bot')] = [old]
    engine.forget_memory(user_id='A', agent_id='bot', memory_id='old')
    memory.memories[('A', 'bot')].append(dict(old, id='late-new-id'))
    assert engine.search_long_term(user_id='A', agent_id='bot', query='fact') == []
    assert engine.read_memory(user_id='A', agent_id='bot', memory_id='late-new-id') is None


def test_reference_payload_retains_event_time_and_never_uses_recall_date(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    memory.search_results = [{'id': 'dated', 'memory': '去了游乐场',
                              'metadata': {'observed_at': '2026-09-21T12:00:00+08:00', 'event_date': '2026-09-20'}}]
    exchange(engine, 'A', 'hi')
    guest = exchange(engine, GUEST_USER_ID, '继续', previous='收到')
    assert 'event_date=2026-09-20' in text(guest)
    assert 'source_at=2026-09-21T12:00:00+08:00' in text(guest)


def test_plan_state_does_not_become_completed_just_because_time_passed(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    result = exchange(engine, 'A', '确认计划：公园；状态：计划中', query_occurred_at='2020-01-01T09:00:00+08:00')
    assert result['core_write']['status'] == 'persisted'
    assert store.semantic_context('A', 'bot')[0]['status'] == 'planned'
    result = exchange(engine, 'A', '确认计划：公园；状态：已取消', previous='知道了')
    assert result['core_write']['status'] == 'persisted'
    assert store.semantic_context('A', 'bot')[0]['status'] == 'cancelled'
    assert len(store.semantic_history('A', 'bot')) == 2


@pytest.mark.parametrize('query,key,value', [('我在北京工作', 'work_location', '北京'),
                                           ('我现在住在上海', 'residence', '上海')])
def test_clear_personal_location_fact_has_correct_attribute(tmp_path, query, key, value):
    engine, store, memory, _, _ = make_engine(tmp_path)
    result = exchange(engine, 'A', query)
    assert result['core_write']['status'] == 'persisted'
    current = store.semantic_context('A', 'bot')[0]
    assert current['attribute'] == key
    assert current['value'] == value


def test_explicit_remember_normalizes_dates_and_preserves_original_evidence(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    receipt = engine.remember(user_id='A', agent_id='bot', contents=['昨天去游乐场'],
                              observed_at='2026-09-21T09:00:00+08:00')
    row = engine.read_memory(user_id='A', agent_id='bot', memory_id=receipt.records[0].memory_id)
    assert row['memory'] == '2026-09-20去游乐场'
    assert row['metadata']['original_contents'] == ['昨天去游乐场']
    assert row['metadata']['observed_at'] == '2026-09-21T09:00:00+08:00'


def test_fresh_recall_reanchors_only_after_old_anchor_is_token_clipped(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    memory.search_results = [{'id': 'm', 'memory': 'fact'}]
    exchange(engine, 'A', 'one')
    old_anchor = store.list_shared_messages('bot')[0].message_id
    # A context-only budget change clips the first message without rotating storage.
    engine.context_builder.max_l1_tokens = 5
    next_response = exchange(engine, 'A', 'two', previous='yes')
    selected = engine.context_builder.select_l1(store.list_shared_messages('bot'))
    assert old_anchor not in {m.message_id for m in selected}
    refs = store.shared_memory_refs('bot', selected_message_ids=[m.message_id for m in selected])
    assert len(refs) == 1
    assert refs[0]['anchor_message_id'] != old_anchor
    assert 'fact' in text(next_response)


def test_robot_promise_state_changes_only_after_explicit_confirmation(tmp_path):
    engine, store, memory, _, _ = make_engine(tmp_path)
    exchange(engine, 'A', 'hi')
    exchange(engine, 'A', '继续', previous='我答应你陪你练习')
    result = exchange(engine, 'A', '确认机器人承诺：陪你练习；状态：已完成', previous='好的')
    assert result['core_write']['status'] == 'persisted'
    assert store.semantic_context('A', 'bot')[0]['status'] == 'completed'
    history = store.semantic_history('A', 'bot')
    assert {r['source_role'] for r in history} == {'user', 'assistant'}
