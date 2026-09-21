"""HTTP contract tests for the actual custom bumi-agent lifecycle, no external APIs."""
from __future__ import annotations

import json
import pytest

from bumi_mem.agent_protocol import AgentPrepareRequest, AgentCompleteRequest
from bumi_mem.turn_store import TurnConflict, TurnNotFound
from test_agent_api import harness, AUTH
from test_bumi_memory import make_engine


def post(client, **payload):
    return client.post('/v1/memory/context', headers=AUTH, json=payload)


def prepare(client, cid='run:turn-1', user='A', query='你好', **extra):
    payload = dict(operation='prepare', client_turn_id=cid, query=query)
    if user is not None:
        payload['user_id'] = user
    return post(client, **payload, **extra)


def complete(client, turn, status='completed', reply='好的', **extra):
    return post(client, operation='complete', turn_id=turn, status=status, reply=reply, **extra)


def dds(*, covers=True, samples=16000):
    return dict(delivery_basis='dds_successful_publish', alignment_covers_output=covers,
                alignment_status='completed' if covers else 'missing_subtitles', output_samples=samples, sample_rate=16000)


def test_prepare_returns_only_one_memory_supplement_without_answering(harness):
    client, engine, store, memory, chat, summary, *_ = harness
    response = prepare(client, query='以后叫我小林')
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['operation'] == 'prepare' and body['has_prompt']
    assert body['turn_id'].startswith('turn_') and body['client_turn_id'] == 'run:turn-1'
    assert body['core_write']['status'] == 'persisted'
    assert body['memory_context'][0]['content'].startswith(body['messages'][0]['content'])
    assert body['source_message_id'] == store.list_shared_messages('bot')[0].message_id
    assert body['occurred_at'] == store.list_shared_messages('bot')[0].created_at
    assert body['memory_context'][0]['role'] == 'system'
    assert len(store.list_shared_messages('bot')) == 1
    assert not chat.calls and not summary.calls


def test_prepare_retry_is_identical_and_changed_query_time_or_owner_conflicts(harness):
    client, _, store, memory, *_ = harness
    first = prepare(client, occurred_at='2026-09-21T12:00:00+08:00')
    assert prepare(client, occurred_at='2026-09-21T12:00:00+08:00').json() == first.json()
    assert prepare(client, query='different', occurred_at='2026-09-21T12:00:00+08:00').status_code == 409
    assert prepare(client, occurred_at='2026-09-21T12:00:01+08:00').status_code == 409
    assert prepare(client, user='B', occurred_at='2026-09-21T12:00:00+08:00').status_code == 409
    assert len(store.list_shared_messages('bot')) == 1 and len(memory.search_calls) == 1


def test_new_question_cannot_cancel_old_turn_or_receive_its_late_reply(harness):
    client, _, store, *_ = harness
    a = prepare(client).json()['turn_id']
    assert prepare(client, 'run:turn-2', 'B', 'B question').status_code == 409
    assert store.get_turn(a, 'bot')['status'] == 'awaiting_reply'
    interrupted = complete(client, a, status='interrupted', reply='A prefix', speech=dds())
    assert interrupted.status_code == 200
    b = prepare(client, 'run:turn-2', 'B', 'B question').json()['turn_id']
    # A frozen duplicate is acknowledged; a changed late reply cannot move to B.
    assert complete(client, a, status='interrupted', reply='A prefix', speech=dds()).json() == interrupted.json()
    assert complete(client, a, reply='A late full answer', speech=dds()).status_code == 409
    assert store.get_turn(b, 'bot')['status'] == 'awaiting_reply'
    with store._connect() as c:
        rows = c.execute("SELECT m.content,s.user_id FROM messages m JOIN sessions s USING(session_id) WHERE m.role='assistant'").fetchall()
    assert [(r['content'], r['user_id']) for r in rows] == [('A prefix', 'A')]


def test_empty_completed_reply_with_missing_alignment_is_not_silence(harness):
    client, engine, store, memory, *_ = harness
    turn = prepare(client).json()['turn_id']
    result = complete(client, turn, reply='', speech=dds(covers=False))
    assert result.status_code == 200, result.text
    assert result.json()['status'] == 'completed' and not result.json()['transcript_complete']
    assert not result.json()['reply_saved'] and not result.json()['has_prompt']
    messages = store.list_shared_messages('bot')
    assert [m.role for m in messages] == ['user', 'system']
    assert 'speech_history_incomplete' in messages[1].content
    assert messages[1].context_only
    next_turn = prepare(client, 'run:turn-2')
    assert 'speech_history_incomplete' in json.dumps(next_turn.json()['messages'])
    complete(client, next_turn.json()['turn_id'])
    engine.end_session(user_id='A', agent_id='bot')
    while engine.worker.run_once():
        pass
    assert all('speech_history_incomplete' not in m['content'] for call in memory.add_calls for m in call['messages'])


def test_interrupted_prefix_is_not_new_long_term_evidence_or_promise(harness):
    client, engine, store, memory, *_ = harness
    turn = prepare(client).json()['turn_id']
    result = complete(client, turn, status='interrupted', reply='我答应你明天', speech=dds())
    assert result.status_code == 200
    assert store.semantic_context('A', 'bot') == []
    engine.end_session(user_id='A', agent_id='bot')
    engine.worker.run_once()
    assert all('我答应你明天' not in m['content'] for c in memory.add_calls for m in c['messages'])


@pytest.mark.parametrize('status', ['unknown', 'timeout', 'unavailable', 'insufficient_audio', 'disabled'])
def test_unknown_identity_status_uses_guest_and_never_direct_l3(harness, status):
    client, _, store, memory, *_ = harness
    result = prepare(client, user=None, speaker={'status': status, 'profile_id': 'untrusted-candidate', 'score': 0.99})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body['user_id'] == '__guest__' and body['recall_status'] == 'disabled'
    assert 'unconfirmed' in body['messages'][-1]['content']
    assert not memory.search_calls
    assert store.list_shared_messages('bot')[-1].user_id == '__guest__'


def test_matched_identity_uses_profile_id_not_display_label(harness):
    client, _, _, memory, *_ = harness
    result = prepare(client, user=None, speaker={'status': 'matched', 'profile_id': 'profile-001', 'label': '小林', 'score': 0.99})
    assert result.status_code == 200, result.text
    assert result.json()['user_id'] == 'profile-001'
    assert memory.search_calls[-1]['user_id'] == 'profile-001'


@pytest.mark.parametrize('payload', [
    dict(operation='prepare', client_turn_id='x', query='hi'),
    dict(operation='prepare', client_turn_id='x', query='hi', user_id='A', speaker={'status': 'unknown'}),
    dict(operation='prepare', client_turn_id='x', query='hi', speaker={'status': 'matched'}),
    dict(operation='prepare', client_turn_id='x', query='hi', user_id='A', occurred_at='2026-09-21T12:00:00'),
    dict(operation='complete', turn_id='x', status='cancelled', reply='not cancelled'),
    dict(operation='complete', turn_id='x', status='completed', reply='words', speech=dds(samples=0)),
    dict(operation='prepare', client_turn_id='x', query='hi', user_id='A', previous_ai_reply='legacy cannot mix'),
])
def test_invalid_protocol_requests_are_rejected_without_mutation(harness, payload):
    client, _, store, *_ = harness
    assert post(client, **payload).status_code == 422
    assert store.list_shared_messages('bot') == []


def test_independent_welcome_speech_before_any_user_is_persistent_and_idempotent(harness):
    client, _, store, memory, *_ = harness
    payload = dict(operation='agent_event', event_id='run:welcome', kind='speech', text='你好，我是 Bumi。', speech=dds())
    first = post(client, **payload)
    assert first.status_code == 200, first.text
    assert post(client, **payload).json() == first.json()
    assert post(client, **dict(payload, text='changed')).status_code == 409
    assert store.pending_agent_turn('bot') is None
    assert not memory.search_calls
    following = prepare(client).json()
    assert '你好，我是 Bumi。' in json.dumps(following['messages'], ensure_ascii=False)
    assert store.list_shared_messages('bot')[0].user_id == ''


def test_control_event_does_not_close_turn_execute_action_or_trigger_answer(harness):
    client, _, store, memory, chat, *_ = harness
    turn = prepare(client).json()['turn_id']
    payload = dict(operation='agent_event', event_id='run:stop', kind='local_control', control={
        'command': 'stop', 'user_text': '停一下', 'reason': 'user_requested_stop',
        'result': {'status': 'interrupted', 'started': True}, 'feedback_status': 'played', 'llm_followup': False})
    result = post(client, **payload)
    assert result.status_code == 200, result.text
    assert not result.json()['has_prompt'] and not chat.calls
    assert store.get_turn(turn, 'bot')['status'] == 'awaiting_reply'
    assert 'local_control' in store.list_shared_messages('bot')[-1].content
    assert store.list_shared_messages('bot')[-1].context_only


def test_pending_is_metadata_only_and_can_close_recovered_unknown_output(harness):
    client, _, store, *_ = harness
    assert post(client, operation='pending').json()['pending'] is None
    turn = prepare(client).json()['turn_id']
    pending = post(client, operation='pending').json()
    assert pending['pending']['turn_id'] == turn
    assert 'messages' not in pending and 'query' not in pending['pending']
    result = complete(client, turn, status='failed', reply='', speech={
        'delivery_basis': 'unknown', 'alignment_covers_output': False, 'alignment_status': 'restart_unknown'})
    assert result.status_code == 200, result.text
    assert post(client, operation='pending').json()['pending'] is None
    assert 'restart_unknown' in store.list_shared_messages('bot')[-1].content


def test_agent_turns_do_not_implicitly_expire_or_mix_with_legacy(harness):
    client, engine, store, *_ = harness
    first = prepare(client).json()
    with store._connect(write=True) as c:
        c.execute('UPDATE interaction_turns SET expires_at=0')
    assert prepare(client).json() == first
    assert post(client, user_id='B', query='legacy', previous_ai_reply=None).status_code == 409
    with pytest.raises(TurnConflict, match='agent_busy'):
        engine.end_session(user_id='A', agent_id='bot')
    assert complete(client, first['turn_id']).status_code == 200
    assert prepare(client).status_code == 409  # completed input is not regenerated


def test_wrong_agent_cannot_complete_and_changed_speech_evidence_cannot_replay(harness):
    client, engine, store, *_ = harness
    turn = prepare(client).json()['turn_id']
    with pytest.raises(TurnNotFound):
        engine.complete_agent(request=AgentCompleteRequest(operation='complete', turn_id=turn, status='completed', reply='x'), agent_id='other')
    result = complete(client, turn, speech=dds())
    assert result.status_code == 200
    assert complete(client, turn, speech=dds(covers=False)).status_code == 409


def test_restart_recovers_pending_and_completed_receipt(tmp_path):
    engine, store, backend, *_ = make_engine(tmp_path)
    request = AgentPrepareRequest(operation='prepare', client_turn_id='run:turn-001', user_id='A', query='hello')
    prepared = engine.prepare_agent(request=request, agent_id='bot')
    engine.close()
    restarted, _, _, *_ = make_engine(tmp_path, long_term=backend)
    assert restarted.prepare_agent(request=request, agent_id='bot') == prepared
    completion = AgentCompleteRequest(operation='complete', turn_id=prepared['turn_id'], status='interrupted', reply='prefix')
    receipt = restarted.complete_agent(request=completion, agent_id='bot')
    restarted.close()
    again, store3, _, *_ = make_engine(tmp_path, long_term=backend)
    assert again.complete_agent(request=completion, agent_id='bot') == receipt
    assert [m.content for m in store3.list_shared_messages('bot')] == ['hello', 'prefix']


def test_snapshot_after_cross_user_revocation_excludes_delayed_output(harness):
    client, engine, store, *_ = harness
    engine.remember(user_id='A', agent_id='bot', contents=['old A fact'])
    a = prepare(client).json()['turn_id']
    complete(client, a)
    b = prepare(client, 'run:2', 'B').json()['turn_id']
    engine.forget_all(user_id='A', agent_id='bot')
    result = complete(client, b, reply='late old A fact', speech=dds())
    assert result.status_code == 200
    with store._connect() as c:
        row = c.execute("SELECT memory_excluded FROM messages WHERE content='late old A fact'").fetchone()
        assert row[0] == 1


def test_prepare_failure_after_input_commit_retries_without_duplicate(harness, monkeypatch):
    client, engine, store, *_ = harness
    original = engine._prompt_for_turn
    def fail(*args, **kwargs):
        raise RuntimeError('simulated response build failure')
    monkeypatch.setattr(engine, '_prompt_for_turn', fail)
    assert prepare(client).status_code == 503
    assert len(store.list_shared_messages('bot')) == 1
    monkeypatch.setattr(engine, '_prompt_for_turn', original)
    assert prepare(client).status_code == 200
    assert len(store.list_shared_messages('bot')) == 1


def test_completion_with_notice_rolls_back_atomically_if_rotation_fails(harness, monkeypatch):
    client, engine, store, *_ = harness
    engine.config.max_messages = 2
    turn = prepare(client).json()['turn_id']
    original = store._submit_shared
    monkeypatch.setattr(store, '_submit_shared', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('rotation failed')))
    assert complete(client, turn, reply='known prefix', speech=dds(covers=False)).status_code == 503
    assert [m.role for m in store.list_shared_messages('bot')] == ['user']
    assert store.get_turn(turn, 'bot')['status'] == 'awaiting_reply'
    monkeypatch.setattr(store, '_submit_shared', original)
    response = complete(client, turn, reply='known prefix', speech=dds(covers=False))
    assert response.status_code == 200
    assert [m.role for m in store.list_shared_messages('bot')] == ['user', 'assistant', 'system']


def test_control_event_revokes_cached_prompt_but_duplicate_event_is_noop(harness):
    client, _, store, *_ = harness
    turn = prepare(client).json()['turn_id']
    event = dict(operation='agent_event', event_id='run:stop', kind='local_control', control={
        'command': 'stop', 'user_text': 'stop', 'reason': 'user_requested_stop',
        'result': {'status': 'interrupted'}, 'feedback_status': 'played'})
    first = post(client, **event).json()
    epoch = store.context_epoch('bot')
    assert post(client, **event).json() == first
    assert store.context_epoch('bot') == epoch
    result = prepare(client)
    assert result.status_code == 409 and result.json()['detail']['code'] == 'shared_memory_context_changed'
    assert complete(client, turn, status='interrupted', reply='confirmed prefix', speech=dds()).status_code == 200


def test_v6_database_upgrade_preserves_legacy_pending_turn(tmp_path):
    import sqlite3
    engine, store, backend, *_ = make_engine(tmp_path)
    old = engine.process_context(user_id='A', agent_id='bot', query='legacy', previous_ai_reply=None, idempotency_key='old')
    engine.close()
    with sqlite3.connect(store.path) as c:
        c.execute('DROP TABLE agent_turn_requests')
        c.execute('DROP TABLE agent_event_receipts')
        c.execute('PRAGMA user_version=6')
    new, upgraded, _, *_ = make_engine(tmp_path, long_term=backend)
    assert new.process_context(user_id='A', agent_id='bot', query='legacy', previous_ai_reply=None, idempotency_key='old') == old
    new.process_context(user_id='A', agent_id='bot', query=None, previous_ai_reply='legacy answer', idempotency_key='old-reply')
    current = new.prepare_agent(request=AgentPrepareRequest(operation='prepare', client_turn_id='run:new', user_id='A', query='new'), agent_id='bot')
    assert current['operation'] == 'prepare'
    assert [m.content for m in upgraded.list_shared_messages('bot')] == ['legacy', 'legacy answer', 'new']


def test_independent_incomplete_speech_records_prefix_and_single_notice(harness):
    client, _, store, *_ = harness
    body = dict(operation='agent_event', event_id='run:incomplete', kind='speech', text='known prefix', speech=dds(covers=False))
    result = post(client, **body)
    assert result.status_code == 200, result.text
    assert [m.role for m in store.list_shared_messages('bot')] == ['assistant', 'system']
    assert post(client, **body).json() == result.json()
    empty = dict(operation='agent_event', event_id='run:empty', kind='speech', text='   ', speech=dds(covers=False))
    assert post(client, **empty).status_code == 200
    assert [m.role for m in store.list_shared_messages('bot')] == ['assistant', 'system', 'system']


def test_concurrent_duplicate_prepare_persists_exactly_one_user_message(harness):
    from concurrent.futures import ThreadPoolExecutor
    client, _, store, backend, *_ = harness
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: prepare(client).json(), range(4)))
    assert all(response == responses[0] for response in responses)
    assert len(store.list_shared_messages('bot')) == 1
    assert len(backend.search_calls) == 1
