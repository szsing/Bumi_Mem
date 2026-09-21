"""No-Pipecat/no-hardware bridge tests, including the real Agent's stdlib modules.

Set BUMI_AGENT_REPO to run the cross-repository tests in CI. Local Windows path
is a convenience; absent checkout causes only the real-source tests to skip.
"""
from __future__ import annotations

import ast
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import typing

import httpx
import pytest

from bumi_mem.agent_bridge import (client_turn_id, timeline_occurred_at, speaker_for_memory,
    prepare_payload, complete_payload_from_speech, local_control_payload, merge_agent_context,
    MEMORY_BLOCK_PREFIX)
from bumi_mem.client import BumiMemoryClient, AsyncBumiMemoryClient
from test_agent_api import harness, AUTH

AGENT_REPO = Path(os.environ.get('BUMI_AGENT_REPO', r'D:\代码\Bumi\_agent'))


def load_real(name, relative):
    path = AGENT_REPO / relative
    if not path.is_file():
        pytest.skip('set BUMI_AGENT_REPO to the actual Agent checkout')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def post(client, payload):
    response = client.post('/v1/memory/context', headers=AUTH, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_identity_shadow_mode_and_labels_do_not_enable_personal_access():
    identity = dict(status='matched', profile_id='profile-1', label='not-the-id', score=.99)
    assert speaker_for_memory(identity, enabled=False, mode='label') == {'status': 'disabled'}
    assert speaker_for_memory(identity, enabled=True, mode='shadow') == {'status': 'disabled'}
    assert speaker_for_memory(identity, enabled=True, mode='label') == {'status': 'matched', 'profile_id': 'profile-1'}
    for status in ['unknown', 'timeout', 'unavailable', 'insufficient_audio']:
        assert speaker_for_memory(dict(identity, status=status), enabled=True, mode='label') == {'status': status}


def test_ids_are_session_scoped_and_time_uses_wall_clock():
    assert client_turn_id('run-A', 'turn-000001') != client_turn_id('run-B', 'turn-000001')
    timeline = SimpleNamespace(speech_started_wall_time=0, speech_started_monotonic_ns=999999999)
    assert timeline_occurred_at(timeline, timezone_info=timezone.utc) == '1970-01-01T00:00:00+00:00'
    with pytest.raises(ValueError):
        prepare_payload(query=b'raw pcm', request_turn_id='run:1')


def test_merge_preserves_tools_controls_and_user_once_without_mutating_history():
    original = [{'role': 'system', 'content': 'Robot persona'}, {'role': 'user', 'content': 'question'},
                {'role': 'assistant', 'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'stop', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'interrupted'},
                {'role': 'system', 'content': '{"event":"local_control"}'}]
    before = deepcopy(original)
    response = dict(operation='prepare', has_prompt=True, user_id='A', client_turn_id='run:1',
                    memory_context=[{'role': 'system', 'content': 'confirmed memory'}],
                    messages=[{'role': 'user', 'content': 'must not duplicate history'}])
    view = merge_agent_context(original, response, expected_user_id='A', expected_client_turn_id='run:1')
    assert original == before
    assert view[0] == original[0] and view[2:] == original[1:]
    assert sum(m.get('role') == 'user' for m in view) == 1
    assert merge_agent_context(view, response, expected_user_id='A', expected_client_turn_id='run:1') == view
    with pytest.raises(ValueError):
        merge_agent_context(original, response, expected_user_id='B', expected_client_turn_id='run:1')
    with pytest.raises(ValueError):
        merge_agent_context(original, response, expected_user_id='A', expected_client_turn_id='run:2')


def test_sync_and_async_clients_send_identical_frozen_operations():
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={'ok': True})
    with BumiMemoryClient('http://memory', 'test', transport=httpx.MockTransport(handle)) as client:
        client.prepare_turn(client_turn_id='run:1', query='hi', speaker={'status': 'unknown'})
        client.complete_turn(turn_id='server-turn', status='interrupted', reply='prefix')
        client.agent_request({'operation': 'pending'})
    async def run():
        async with AsyncBumiMemoryClient('http://memory', 'test', transport=httpx.MockTransport(handle)) as client:
            await client.prepare_turn(client_turn_id='run:1', query='hi', speaker={'status': 'unknown'})
            await client.complete_turn(turn_id='server-turn', status='interrupted', reply='prefix')
            await client.agent_request({'operation': 'pending'})
    asyncio.run(run())
    assert captured[:3] == captured[3:]
    assert 'previous_ai_reply' not in captured[0]


def test_real_speaker_identity_maps_profile_and_not_biometrics(harness):
    client, _, _, backend, *_ = harness
    module = load_real('agent_speaker_types_contract', 'bumi_agent/speaker_identity/types.py')
    identity = module.SpeakerIdentity(status='matched', profile_id='stable-profile', label='小林', score=.95)
    payload = prepare_payload(query='hello', request_turn_id='run:identity', identity=identity,
                              identity_enabled=True, identity_mode='label', occurred_at='2026-09-21T10:00:00+08:00')
    result = post(client, payload)
    assert result['user_id'] == 'stable-profile'
    assert backend.search_calls[-1]['user_id'] == 'stable-profile'
    assert 'score' not in payload['speaker'] and 'label' not in payload['speaker']


def test_real_speech_progress_submits_committed_prefix_not_generated_tail(harness):
    client, engine, store, backend, *_ = harness
    module = load_real('agent_speech_contract', 'bumi_agent/speech_progress.py')
    turn = post(client, dict(operation='prepare', client_turn_id='run:speech', user_id='A', query='hi'))['turn_id']
    frozen = []
    speech = module.SpeechProgress(1000, history=[], on_complete=lambda: frozen.append(
        complete_payload_from_speech(turn_id=turn, speech=speech, status='interrupted')))
    segment = speech.begin_segment()
    speech.add_text(segment, 'Hello world')
    speech.add_words([dict(word='Hello', startTime=0, endTime=.4), dict(word='world', startTime=.4, endTime=.8)])
    speech.publish(lambda: None, 400)
    speech.stop_output()
    speech.finish_generation()
    assert len(frozen) == 1
    assert frozen[0]['reply'] == 'Hello'
    assert frozen[0]['status'] == 'interrupted'
    receipt = post(client, frozen[0])
    assert receipt['reply_saved']
    assert [m.content for m in store.list_shared_messages('bot')] == ['hi', 'Hello']
    engine.end_session(user_id='A', agent_id='bot')
    engine.worker.run_once()
    assert all('world' not in m['content'] for c in backend.add_calls for m in c['messages'])


def test_real_missing_subtitles_produce_unknown_tail_notice_not_fabricated_text(harness):
    client, _, store, *_ = harness
    module = load_real('agent_speech_missing_contract', 'bumi_agent/speech_progress.py')
    turn = post(client, dict(operation='prepare', client_turn_id='run:missing', user_id='A', query='hi'))['turn_id']
    frozen, history = [], []
    speech = module.SpeechProgress(1000, history=history, on_complete=lambda: frozen.append(
        complete_payload_from_speech(turn_id=turn, speech=speech, status='completed')))
    segment = speech.begin_segment()
    speech.add_text(segment, 'Do not claim this transcript is known')
    speech.publish(lambda: None, 1000)
    speech.finish_generation()
    speech.finish_alignment()
    speech.finish_output()
    assert speech.committed_text == '' and frozen[0]['speech']['alignment_covers_output'] is False
    assert any('speech_history_incomplete' in m.get('content', '') for m in history)
    result = post(client, frozen[0])
    assert result['status'] == 'completed' and not result['transcript_complete']
    assert 'Do not claim' not in json.dumps([m.content for m in store.list_shared_messages('bot')])
    assert any('speech_history_incomplete' in m.content for m in store.list_shared_messages('bot'))


def test_real_conversation_control_event_is_accepted_without_reexecution(harness):
    client, _, store, *_ = harness
    module = load_real('agent_conversation_contract', 'bumi_agent/conversation_state.py')
    history = []
    conversation = module.ConversationState(history.append)
    token = conversation.begin_control('stop', '停下来')
    conversation.finish_control(token, {'status': 'cancelled', 'started': False}, feedback_status='played')
    payload = local_control_payload(event_id='run:stop-1', history_message=history[0])
    result = post(client, payload)
    assert not result['has_prompt']
    stored = json.loads(store.list_shared_messages('bot')[0].content)
    assert stored['event'] == 'local_control'
    assert stored['result']['started'] is False and stored['llm_followup'] is False


def test_real_agent_message_builder_keeps_tool_chain_and_identity_labels(harness):
    path = AGENT_REPO / 'bumi_agent/agent.py'
    if not path.is_file():
        pytest.skip('actual Agent checkout not available')
    # Exercise its exact pure formatting methods without importing hardware/skills.
    tree = ast.parse(path.read_text(encoding='utf8'))
    original_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ConversationAgent')
    methods = [n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name in {'_messages_for_llm', '_trimmed_history'}]
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {'SYSTEM_PROMPT', 'SPEAKER_ID_SYSTEM_PROMPT'}}
    cls = ast.ClassDef(name='ActualMessageView', bases=[], keywords=[], body=methods, decorator_list=[])
    program = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = dict(Any=typing.Any, **constants)
    exec(compile(program, str(path), 'exec'), namespace)
    agent = namespace['ActualMessageView']()
    agent.settings = SimpleNamespace(speaker_id_enabled=True, speaker_id_mode='label')
    agent.history = [dict(role='user', content='question', speaker={'status': 'matched', 'label': '小林'}),
                     dict(role='assistant', tool_calls=[{'id': 'call-1', 'type': 'function', 'function': {'name': 'camera', 'arguments': '{}'}}]),
                     dict(role='tool', tool_call_id='call-1', content='real result'),
                     dict(role='user', content=[{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]),
                     dict(role='assistant', content='')]
    before = deepcopy(agent.history)
    client = harness[0]
    response = post(client, dict(operation='prepare', client_turn_id='run:format', user_id='A', query='question'))
    original = agent._messages_for_llm()
    view = merge_agent_context(original, response, expected_user_id='A', expected_client_turn_id='run:format')
    assert agent.history == before
    assert view[0] == original[0] and view[2:] == original[1:]
    assert '本地可信身份信息' in view[2]['content']
    assert any(m.get('tool_call_id') == 'call-1' for m in view)
    assert sum(m['role'] == 'system' and m.get('content', '').startswith(MEMORY_BLOCK_PREFIX) for m in view) == 1


def test_cold_start_restores_only_prior_spoken_history_not_current_query_or_tools(harness):
    from bumi_mem.agent_bridge import cold_start_spoken_history
    client = harness[0]
    first = post(client, dict(operation='prepare', client_turn_id='run:old', user_id='A', query='old question'))
    post(client, dict(operation='complete', turn_id=first['turn_id'], status='completed', reply='old answer'))
    new = post(client, dict(operation='prepare', client_turn_id='new-run:1', user_id='A', query='new question'))
    history = cold_start_spoken_history(new, existing_history=[], expected_user_id='A', expected_client_turn_id='new-run:1')
    assert [m['role'] for m in history] == ['user', 'assistant']
    assert 'new question' not in json.dumps(history)
    assert all('tool_calls' not in m for m in history)
    with pytest.raises(ValueError):
        cold_start_spoken_history(new, existing_history=[{'role':'tool','content':'real result'}],
                                  expected_user_id='A', expected_client_turn_id='new-run:1')


def test_timeline_uses_conversation_local_date_across_utc_midnight(harness):
    client, engine, store, backend, *_ = harness
    timestamp = datetime(2026, 9, 20, 16, 5, tzinfo=timezone.utc).timestamp()
    timeline = SimpleNamespace(speech_started_wall_time=timestamp)
    observed = timeline_occurred_at(timeline, timezone_info=timezone(timedelta(hours=8)))
    assert observed == '2026-09-21T00:05:00+08:00'
    prepared = post(client, dict(operation='prepare', client_turn_id='run:midnight', user_id='A',
                                 query='昨天去了一趟游乐场', occurred_at=observed))
    post(client, dict(operation='complete', turn_id=prepared['turn_id'], status='completed', reply='知道了'))
    engine.end_session(user_id='A', agent_id='bot')
    engine.worker.run_once()
    assert '2026-09-20' in backend.add_calls[0]['messages'][0]['content']


def test_speech_snapshot_does_not_guess_task_success():
    from bumi_mem.agent_bridge import settled_speech_snapshot
    speech = SimpleNamespace(settled=True, interrupted=False, committed_text='prefix', alignment_status='completed',
                             output_samples=1000, sample_rate=1000, alignment_covers_output=lambda: True)
    snapshot = settled_speech_snapshot(speech)
    assert 'status' not in snapshot
    failed = complete_payload_from_speech(turn_id='turn-known', speech=speech, status='failed')
    assert failed['status'] == 'failed' and failed['reply'] == 'prefix'
