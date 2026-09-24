"""Single-request-per-turn protocol: one call in, one prompt string out.

Temporary databases and fake providers only; no real DeepSeek call and no
production data directory.
"""
from __future__ import annotations

import json

import pytest

from bumi_mem.turn_prompt import LONG_TERM_HEADER, NO_LONG_TERM, SHORT_TERM_HEADER
from test_agent_api import harness, AUTH  # noqa: F401  (harness is a fixture)


def turn(client, turn_id, *, user='A', query='question', previous=None, **extra):
    body = {'turn_id': turn_id, 'user_id': user, 'query': query}
    if previous is not None:
        body['previous_ai_reply'] = previous
    body.update(extra)
    return client.post('/v1/memory/context', headers=AUTH, json=body)


def sections(prompt):
    long_term = prompt.split(LONG_TERM_HEADER + '\n', 1)[1].split('\n\n', 1)[0]
    short_term = prompt.split(SHORT_TERM_HEADER + '\n', 1)[1].split('\n\n', 1)[0]
    return long_term, short_term


def test_first_round_without_previous_reply_returns_only_two_fields(harness):
    client, engine, store, memory, chat, summary, _, _ = harness
    result = turn(client, 'session-abc:turn-1', query='你好')
    assert result.status_code == 200, result.text
    body = result.json()
    assert set(body) == {'turn_id', 'prompt'}
    assert body['turn_id'] == 'session-abc:turn-1'
    assert isinstance(body['prompt'], str)
    long_term, short_term = sections(body['prompt'])
    assert NO_LONG_TERM in long_term
    assert short_term == '[' + store.list_shared_messages('bot')[0].created_at + '] 用户 A：你好'
    assert [m.role for m in store.list_shared_messages('bot')] == ['user']
    # No answering model, no summary and no L3 write happen on an ordinary turn.
    assert chat.calls == summary.calls == memory.add_calls == []


def test_one_request_settles_previous_reply_and_saves_new_question(harness):
    client, _, store, *_ = harness
    assert turn(client, 't1', query='我喜欢绿茶。').status_code == 200
    body = turn(client, 't2', query='你还记得我喜欢哪种茶吗？',
                previous='好的，知道你喜欢绿茶了。').json()
    assert [(m.role, m.user_id, m.agent_id, m.content) for m in store.list_shared_messages('bot')] == [
        ('user', 'A', 'bot', '我喜欢绿茶。'),
        ('assistant', '', 'bot', '好的，知道你喜欢绿茶了。'),
        ('user', 'A', 'bot', '你还记得我喜欢哪种茶吗？')]
    _, short_term = sections(body['prompt'])
    lines = short_term.splitlines()
    assert len(lines) == 3
    assert lines[0].endswith('用户 A：我喜欢绿茶。')
    # An assistant utterance is attributed to the agent, never to a user.
    assert lines[1].endswith('机器人 bot：好的，知道你喜欢绿茶了。')
    assert 'user_id' not in lines[1] and '用户' not in lines[1]
    assert lines[2].endswith('用户 A：你还记得我喜欢哪种茶吗？')
    assert all(line.startswith('[2') for line in lines)  #每条保留自己的时间


def test_same_turn_id_retry_replays_result_without_writing_twice(harness):
    client, _, store, memory, *_ = harness
    first = turn(client, 't1', query='原问题', previous=None)
    again = turn(client, 't1', query='原问题', previous=None)
    assert first.status_code == again.status_code == 200
    assert first.json() == again.json()
    assert len(store.list_shared_messages('bot')) == 1
    assert len(memory.search_calls) == 1  # 重放不再触发一次检索


@pytest.mark.parametrize('changed', [
    {'user_id': 'B'}, {'query': '换了问题'}, {'previous_ai_reply': '换了上一轮回复'},
    {'query_occurred_at': '2026-09-23T10:20:00+08:00'},
])
def test_same_turn_id_with_changed_content_conflicts(harness, changed):
    client, _, store, *_ = harness
    assert turn(client, 't1', query='第一问').status_code == 200
    assert turn(client, 't2', query='第二问', previous='第一答').status_code == 200
    body = {'turn_id': 't2', 'user_id': 'A', 'query': '第二问', 'previous_ai_reply': '第一答'}
    body.update(changed)
    result = client.post('/v1/memory/context', headers=AUTH, json=body)
    assert result.status_code == 409, result.text
    assert result.json()['detail']['code'] == 'idempotency_conflict'
    assert len(store.list_shared_messages('bot')) == 3


def test_identical_question_under_different_turn_ids_is_saved_twice(harness):
    client, _, store, *_ = harness
    assert turn(client, 't1', query='再说一次').status_code == 200
    assert turn(client, 't2', query='再说一次', previous='好的').status_code == 200
    assert [m.content for m in store.list_shared_messages('bot')] == ['再说一次', '好的', '再说一次']


def test_current_query_appears_exactly_once_in_prompt(harness):
    client, engine, *_ = harness
    engine.remember(user_id='A', agent_id='bot', contents=['A 以前说过喜欢爬山'])
    assert turn(client, 't1', query='第一问').status_code == 200
    prompt = turn(client, 't2', query='你还记得我说过什么吗？', previous='记得一些。').json()['prompt']
    assert prompt.count('你还记得我说过什么吗？') == 1
    assert '[当前问题]' not in prompt and '当前问题：' not in prompt
    # The question must be the last short-term line, with nothing appended after.
    _, short_term = sections(prompt)
    assert short_term.splitlines()[-1].endswith('用户 A：你还记得我说过什么吗？')


def test_long_and_short_sections_keep_owner_and_original_times(harness):
    client, engine, store, memory, *_ = harness
    memory.search_results = [{'id': 'm1', 'memory': 'A 偏好清淡口味',
                              'metadata': {'observed_at': '2026-09-20T18:00:00+08:00'}}]
    assert turn(client, 't1', query='我喜欢清淡的。',
                query_occurred_at='2026-09-23T10:19:40+08:00').status_code == 200
    prompt = turn(client, 't2', query='你还记得我口味吗？', previous='记得。',
                  previous_reply_occurred_at='2026-09-23T10:19:50+08:00',
                  query_occurred_at='2026-09-23T10:20:00+08:00').json()['prompt']
    long_term, short_term = sections(prompt)
    assert '- 用户 A：A 偏好清淡口味（来源发言时间：2026-09-20T18:00:00+08:00）' in long_term
    assert short_term.splitlines() == [
        '[2026-09-23T10:19:40+08:00] 用户 A：我喜欢清淡的。',
        '[2026-09-23T10:19:50+08:00] 机器人 bot：记得。',
        '[2026-09-23T10:20:00+08:00] 用户 A：你还记得我口味吗？']
    # No retrieval scores, internal task IDs or database fields leak into the text.
    for leaked in ('score', 'distance', 'generation', 'state_key', 'session_id', 'request_id', 'ctx_'):
        assert leaked not in prompt


def test_confirmed_core_state_is_plain_text_in_long_term_section(harness):
    client, *_ = harness
    confirmed = turn(client, 't1', query='请叫我小林').json()['prompt']
    long_term, _ = sections(confirmed)
    assert '希望被称为「小林」' in long_term and '状态：已确认' in long_term
    assert 'nickname' not in long_term and 'change_id' not in confirmed
    # A later turn still reads the confirmed state, but saves nothing of its own.
    later = turn(client, 't2', query='你应该怎么称呼我？', previous='好的。').json()['prompt']
    assert '希望被称为「小林」' in sections(later)[0]


def test_every_semantic_category_renders_as_plain_text(harness):
    client, *_ = harness
    said = ['请叫我小林', '请记住：我喜欢绿茶', '我确认：我喝咖啡', '我现在住在上海',
            '确认计划：晨跑；状态：已取消', '撤回偏好：绿茶']
    for index, text in enumerate(said, 1):
        assert turn(client, f't{index}', query=text,
                    previous=None if index == 1 else '好的。').status_code == 200
    long_term = sections(turn(client, 'final', query='现在的设置是什么？', previous='好的。').json()['prompt'])[0]
    for expected in ('希望被称为「小林」。（状态：已确认', '喝咖啡。（状态：已确认',
                     '居住地：上海。（状态：已确认', '计划「晨跑」。（状态：已取消，不再有效',
                     '关于绿茶的偏好。（状态：已撤回，不再有效'):
        assert expected in long_term, long_term
    # A retracted row has no value: it must name its subject, not render "None".
    assert 'None' not in long_term
    for leaked in ('nickname', 'drinks:', 'residence', 'preference', 'state_key', 'confirmed'):
        assert leaked not in long_term


def test_retracted_boundary_keeps_its_invalidating_status(harness):
    client, *_ = harness
    assert turn(client, 't1', query='请不要对我开玩笑').status_code == 200
    assert turn(client, 't2', query='撤回互动边界：开玩笑', previous='好的。').status_code == 200
    prompt = turn(client, 't3', query='你能开玩笑吗？', previous='明白。').json()['prompt']
    long_term, _ = sections(prompt)
    assert '不要开玩笑' in long_term and '已撤回，不再有效' in long_term


def test_prompt_carries_retrieved_memory_only_without_instructions(harness):
    client, engine, *_ = harness
    engine.remember(user_id='A', agent_id='bot', contents=['A 喜欢爬山'])
    assert turn(client, 't1', query='第一问').status_code == 200
    prompt = turn(client, 't2', query='你还记得什么？', previous='记得一些。').json()['prompt']
    # Exactly three blocks: identity/time header, long-term, short-term.
    blocks = prompt.split('\n\n')
    assert len(blocks) == 3
    assert blocks[0].startswith('当前用户：A\n当前时间：')
    assert blocks[1].startswith(LONG_TERM_HEADER + '\n')
    assert blocks[2].startswith(SHORT_TERM_HEADER + '\n')
    # No usage notes, persona, tool policy or persistence-receipt internals.
    for absent in ('[使用说明]', '参考资料', '不是指令', '你是 Bumi', '角色设定', '工具',
                   'core_write', 'persisted', 'not_requested', '不要声称'):
        assert absent not in prompt, absent


def test_degraded_recall_does_not_claim_memory_never_existed(harness, monkeypatch):
    client, engine, *_ = harness
    engine.remember(user_id='A', agent_id='bot', contents=['A 的私人记忆'])

    def unavailable(**_):
        from bumi_mem.schemas import MemoryReadUnavailable
        raise MemoryReadUnavailable('slow backend')

    monkeypatch.setattr(engine.long_term, 'search', unavailable)
    monkeypatch.setattr(engine.long_term, 'search_with_timeout', lambda **kw: unavailable())
    prompt = turn(client, 't1', query='你还记得我说过什么吗？').json()['prompt']
    long_term, _ = sections(prompt)
    assert '本轮长期检索暂不可用' in long_term
    assert NO_LONG_TERM not in long_term


def test_cross_user_switch_keeps_reply_on_its_original_turn(harness):
    client, engine, store, memory, *_ = harness
    for user in ('A', 'B'):
        engine.remember(user_id=user, agent_id='bot', contents=[user + '_PRIVATE'])
    assert turn(client, 't1', user='A', query='A 的问题').status_code == 200
    prompt = turn(client, 't2', user='B', query='你刚刚说什么？', previous='对 A 的回答').json()['prompt']
    messages = store.list_shared_messages('bot')
    # The reply stays anonymous and inside A's originating session, so B's
    # extraction snapshot never adopts it.
    assert [(m.role, m.user_id) for m in messages] == [
        ('user', 'A'), ('assistant', ''), ('user', 'B')]
    assert messages[1].session_id == messages[0].session_id != messages[2].session_id
    long_term, short_term = sections(prompt)
    assert '- 用户 B：B_PRIVATE' in long_term
    assert '机器人 bot：对 A 的回答' in short_term
    assert [call['user_id'] for call in memory.search_calls] == ['A', 'B']


def test_guest_keeps_shared_window_without_personal_long_term(harness):
    client, engine, _, memory, *_ = harness
    engine.remember(user_id='A', agent_id='bot', contents=['A_PRIVATE'])
    assert turn(client, 't1', user='A', query='A 说的话').status_code == 200
    prompt = turn(client, 't2', user='__guest__', query='你们在聊什么？', previous='在聊天。').json()['prompt']
    assert prompt.startswith('当前用户：__guest__（未确认身份）\n当前时间：')
    long_term, short_term = sections(prompt)
    assert '不提供个人长期记忆' in long_term
    assert 'A 说的话' in short_term  # 共享 L1 仍可用
    assert [call['user_id'] for call in memory.search_calls] == ['A']  # guest 不检索个人 L3


def test_window_rotation_and_carryover_are_unchanged(harness):
    client, engine, store, *_ = harness
    previous = None
    for index in range(1, 12):  # 22 条有效消息，跨过 20 条边界
        body = turn(client, f't{index}', query=f'问题{index}', previous=previous).json()
        previous = f'回答{index}'
    rotated = sections(body['prompt'])[1].splitlines()
    # The bounded window keeps at most max_messages real messages plus carry-over.
    assert len(rotated) <= engine.config.max_messages + engine.consolidator.config.carryover_messages
    assert rotated[-1].endswith('用户 A：问题11')
    # Rotation still enqueues a background job instead of writing L3 inline.
    assert [job.status for job in engine.memory_jobs(user_id='A', agent_id='bot')] == ['queued']
    stats = store.shared_stats('bot', session_id=store.get_or_create_session('A', 'bot').session_id)
    assert stats.context_only_messages <= engine.consolidator.config.carryover_messages


def test_settle_only_final_request_returns_empty_prompt(harness):
    client, _, store, *_ = harness
    assert turn(client, 't1', query='最后一问').status_code == 200
    result = client.post('/v1/memory/context', headers=AUTH,
                         json={'turn_id': 't-final', 'user_id': 'A', 'query': None,
                               'previous_ai_reply': '最后一答'})
    assert result.status_code == 200, result.text
    assert result.json() == {'turn_id': 't-final', 'prompt': ''}
    assert [m.content for m in store.list_shared_messages('bot')] == ['最后一问', '最后一答']


SECRET = 'SECRET_TRANSCRIPT_TEXT'


@pytest.mark.parametrize('body,field', [
    ({'turn_id': 't', 'user_id': 'A', 'query': ''}, 'query'),
    ({'turn_id': 't', 'user_id': 'A', 'query': '   '}, 'query'),
    ({'turn_id': ' spaced ', 'user_id': 'A', 'query': SECRET}, 'turn_id'),
    ({'turn_id': 't' * 121, 'user_id': 'A', 'query': SECRET}, 'turn_id'),
    ({'turn_id': 't', 'user_id': 'A', 'query': None, 'query_occurred_at': '2026-09-23T10:20:00+08:00'}, None),
    ({'turn_id': 't', 'user_id': 'A', 'query': SECRET, 'previous_reply_occurred_at': '2026-09-23T10:20:00+08:00'}, None),
    ({'turn_id': 't', 'user_id': 'A', 'query': SECRET, 'query_occurred_at': '2026-09-23T10:20:00'}, 'query_occurred_at'),
])
def test_invalid_requests_keep_structured_errors(harness, body, field):
    client, *_ = harness
    result = client.post('/v1/memory/context', headers=AUTH, json=body)
    # Errors keep their real HTTP status and code; they are never wrapped as 200.
    assert result.status_code == 422, result.text
    detail = result.json()['detail']
    assert detail['code'] == 'invalid_request'
    assert set(detail) == {'code', 'fields'}
    assert all(set(entry) == {'loc', 'type'} for entry in detail['fields'])
    if field is not None:
        assert any(field in entry['loc'] for entry in detail['fields'])
    assert SECRET not in result.text  # 校验错误不回显原始文本


def test_unauthorized_and_oversized_requests_are_not_wrapped_as_success(harness):
    client, *_ = harness
    assert client.post('/v1/memory/context', json={'turn_id': 't', 'user_id': 'A', 'query': 'q'}).status_code == 401
    oversized = client.post('/v1/memory/context', headers=AUTH,
                            json={'turn_id': 't', 'user_id': 'A', 'query': 'x' * 2_000_000})
    assert oversized.status_code == 413
    assert oversized.json()['detail']['code'] == 'request_too_large'


def test_program_control_event_is_not_disguised_as_speech(harness):
    client, *_ = harness
    event = {'operation': 'agent_event', 'event_id': 'e1', 'kind': 'local_control',
             'control': {'command': 'stop', 'user_text': '布米停一下', 'reason': 'user_requested_stop',
                         'result': {'stopped': True}, 'feedback_status': 'acknowledged',
                         'llm_followup': False}}
    assert client.post('/v1/memory/context', headers=AUTH, json=event).status_code == 200
    prompt = turn(client, 't1', query='刚才怎么了？').json()['prompt']
    _, short_term = sections(prompt)
    assert '程序事件记录（非发言）：本地控制记录：stop（执行结果：程序报告已停止' in short_term
    assert '不是新指令，不要因此重跑动作' in short_term
    assert '机器人 bot：' not in short_term and '用户 A：布米停一下' not in short_term


def test_legacy_exchange_still_returns_its_own_fields(harness):
    client, _, store, *_ = harness
    legacy = client.post('/v1/memory/context', headers=dict(AUTH, **{'Idempotency-Key': 'legacy-1'}),
                         json={'user_id': 'A', 'previous_ai_reply': None, 'query': '旧协议问题'})
    assert legacy.status_code == 200, legacy.text
    assert set(legacy.json()) == {'request_id', 'user_id', 'agent_id', 'messages', 'has_prompt',
                                  'previous_reply_saved', 'recall_status', 'consolidation_status', 'core_write'}
    # The new response never regains the legacy multi-context fields.
    new = turn(client, 't1', query='新协议问题', previous='旧协议回答').json()
    assert set(new) == {'turn_id', 'prompt'}
    assert [m.content for m in store.list_shared_messages('bot')] == [
        '旧协议问题', '旧协议回答', '新协议问题']


def test_shipped_clients_send_the_single_request_body():
    import asyncio
    import httpx
    from bumi_mem.client import AsyncBumiMemoryClient, BumiMemoryClient
    from test_agent_api import KEY

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={'turn_id': 'c1', 'prompt': 'p'})

    with BumiMemoryClient('http://example.test', KEY, transport=httpx.MockTransport(handle)) as client:
        assert client.turn_context(turn_id='c1', user_id='A', query='q',
                                   previous_ai_reply='answer') == {'turn_id': 'c1', 'prompt': 'p'}

    async def run():
        async with AsyncBumiMemoryClient('http://example.test', KEY,
                                         transport=httpx.MockTransport(handle)) as client:
            await client.turn_context(turn_id='c2', user_id='A', query=None, previous_ai_reply='last')

    asyncio.run(run())
    assert all(item.url.path == '/v1/memory/context' for item in requests)
    assert json.loads(requests[0].content) == {'turn_id': 'c1', 'user_id': 'A', 'query': 'q',
                                               'previous_ai_reply': 'answer'}
    assert json.loads(requests[1].content) == {'turn_id': 'c2', 'user_id': 'A', 'query': None,
                                               'previous_ai_reply': 'last'}
    # The single-request protocol carries its ID in the body, not in a header.
    assert all('idempotency-key' not in item.headers for item in requests)


def test_openapi_documents_the_single_request_protocol(harness):
    client, *_ = harness
    schema = client.get('/openapi.json').json()
    body = schema['paths']['/v1/memory/context']['post']['requestBody']
    refs = [item['$ref'] for item in body['content']['application/json']['schema']['anyOf']]
    assert refs[0].endswith('/TurnContextRequest')  # 主路径排在最前
    request = schema['components']['schemas']['TurnContextRequest']
    assert request['required'] == ['turn_id', 'user_id', 'query']
    assert set(request['properties']) == {'turn_id', 'user_id', 'query', 'previous_ai_reply',
                                          'query_occurred_at', 'previous_reply_occurred_at'}
    response = schema['components']['schemas']['TurnContextResponse']
    assert set(response['properties']) == {'turn_id', 'prompt'}


def test_legacy_key_and_turn_id_share_no_idempotency_namespace(harness):
    client, _, store, *_ = harness
    legacy = client.post('/v1/memory/context', headers=dict(AUTH, **{'Idempotency-Key': 'shared-id'}),
                         json={'user_id': 'A', 'previous_ai_reply': None, 'query': '旧协议问题'})
    assert legacy.status_code == 200
    # Same literal ID, different protocol: a real new turn, not a replayed one.
    result = turn(client, 'shared-id', query='新协议问题', previous='旧协议回答')
    assert result.status_code == 200, result.text
    assert set(result.json()) == {'turn_id', 'prompt'}
    assert len(store.list_shared_messages('bot')) == 3
