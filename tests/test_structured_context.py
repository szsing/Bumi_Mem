"""响应里的结构化记忆视图：短期窗口 / 长期引用（带 owner）/ 当前用户核心状态。"""
from test_agent_api import harness, AUTH


def post(client, **payload):
    return client.post('/v1/memory/context', headers=AUTH, json=payload)


def seed(memory, rows, user='A', agent='bot'):
    memory.memories[(user, agent)] = rows


def test_prepare_exposes_short_term_and_long_term(harness):
    client, engine, store, memory, *_ = harness
    seed(memory, [{'id': 'm_1', 'memory': '用户住在杭州，喜欢手冲咖啡。',
                   'created_at': '2026-09-20T10:00:00+08:00',
                   'updated_at': '2026-09-22T09:00:00+08:00'}])
    body = post(client, operation='prepare', client_turn_id='t1', user_id='A',
                query='我住在哪个城市？').json()

    # 短期：本轮的 query 在窗口最后一条，原文（不带 speaker 标头）
    last = body['short_term'][-1]
    assert last['role'] == 'user' and last['content'] == '我住在哪个城市？'
    assert last['user_id'] == 'A' and last['carry_over'] is False
    assert last['created_at'] and last['message_id']
    # 长期：带 owner 的引用，而不是无归属的字符串
    ref = body['long_term'][0]
    assert ref['owner_user_id'] == 'A' and ref['memory_id'] == 'm_1'
    assert '杭州' in ref['memory'] and ref['reference_status'] == 'active'
    assert ref['created_at'] == '2026-09-20T10:00:00+08:00'
    assert body['core_state'] == []


def test_single_turn_response_carries_the_same_views(harness):
    client, engine, store, memory, *_ = harness
    seed(memory, [{'id': 'm_1', 'memory': '用户住在杭州。'}])
    body = post(client, turn_id='doc-1', user_id='A', query='我住在哪？').json()
    assert body['short_term'][-1]['content'] == '我住在哪？'
    assert body['long_term'][0]['owner_user_id'] == 'A'
    # prompt 与结构化视图同源：prompt 里也能看到同一事实
    assert '杭州' in body['prompt'] and '杭州' in body['long_term'][0]['memory']


def test_guest_gets_short_term_but_no_long_term_or_core_state(harness):
    client, engine, store, memory, *_ = harness
    seed(memory, [{'id': 'm_1', 'memory': 'A 的私密事实'}])
    body = post(client, operation='prepare', client_turn_id='t2', user_id='__guest__',
                query='你好').json()
    assert body['short_term'][-1]['content'] == '你好'
    assert body['short_term'][-1]['user_id'] == '__guest__'
    assert body['long_term'] == [] and body['core_state'] == []
    assert memory.search_calls == []          # 陌生身份不检索个人 L3
