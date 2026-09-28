"""流式前缀预取：缓存策略、协议/引擎集成，以及"不写 L1"的硬约束。"""
from __future__ import annotations

from bumi_mem.agent_bridge import prefetch_payload
from bumi_mem.prefetch import RetrievalPrefetchCache, normalize_prefix
from test_agent_api import harness, AUTH


def post(client, **payload):
    return client.post('/v1/memory/context', headers=AUTH, json=payload)


def seed(memory, rows, user='A', agent='bot'):
    memory.memories[(user, agent)] = rows


def prefetch(client, *, cid='run:turn-1', prefix='我想找一家安静的', user='A', **extra):
    payload = dict(operation='prefetch', client_turn_id=cid, query_prefix=prefix)
    if user is not None:
        payload['user_id'] = user
    return post(client, **payload, **extra)


def prepare(client, *, cid='run:turn-1', query='我想找一家安静的咖啡馆', user='A', prefetch_id=None):
    payload = dict(operation='prepare', client_turn_id=cid, query=query, user_id=user)
    if prefetch_id is not None:
        payload['prefetch_id'] = prefetch_id
    return post(client, **payload)


# --------------------------------------------------------------------------- #
# 缓存策略（纯内存，不依赖引擎）
# --------------------------------------------------------------------------- #
def test_prefix_normalization_ignores_whitespace() -> None:
    assert normalize_prefix(' 我 想找 一家 ') == '我想找一家'


def test_cache_hit_within_growth_then_consumed() -> None:
    cache = RetrievalPrefetchCache(ttl_seconds=3, max_growth_chars=6)
    entry = cache.put(agent_id='bot', user_id='A', client_turn_id='t1',
                      query_prefix='我想找一家安静的', generation=0, recall_status='ok',
                      rows=[{'id': 'm1'}])
    outcome, rows, age = cache.take(entry.prefetch_id, agent_id='bot', user_id='A',
                                    client_turn_id='t1', query='我想找一家安静的咖啡馆',
                                    generation=0)
    assert (outcome, [row['id'] for row in rows]) == ('hit', ['m1'])
    assert age is not None and age >= 0
    # 消费式：同一 prefetch_id 不能被第二轮复用
    assert cache.take(entry.prefetch_id, agent_id='bot', user_id='A', client_turn_id='t1',
                      query='我想找一家安静的咖啡馆', generation=0)[0] == 'miss'
    assert cache.stats.hit == 1 and cache.stats.miss == 1


def test_cache_rejects_stale_prefix_owner_and_generation() -> None:
    cache = RetrievalPrefetchCache(ttl_seconds=3, max_growth_chars=3)

    def fresh():
        return cache.put(agent_id='bot', user_id='A', client_turn_id='t1',
                         query_prefix='我想找一家', generation=0, recall_status='ok',
                         rows=[{'id': 'm1'}]).prefetch_id

    # 终稿不以缓存前缀开头
    assert cache.take(fresh(), agent_id='bot', user_id='A', client_turn_id='t1',
                      query='今天天气不错', generation=0)[0] == 'stale'
    # 增长超过阈值
    assert cache.take(fresh(), agent_id='bot', user_id='A', client_turn_id='t1',
                      query='我想找一家安静的咖啡馆', generation=0)[0] == 'stale'
    # 身份/轮次/agent/generation 不一致
    assert cache.take(fresh(), agent_id='bot', user_id='B', client_turn_id='t1',
                      query='我想找一家', generation=0)[0] == 'stale'
    assert cache.take(fresh(), agent_id='bot', user_id='A', client_turn_id='t2',
                      query='我想找一家', generation=0)[0] == 'stale'
    assert cache.take(fresh(), agent_id='other', user_id='A', client_turn_id='t1',
                      query='我想找一家', generation=0)[0] == 'stale'
    assert cache.take(fresh(), agent_id='bot', user_id='A', client_turn_id='t1',
                      query='我想找一家', generation=7)[0] == 'stale'
    assert cache.stats.stale == 6


def test_cache_ttl_expiry_and_scope_invalidation() -> None:
    now = [0.0]
    cache = RetrievalPrefetchCache(ttl_seconds=1.0, clock=lambda: now[0])
    entry = cache.put(agent_id='bot', user_id='A', client_turn_id='t1',
                      query_prefix='我想找一家', generation=0, recall_status='ok', rows=[])
    now[0] = 2.0
    assert cache.take(entry.prefetch_id, agent_id='bot', user_id='A', client_turn_id='t1',
                      query='我想找一家', generation=0)[0] == 'expired'
    cache.put(agent_id='bot', user_id='A', client_turn_id='t1', query_prefix='我想找一家',
              generation=0, recall_status='ok', rows=[])
    assert cache.pending() == 1
    assert cache.invalidate_scope(user_id='A', agent_id='bot') == 1
    assert cache.pending() == 0


# --------------------------------------------------------------------------- #
# 协议与引擎集成
# --------------------------------------------------------------------------- #
def test_prefetch_does_not_write_l1_or_create_a_turn(harness):
    client, engine, store, memory, *_ = harness
    seed(memory, [{'id': 'm1', 'memory': '用户常去一家安静的咖啡馆'}])
    response = prefetch(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['operation'] == 'prefetch' and body['prefetch_id'].startswith('pf_')
    assert body['candidate_count'] == 1 and body['recall_status'] == 'ok'
    assert 'messages' not in body and 'memory_context' not in body
    assert store.list_shared_messages('bot') == []          # 没有写 L1
    assert store.pending_agent_turn('bot') is None          # 没有建轮次
    assert len(memory.search_calls) == 1                    # 只跑了一次检索


def test_prepare_reuses_prefetch_and_skips_a_second_search(harness):
    client, engine, store, memory, *_ = harness
    seed(memory, [{'id': 'm1', 'memory': '用户常去一家安静的咖啡馆'}])
    pf = prefetch(client).json()
    response = prepare(client, prefetch_id=pf['prefetch_id'])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['recall_status'] == 'ok' and body['has_prompt']
    assert len(memory.search_calls) == 1                    # 复用：没有再检索
    assert engine.prefetch_stats()['hit'] == 1
    assert engine.prefetch_stats()['serial_samples'] == 1


def test_prepare_recomputes_when_prefetch_is_stale(harness):
    client, engine, store, memory, *_ = harness
    seed(memory, [{'id': 'm1', 'memory': '用户常去一家安静的咖啡馆'}])
    pf = prefetch(client).json()
    response = prepare(client, query='完全不相干的问题', prefetch_id=pf['prefetch_id'])
    assert response.status_code == 200, response.text
    assert len(memory.search_calls) == 2                    # 预取一次 + 重算一次
    assert engine.prefetch_stats()['stale'] == 1


def test_prepare_ignores_expired_prefetch(harness):
    client, engine, store, memory, *_ = harness
    seed(memory, [{'id': 'm1', 'memory': '用户常去一家安静的咖啡馆'}])
    pf = prefetch(client).json()
    entry = next(iter(engine.prefetch._entries.values()))
    entry.expires_at = entry.created_at - 1                 # 模拟已过期
    assert prepare(client, prefetch_id=pf['prefetch_id']).status_code == 200
    assert len(memory.search_calls) == 2
    assert engine.prefetch_stats()['expired'] == 1


def test_guest_prefetch_is_not_cached(harness):
    client, engine, store, memory, *_ = harness
    body = prefetch(client, user='__guest__').json()
    assert body['prefetch_id'] is None and body['recall_status'] == 'not_requested'
    assert body['candidate_count'] == 0
    assert engine.prefetch.pending() == 0
    assert memory.search_calls == []


def test_forget_all_drops_pending_prefetch(harness):
    client, engine, store, memory, *_ = harness
    seed(memory, [{'id': 'm1', 'memory': '用户常去一家安静的咖啡馆'}])
    prefetch(client)
    assert engine.prefetch.pending() == 1
    engine.forget_all(user_id='A', agent_id='bot')
    assert engine.prefetch.pending() == 0


def test_bridge_helper_builds_prefetch_payload() -> None:
    payload = prefetch_payload(query_prefix=' 我想找一家安静的 ', request_turn_id='run:1',
                               identity={'status': 'matched', 'profile_id': 'p1'},
                               identity_enabled=True, identity_mode='label', prefix_revision=2)
    assert payload['operation'] == 'prefetch'
    assert payload['query_prefix'] == '我想找一家安静的'
    assert payload['client_turn_id'] == 'run:1' and payload['prefix_revision'] == 2
    assert payload['speaker']['profile_id'] == 'p1'
    try:
        prefetch_payload(query_prefix='   ', request_turn_id='run:1')
    except ValueError as exc:
        assert 'prefix' in str(exc)
    else:  # pragma: no cover - 明确失败比静默通过好
        raise AssertionError('empty prefix must be rejected')
