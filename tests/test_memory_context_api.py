from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from bumi_mem.client import BumiMemoryClient, AsyncBumiMemoryClient
from bumi_mem.schemas import MemoryReadUnavailable
from bumi_mem.session_store import SQLiteSessionStore
from test_agent_api import harness, AUTH, KEY
from test_bumi_memory import make_engine


def context(client, user='A', query='question', previous=None, key=None):
    headers = dict(AUTH)
    if key is not None:
        headers['Idempotency-Key'] = key
    return client.post('/v1/memory/context', headers=headers,
                       json={'user_id': user, 'previous_ai_reply': previous, 'query': query})


def drain(engine):
    while engine.worker.run_once():
        pass


def test_single_endpoint_first_query_does_not_invent_ai_or_call_answer_model(harness):
    client, engine, store, memory, chat, summary, _, _ = harness
    engine.remember(user_id='A',agent_id='bot',contents=['A_PRIVATE'])
    result = context(client,key='request-1')
    assert result.status_code == 200, result.text
    body = result.json()
    assert set(body) == {'request_id','user_id','agent_id','messages','has_prompt',
                         'previous_reply_saved','recall_status','consolidation_status','core_write'}
    assert body['has_prompt'] and not body['previous_reply_saved']
    assert body['consolidation_status'] == 'not_triggered'
    assert 'A_PRIVATE' in body['messages'][0]['content']
    assert '你是 Bumi' not in body['messages'][0]['content']
    assert body['messages'][-1]['content'].endswith('\nquestion')
    assert len(store.list_shared_messages('bot')) == 1
    assert chat.calls == summary.calls == memory.add_calls == []


@pytest.mark.parametrize('empty',[None,'','   '])
def test_first_round_accepts_empty_previous_reply(harness, empty):
    client, _, store, *_ = harness
    assert context(client, previous=empty).status_code == 200
    assert [m.role for m in store.list_shared_messages('bot')] == ['user']


def test_A_then_B_agent_reply_has_only_agent_identity_and_B_has_private_L3(harness):
    client, engine, store, memory, *_ = harness
    for user in ['A','B']:
        engine.remember(user_id=user,agent_id='bot',contents=[user+'_PRIVATE'])
    assert context(client,'A','A_PUBLIC').status_code == 200
    result = context(client,'B','你刚刚说什么？','AGENT_PUBLIC').json()
    assert result['previous_reply_saved']
    assert [m.role for m in store.list_shared_messages('bot')] == ['user','assistant','user']
    assert [(m.user_id,m.agent_id,m.content) for m in store.list_shared_messages('bot')] == [
        ('A','bot','A_PUBLIC'),('','bot','AGENT_PUBLIC'),('B','bot','你刚刚说什么？')]
    messages = result['messages']
    direct_context, shared_context = messages[0]['content'].split('[共享 L3 引用 / shared memory references]\n', 1)
    assert 'A_PRIVATE' not in direct_context
    refs = json.loads(shared_context.split('\n', 1)[0])
    assert any(r['owner_user_id'] == 'A' and r['memory'] == 'A_PRIVATE' for r in refs)
    assert any(r['owner_user_id'] == 'B' and r['memory'] == 'B_PRIVATE' for r in refs)
    assert 'A_PUBLIC' in messages[1]['content']
    header = messages[2]['content'].split('\n',1)[0]
    assert '"agent_id": "bot"' in header and '"speaker_role": "assistant"' in header
    assert 'user_id' not in header and 'reply_to' not in header
    assert [call['user_id'] for call in memory.search_calls] == ['A', 'B']


def test_exchange_retry_never_duplicates_previous_reply_or_current_query(harness):
    client, _, store, memory, *_ = harness
    first = context(client,'A','a',key='first').json()
    body = context(client,'B','b','answer a',key='second').json()
    replay = context(client,'B','b','answer a',key='second')
    assert replay.status_code == 200 and replay.json() == body
    assert len(store.list_shared_messages('bot')) == 3 and len(memory.search_calls) == 2
    for user, query, previous in [('C','b','answer a'),('B','changed','answer a'),('B','b','changed')]:
        assert context(client,user,query,previous,key='second').status_code == 409
    assert context(client,'A','a',key='first').status_code == 409
    with sqlite3.connect(store.path) as c:
        assert c.execute('SELECT response_json FROM context_requests WHERE request_id=?',
                         (first['request_id'],)).fetchone()[0] is None


def test_flush_uses_same_endpoint_no_new_query_and_is_idempotent(harness):
    client, _, store, memory, *_ = harness
    context(client,'A','a')
    response = context(client,'B',None,'last agent answer',key='flush')
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['has_prompt'] is False and body['messages'] == []
    assert body['recall_status'] == 'not_requested' and body['previous_reply_saved']
    assert context(client,'B',None,'last agent answer',key='flush').json() == body
    assert len(store.list_shared_messages('bot')) == 2 and len(memory.search_calls) == 1
    assert [(m.user_id,m.role) for m in store.list_shared_messages('bot')] == [('A','user'),('','assistant')]
    assert context(client,'C','new').status_code == 200
    assert context(client,'B',None,'last agent answer',key='flush').json() == body


def test_nonempty_reply_without_previous_turn_rejected_without_saving_query(harness):
    client, _, store, *_ = harness
    response = context(client, previous='orphan')
    assert response.status_code == 409 and response.json()['detail']['code'] == 'no_previous_turn'
    assert store.list_shared_messages('bot') == []


def test_empty_previous_reply_explicitly_closes_no_output_round(harness):
    client, _, store, *_ = harness
    first = context(client,'A','a').json()
    result = context(client,'B','b',previous=None)
    assert result.status_code == 200 and not result.json()['previous_reply_saved']
    assert [m.role for m in store.list_shared_messages('bot')] == ['user','user']
    with sqlite3.connect(store.path) as c:
        assert c.execute('SELECT t.status FROM interaction_turns t JOIN context_requests r USING(turn_id) '
                         'WHERE r.request_id=?',(first['request_id'],)).fetchone()[0] == 'cancelled'


def test_long_user_pause_does_not_expire_previous_agent_reply(harness):
    client, _, store, *_ = harness
    context(client,'A','a')
    with sqlite3.connect(store.path) as c:
        c.execute('UPDATE interaction_turns SET expires_at=0')
    result = context(client,'B','b','late but valid agent speech')
    assert result.status_code == 200 and result.json()['previous_reply_saved']
    assert 'late but valid agent speech' in str(result.json()['messages'])


def test_rotation_happens_after_previous_reply_before_current_query(harness):
    client, engine, store, memory, chat, summary, _, _ = harness
    engine.config.max_messages = 4
    context(client,'A','A_FACT')
    context(client,'B','B_FACT','A_ANSWER')
    result = context(client,'C','C_CURRENT','B_ANSWER').json()
    assert result['consolidation_status'] == 'queued'
    rows = store.list_shared_messages('bot')
    assert [m.context_only for m in rows] == [True,True,True,True,False]
    assert [m.content for m in rows] == ['A_FACT','A_ANSWER','B_FACT','B_ANSWER','C_CURRENT']
    assert [m.user_id for m in rows] == ['A','','B','','C']
    assert chat.calls == summary.calls == memory.add_calls == []
    drain(engine)
    assert len(summary.calls) == len(memory.add_calls) == 2
    assert [call['user_id'] for call in memory.add_calls] == ['A','B']
    for call in memory.add_calls:
        assert [m['content'].split('\n', 1)[1] for m in call['messages']] == [call['user_id']+'_FACT',call['user_id']+'_ANSWER']
    assert not store.list_jobs(user_id='C',agent_id='bot')


def test_flush_can_trigger_background_memory_extraction(harness):
    client, engine, _, memory, _, summary, _, _ = harness
    engine.config.max_messages = 2
    context(client,'A','A_FACT')
    result = context(client,'B',None,'agent final answer').json()
    assert result['consolidation_status'] == 'queued'
    drain(engine)
    assert len(summary.calls) == len(memory.add_calls) == 1
    assert memory.add_calls[0]['user_id'] == 'A'  # source query, not the current request user


def test_transaction_rolls_back_previous_reply_rotation_and_query_together(harness, monkeypatch):
    client, engine, store, *_ = harness
    engine.config.max_messages = 2
    context(client,'A','a')
    original = store._reserve_turn
    def fail(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('sensitive internal error')
    monkeypatch.setattr(store,'_reserve_turn',fail)
    response = context(client,'B','b','answer',key='second')
    assert response.status_code == 503 and 'sensitive internal' not in response.text
    assert [m.content for m in store.list_shared_messages('bot')] == ['a']
    assert store.list_jobs(user_id='A',agent_id='bot') == []
    monkeypatch.setattr(store,'_reserve_turn',original)
    assert context(client,'B','b','answer',key='second').status_code == 200
    assert len(store.list_jobs(user_id='A',agent_id='bot')) == 1
    assert [m.content for m in store.list_shared_messages('bot')] == ['a','answer','b']


@pytest.mark.parametrize('stage',['prompt','response'])
def test_retry_after_post_commit_failure_recovers_without_reappending(harness, monkeypatch, stage):
    client, engine, store, memory, *_ = harness
    context(client,'A','a')
    obj,name = (engine,'_prompt_for_turn') if stage == 'prompt' else (store,'save_context_response')
    original = getattr(obj,name)
    def fail(*args,**kwargs):
        raise RuntimeError('injected')
    monkeypatch.setattr(obj,name,fail)
    assert context(client,'B','b','answer',key='second').status_code == 503
    assert [m.content for m in store.list_shared_messages('bot')] == ['a','answer','b']
    monkeypatch.setattr(obj,name,original)
    result = context(client,'B','b','answer',key='second')
    assert result.status_code == 200 and result.json()['previous_reply_saved']
    assert len(store.list_shared_messages('bot')) == 3
    assert len(memory.search_calls) == 2


def test_single_request_restart_preserves_pending_previous_speech(tmp_path):
    engine, store, memory, *_ = make_engine(tmp_path)
    response = engine.process_context(user_id='A',agent_id='bot',query='a',previous_ai_reply=None,idempotency_key='a')
    engine.close()
    restarted, _, _, *_ = make_engine(tmp_path,long_term=memory)
    assert restarted.process_context(user_id='A',agent_id='bot',query='a',previous_ai_reply=None,idempotency_key='a') == response
    result = restarted.process_context(user_id='B',agent_id='bot',query='b',previous_ai_reply='agent answer',idempotency_key='b')
    assert result['previous_reply_saved']
    assert [(m.role,m.user_id) for m in store.list_shared_messages('bot')] == [('user','A'),('assistant',''),('user','B')]
    assert len(memory.search_calls) == 2


def test_read_failure_degrades_to_short_term_after_previous_reply_saved(harness, monkeypatch):
    client, engine, store, *_ = harness
    context(client,'A','a')
    def fail(**kwargs):
        raise MemoryReadUnavailable('private error')
    monkeypatch.setattr(engine.long_term,'search',fail)
    result = context(client,'B','b','agent speech')
    assert result.status_code == 200 and result.json()['recall_status'] == 'unavailable'
    assert result.json()['previous_reply_saved']
    assert len(store.list_shared_messages('bot')) == 3
    assert 'private error' not in result.text


def test_forget_revokes_cached_private_prompt_but_keeps_public_agent_reply(harness):
    client, engine, store, memory, *_ = harness
    engine.config.max_messages = 2
    first = context(client,'A','A_OLD',key='a').json()
    engine.forget_all(user_id='A',agent_id='bot')
    assert context(client,'A','A_OLD',key='a').status_code == 409
    result = context(client,'B','B_NEW','agent speech after A')
    assert result.status_code == 200
    assert 'agent speech after A' in str(result.json()['messages'])
    drain(engine)
    assert not [c for c in memory.add_calls if c['user_id']=='A']
    with sqlite3.connect(store.path) as c:
        assert c.execute('SELECT response_json FROM context_requests WHERE request_id=?',(first['request_id'],)).fetchone()[0] is None
        assert c.execute("SELECT memory_excluded FROM messages WHERE content='agent speech after A' LIMIT 1").fetchone()[0] == 1


def test_concurrent_retries_with_same_key_have_one_database_effect(harness):
    client, _, store, memory, *_ = harness
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _:context(client,key='same'),range(2)))
    assert all(r.status_code == 200 for r in replies)
    assert replies[0].json() == replies[1].json()
    assert len(store.list_shared_messages('bot')) == 1 and len(memory.search_calls) == 1


def test_auth_input_limits_extra_fields_and_old_routes(harness):
    client, _, store, *_ = harness
    body={'user_id':'A','previous_ai_reply':None,'query':'q'}
    for headers in ({},{'Authorization':'Bearer invalid'}):
        assert client.post('/v1/memory/context',json=body,headers=headers).status_code == 401
    for invalid in ({**body,'agent_id':'other'}, {**body,'turn_id':'old-interface'},
                    {**body,'user_id':' A'}, {**body,'user_id':None}, {**body,'query':' '},
                    {**body,'query':'x'*16001}, {**body,'previous_ai_reply':'x'*64001}):
        assert client.post('/v1/memory/context',json=invalid,headers=AUTH).status_code == 422
    assert store.list_shared_messages('bot') == []
    for path in ['/v1/memory/prompt','/v1/memory/reply']:
        assert client.post(path,json=body,headers=AUTH).status_code == 404
    response=client.post('/v1/memory/context',headers={**AUTH,'Content-Type':'application/json'},content=iter([b'x'*600000,b'x'*600000]))
    assert response.status_code == 413


def test_openapi_contains_one_business_endpoint_and_health_ready(harness):
    client, *_ = harness
    schema=client.get('/openapi.json').json()
    assert [path for path in schema['paths'] if path.startswith('/v1/')] == ['/v1/memory/context']
    properties=schema['components']['schemas']['ContextRequest']['properties']
    assert set(properties) == {'user_id','previous_ai_reply','query',
                               'query_occurred_at','previous_reply_occurred_at'}
    required = schema['components']['schemas']['ContextRequest'].get('required', [])
    assert 'query_occurred_at' not in required and 'previous_reply_occurred_at' not in required
    assert KEY not in json.dumps(schema)
    assert client.get('/health/live').status_code == 200
    assert client.get('/health/ready').json() == {'status':'ready','warmed_up':True}


def test_migrate_v4_speaker_identity_without_changing_extraction_owner(tmp_path):
    engine, store, memory, *_ = make_engine(tmp_path)
    engine.chat(user_id='A',agent_id='bot',text='A fact')
    receipt=engine.end_session(user_id='A',agent_id='bot')
    with sqlite3.connect(store.path) as c:
        c.execute("UPDATE messages SET user_id='A' WHERE role='assistant'")
        c.execute('DROP TABLE context_requests')
        c.execute('ALTER TABLE interaction_turns DROP COLUMN protocol')
        c.execute('PRAGMA user_version=4')
    engine.close()
    new, upgraded, _, *_ = make_engine(tmp_path,long_term=memory)
    rows=upgraded.list_shared_messages('bot')
    assert [m.user_id for m in rows] == ['A','']
    assert upgraded.get_job(receipt.job_id).user_id == 'A'
    drain(new)
    assert memory.add_calls[0]['user_id'] == 'A'
    with sqlite3.connect(store.path) as c:
        assert c.execute('PRAGMA user_version').fetchone()[0] == 6


def test_sync_and_async_clients_only_use_context_route():
    requests=[]
    def handle(request):
        requests.append(request)
        return httpx.Response(200,json={'messages':[],'has_prompt':False})
    with BumiMemoryClient('http://example.test',KEY,transport=httpx.MockTransport(handle)) as client:
        client.get_context('B','q',previous_ai_reply='answer',idempotency_key='sync')
    async def run():
        async with AsyncBumiMemoryClient('http://example.test',KEY,transport=httpx.MockTransport(handle)) as client:
            await client.get_context('B',None,previous_ai_reply='last',idempotency_key='async')
    asyncio.run(run())
    assert all(r.url.path == '/v1/memory/context' for r in requests)
    assert json.loads(requests[0].content) == {'user_id':'B','previous_ai_reply':'answer','query':'q'}
    assert json.loads(requests[1].content)['query'] is None
    assert requests[0].headers['idempotency-key'] == 'sync'
