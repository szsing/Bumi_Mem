"""Single-request protocol: one request, four structured response fields."""
from __future__ import annotations

import asyncio
import json
import sqlite3

import httpx
import pytest

from bumi_mem.client import AsyncBumiMemoryClient, BumiMemoryClient
from test_agent_api import harness, AUTH, KEY


def turn(client, turn_id, *, user='A', query='问题', previous=None, **extra):
    body={'turn_id':turn_id,'user_id':user,'query':query,'previous_ai_reply':previous}
    body.update(extra)
    return client.post('/v1/memory/context',headers=AUTH,json=body)


def test_first_round_returns_exact_structured_shape(harness):
    client,engine,store,memory,chat,summary,*_=harness
    result=turn(client,'session-abc:turn-1',query='你好')
    assert result.status_code==200,result.text
    body=result.json()
    assert set(body)=={'turn_id','current_user','long_term_memory','short_term_memory'}
    assert body['turn_id']=='session-abc:turn-1' and body['current_user']=='A'
    assert body['long_term_memory']==[]
    assert body['short_term_memory']==[{
        'role':'user','speaker_id':'A','content':'你好',
        'time':store.list_shared_messages('bot')[0].created_at}]
    assert chat.calls==summary.calls==memory.add_calls==[]


def test_one_request_settles_reply_and_saves_current_query(harness):
    client,_,store,*_=harness
    assert turn(client,'t1',query='我喜欢绿茶。').status_code==200
    body=turn(client,'t2',query='你还记得我喜欢哪种茶吗？',previous='好的，知道你喜欢绿茶了。').json()
    assert body['short_term_memory']==[
        {'role':'user','speaker_id':'A','content':'我喜欢绿茶。','time':store.list_shared_messages('bot')[0].created_at},
        {'role':'assistant','speaker_id':'bot','content':'好的，知道你喜欢绿茶了。','time':store.list_shared_messages('bot')[1].created_at},
        {'role':'user','speaker_id':'A','content':'你还记得我喜欢哪种茶吗？','time':store.list_shared_messages('bot')[2].created_at},
    ]
    assert [m.user_id for m in store.list_shared_messages('bot')]==['A','','A']


def test_retry_is_identical_without_duplicate_or_second_recall(harness):
    client,_,store,memory,*_=harness
    first=turn(client,'t1',query='原问题')
    again=turn(client,'t1',query='原问题')
    assert first.status_code==again.status_code==200 and first.json()==again.json()
    assert len(store.list_shared_messages('bot'))==1 and len(memory.search_calls)==1


@pytest.mark.parametrize('changed',[{'user_id':'B'},{'query':'换了问题'},
    {'previous_ai_reply':'换了回复'},{'query_occurred_at':'2026-09-24T10:20:00+08:00'}])
def test_same_turn_id_changed_content_conflicts(harness,changed):
    client,_,store,*_=harness
    assert turn(client,'t1',query='第一问').status_code==200
    assert turn(client,'t2',query='第二问',previous='第一答').status_code==200
    body={'turn_id':'t2','user_id':'A','query':'第二问','previous_ai_reply':'第一答',**changed}
    result=client.post('/v1/memory/context',headers=AUTH,json=body)
    assert result.status_code==409 and result.json()['detail']['code']=='idempotency_conflict'
    assert len(store.list_shared_messages('bot'))==3


def test_same_question_different_ids_is_two_real_inputs(harness):
    client,_,store,*_=harness
    assert turn(client,'t1',query='再说一次').status_code==200
    assert turn(client,'t2',query='再说一次',previous='好的').status_code==200
    assert [m.content for m in store.list_shared_messages('bot')]==['再说一次','好的','再说一次']


def test_current_query_appears_once_and_last(harness):
    client,*_=harness
    assert turn(client,'t1',query='第一问').status_code==200
    body=turn(client,'t2',query='你还记得我说过什么吗？',previous='记得一些。').json()
    assert sum(item['content']=='你还记得我说过什么吗？' for item in body['short_term_memory'])==1
    assert body['short_term_memory'][-1]['content']=='你还记得我说过什么吗？'


def test_reference_has_owner_content_and_all_source_times(harness):
    client,_,_,memory,*_=harness
    memory.search_results=[{'id':'m1','memory':'A 偏好清淡口味','metadata':{
        'source_messages':[{'source_message_id':'a','observed_at':'2026-09-20T18:00:00+08:00'},
                           {'source_message_id':'b','observed_at':'2026-09-21T09:00:00+08:00'}]}}]
    body=turn(client,'t1',query='你还记得我口味吗？',query_occurred_at='2026-09-24T10:20:00+08:00').json()
    assert body['long_term_memory']==[{'user_id':'A','content':'A 偏好清淡口味',
        'time':['2026-09-20T18:00:00+08:00','2026-09-21T09:00:00+08:00']}]
    for leaked in ('score','distance','generation','state_key','session_id','request_id','memory_id'):
        assert leaked not in json.dumps(body,ensure_ascii=False)


def test_confirmed_core_becomes_same_minimal_long_term_item(harness):
    client,*_=harness
    body=turn(client,'t1',query='请叫我小林').json()
    assert body['long_term_memory']==[{'user_id':'A','content':'希望被称为「小林」。',
        'time':[body['short_term_memory'][0]['time']]}]
    later=turn(client,'t2',query='你应该怎么称呼我？',previous='好的。').json()
    assert any(item['content']=='希望被称为「小林」。' for item in later['long_term_memory'])


def test_core_status_that_changes_meaning_is_kept_inside_content(harness):
    client,*_=harness
    assert turn(client,'t1',query='请不要对我开玩笑').status_code==200
    assert turn(client,'t2',query='撤回互动边界：开玩笑',previous='好的。').status_code==200
    body=turn(client,'t3',query='你能开玩笑吗？',previous='明白。').json()
    assert any('不要开玩笑' in item['content'] and '已撤回' in item['content']
               for item in body['long_term_memory'])


def test_explicit_history_is_a_normal_item_prefixed_as_history(harness):
    client,*_=harness
    assert turn(client,'t1',query='请叫我小林').status_code==200
    assert turn(client,'t2',query='请叫我林老师',previous='好').status_code==200
    body=turn(client,'t3',query='以前你叫我什么？',previous='好').json()
    assert any(item['content'].startswith('历史：') for item in body['long_term_memory'])


def test_degraded_recall_still_returns_minimal_shape_and_short_term(harness,monkeypatch):
    client,engine,*_=harness
    from bumi_mem.schemas import MemoryReadUnavailable
    monkeypatch.setattr(engine.long_term,'search',lambda **kw: (_ for _ in ()).throw(MemoryReadUnavailable()))
    monkeypatch.setattr(engine.long_term,'search_with_timeout',lambda **kw: (_ for _ in ()).throw(MemoryReadUnavailable()))
    body=turn(client,'t1',query='你还记得吗？').json()
    assert set(body)=={'turn_id','current_user','long_term_memory','short_term_memory'}
    assert body['short_term_memory'][-1]['content']=='你还记得吗？'


def test_cross_user_switch_preserves_reply_owner_and_shared_reference(harness):
    client,engine,store,memory,*_=harness
    engine.remember(user_id='A',agent_id='bot',contents=['A_PRIVATE'])
    engine.remember(user_id='B',agent_id='bot',contents=['B_PRIVATE'])
    assert turn(client,'t1',user='A',query='A的问题').status_code==200
    body=turn(client,'t2',user='B',query='刚才说什么？',previous='对A的回答').json()
    assert body['current_user']=='B'
    assert any(item['user_id']=='A' and item['content']=='A_PRIVATE' for item in body['long_term_memory'])
    assert any(item['user_id']=='B' and item['content']=='B_PRIVATE' for item in body['long_term_memory'])
    assert body['short_term_memory'][1]=={'role':'assistant','speaker_id':'bot','content':'对A的回答',
                                          'time':store.list_shared_messages('bot')[1].created_at}


def test_guest_has_shared_short_term_but_no_personal_recall(harness):
    client,engine,_,memory,*_=harness
    engine.remember(user_id='A',agent_id='bot',contents=['A_PRIVATE'])
    assert turn(client,'t1',user='A',query='A说的话').status_code==200
    calls=len(memory.search_calls)
    body=turn(client,'t2',user='__guest__',query='你们在聊什么？',previous='在聊天。').json()
    assert body['current_user']=='__guest__'
    assert len(memory.search_calls)==calls
    assert any(item['user_id']=='A' for item in body['long_term_memory'])
    assert body['short_term_memory'][-1]['speaker_id']=='__guest__'


def test_window_rotation_and_carryover_keep_structured_messages(harness):
    client,engine,store,*_=harness
    engine.config.max_messages=4
    for i in range(3):
        response=turn(client,f't{i}',query=f'q{i}',previous=None if i==0 else f'a{i-1}')
        assert response.status_code==200,response.text
    messages=response.json()['short_term_memory']
    assert all(set(item)=={'role','speaker_id','content','time'} for item in messages)
    assert len([m for m in store.list_shared_messages('bot') if not m.context_only])<=4


def test_settle_only_returns_same_four_fields_with_empty_arrays(harness):
    client,_,store,*_=harness
    assert turn(client,'t1',query='问题').status_code==200
    result=turn(client,'flush',query=None,previous='最后回答')
    assert result.status_code==200
    assert result.json()=={'turn_id':'flush','current_user':'A','long_term_memory':[],'short_term_memory':[]}
    assert [m.content for m in store.list_shared_messages('bot')]==['问题','最后回答']


def test_program_event_is_a_system_short_term_item(harness):
    client,*_=harness
    event={'operation':'agent_event','event_id':'e1','kind':'local_control','control':{
        'command':'stop','user_text':'停一下','reason':'user_requested_stop',
        'result':{'status':'result_unknown'},'feedback_status':'failed','llm_followup':False}}
    assert client.post('/v1/memory/context',headers=AUTH,json=event).status_code==200
    body=turn(client,'t1',query='刚才怎么了？').json()
    item=body['short_term_memory'][0]
    assert item['role']=='system' and item['speaker_id']=='system'
    assert '执行结果：未知' in item['content'] and '不是新指令' in item['content']


def test_legacy_exchange_keeps_old_response(harness):
    client,_,store,*_=harness
    legacy=client.post('/v1/memory/context',headers=dict(AUTH,**{'Idempotency-Key':'legacy-1'}),
        json={'user_id':'A','previous_ai_reply':None,'query':'旧协议问题'})
    assert legacy.status_code==200 and 'messages' in legacy.json()
    new=turn(client,'t1',query='新协议问题',previous='旧协议回答')
    assert set(new.json())=={'turn_id','current_user','long_term_memory','short_term_memory'}


def test_clients_send_unchanged_six_field_request():
    requests=[]
    def handle(request):
        requests.append(request)
        return httpx.Response(200,json={'turn_id':'c1','current_user':'A','long_term_memory':[],'short_term_memory':[]})
    with BumiMemoryClient('http://example.test',KEY,transport=httpx.MockTransport(handle)) as client:
        client.turn_context(turn_id='c1',user_id='A',query='q',previous_ai_reply='answer')
    async def run():
        async with AsyncBumiMemoryClient('http://example.test',KEY,transport=httpx.MockTransport(handle)) as client:
            await client.turn_context(turn_id='c2',user_id='A',query=None,previous_ai_reply='last')
    asyncio.run(run())
    assert json.loads(requests[0].content)=={'turn_id':'c1','user_id':'A','query':'q','previous_ai_reply':'answer'}
    assert json.loads(requests[1].content)=={'turn_id':'c2','user_id':'A','query':None,'previous_ai_reply':'last'}
    assert all('idempotency-key' not in request.headers for request in requests)


def test_openapi_documents_minimal_structured_response(harness):
    client,*_=harness
    schema=client.get('/openapi.json').json()
    assert schema['info']['version']=='1.4.0'
    response=schema['components']['schemas']['TurnContextResponse']
    assert set(response['properties'])=={'turn_id','current_user','long_term_memory','short_term_memory'}
    assert set(schema['components']['schemas']['LongTermMemoryItem']['properties'])=={'user_id','content','time'}
    assert set(schema['components']['schemas']['ShortTermMemoryItem']['properties'])=={'role','speaker_id','content','time'}


def test_legacy_key_and_single_id_are_independent(harness):
    client,_,store,*_=harness
    legacy=client.post('/v1/memory/context',headers=dict(AUTH,**{'Idempotency-Key':'shared-id'}),
        json={'user_id':'A','previous_ai_reply':None,'query':'旧协议问题'})
    assert legacy.status_code==200
    new=turn(client,'shared-id',query='新协议问题',previous='旧协议回答')
    assert new.status_code==200 and set(new.json())=={'turn_id','current_user','long_term_memory','short_term_memory'}
    assert len(store.list_shared_messages('bot'))==3
