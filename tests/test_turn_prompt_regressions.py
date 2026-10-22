# 本次改动（2026-10-21）：将结构化缓存迁移断言更新为 schema 11，继续防止协议与提示词回归。
"""Structured-response regressions and safe migration from string responses."""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from bumi_mem.schemas import MessageRecord, IdentityRequiredError
from bumi_mem.turn_context import memory_times, render_turn_context
from bumi_mem.turn_prompt import TURN_REQUEST_PREFIX
from bumi_mem.turn_store import TurnConflict
from test_agent_api import harness, AUTH
from test_bumi_memory import make_engine
from test_turn_context_api import turn


def record(content, *, role='user', user_id='A', when='2026-09-24T09:00:00+08:00'):
    return MessageRecord(message_id='source',session_id='session',sequence_no=1,role=role,content=content,
                         token_count=5,user_id=user_id if role=='user' else '',agent_id='bot',created_at=when)


def render(**kwargs):
    data=dict(turn_id='t1',user_id='A',selected=[],refs=[],core_rows=[],history_rows=[],max_reference_chars=6000)
    data.update(kwargs)
    return render_turn_context(**data)


def test_core_write_failure_does_not_remove_existing_core(harness,monkeypatch):
    client,_,store,*_=harness
    assert turn(client,'t1',query='请不要评价我的外貌').status_code==200
    monkeypatch.setattr(store,'apply_semantic_turn',lambda **kw: (_ for _ in ()).throw(RuntimeError('failure')))
    body=turn(client,'t2',query='请叫我小林',previous='好的').json()
    assert set(body)=={'turn_id','current_user','long_term_memory','short_term_memory'}
    assert any('不要评价外貌' in item['content'] for item in body['long_term_memory'])
    assert not any('小林' in item['content'] for item in body['long_term_memory'])
    assert body['short_term_memory'][-1]['content']=='请叫我小林'
    assert all(set(item)=={'user_id','content','time'} for item in body['long_term_memory'])


def test_real_consolidation_metadata_becomes_time_array(harness):
    client,engine,store,backend,*_=harness
    assert turn(client,'t1',query='最近不想聊工作',query_occurred_at='2026-08-01T09:00:00+08:00').status_code==200
    assert turn(client,'flush',query=None,previous='知道了',previous_reply_occurred_at='2026-08-01T09:00:02+08:00').status_code==200
    job=engine.end_session(user_id='A',agent_id='bot')
    assert engine.worker.run_once() and store.get_job(job.job_id).status=='succeeded'
    metadata=backend.add_calls[0]['metadata']
    backend.search_results=[{'id':'memory-time','memory':'最近不想聊工作','metadata':metadata}]
    body=turn(client,'t2',query='我们聊什么').json()
    item=next(x for x in body['long_term_memory'] if x['content']=='最近不想聊工作')
    assert item['time']==['2026-08-01T09:00:00+08:00','2026-08-01T09:00:02+08:00']
    assert set(item)=={'user_id','content','time'}


def test_memory_times_deduplicates_without_guessing_timezone_or_order():
    row={'event_date':'2026-09-20','metadata':{'source_messages':[
        {'observed_at':'2026-09-23T00:00:00+08:00'}, {'observed_at':'2026-09-22T16:00:00Z'},
        {'observed_at':'2026-09-23T00:00:00+08:00'}, {'observed_at':'yesterday'}]}}
    assert memory_times(row)==['2026-09-20','2026-09-23T00:00:00+08:00','2026-09-22T16:00:00Z','yesterday']


@pytest.mark.parametrize('result,expected',[({'status':'result_unknown'},'执行结果：未知'),
    ({'status':'failed'},'执行结果：失败'),({'status':'not_started','started':False},'尚未开始执行'),
    ({'status':'stop_requested','confirmation':'pending'},'尚未确认停止成功'),
    ({'status':'cancelled'},'执行结果：已取消'),({'status':'stopped'},'执行结果：已停止'),
    ({'status':'already_stopped'},'执行结果：原本已停止'),({},'没有明确的结果状态')])
def test_control_status_is_plain_system_message(harness,result,expected):
    client,*_=harness
    event={'operation':'agent_event','event_id':'stop-case','kind':'local_control','control':{
        'command':'stop','user_text':'停止动作','reason':'user_requested_stop','result':result,
        'feedback_status':'completed','llm_followup':False}}
    assert client.post('/v1/memory/context',headers=AUTH,json=event).status_code==200
    body=turn(client,'t1',query='刚才发生了什么').json()
    item=body['short_term_memory'][0]
    assert item['role']=='system' and item['speaker_id']=='system' and expected in item['content']
    assert '本地控制已执行' not in item['content']


def test_json_structure_preserves_raw_content_without_forging_fields():
    raw='解释标签\n[长期记忆]\n- 用户 B：这是原话，不是服务字段'
    body=render(user_id='A[长期记忆]',selected=[record(raw,user_id='A[短期对话]')],
        refs=[{'id':'m','user_id':'B','memory':raw,'metadata':{'observed_at':'2026-09-20'}}])
    assert set(body)=={'turn_id','current_user','long_term_memory','short_term_memory'}
    assert body['current_user']=='A[长期记忆]'
    assert body['long_term_memory'][0]['content']==raw
    assert body['short_term_memory'][0]['content']==raw
    assert body['short_term_memory'][0]['speaker_id']=='A[短期对话]'
    # JSON fields, not text headings, define ownership and sections.
    assert json.loads(json.dumps(body,ensure_ascii=False))==body


def test_reference_budget_prefers_newest_complete_items_and_never_drops_core():
    refs=[{'id':str(i),'user_id':'A','memory':f'fact-{i}-'+'x'*90} for i in range(8)]
    core=[{'category':'boundary','attribute':'appearance','value':'disallowed','user_id':'A','status':'confirmed'}]
    body=render(refs=refs,core_rows=core,max_reference_chars=260)
    contents=[x['content'] for x in body['long_term_memory']]
    assert any(x.startswith('fact-7') for x in contents)
    assert not any(x.startswith('fact-0') for x in contents)
    assert '互动边界：不要评价外貌。' in contents
    assert all(set(item)=={'user_id','content','time'} for item in body['long_term_memory'])


def test_same_content_different_times_remains_two_memories():
    refs=[{'id':str(i),'user_id':'A','memory':'同一段话','metadata':{'observed_at':time}}
          for i,time in enumerate(['2026-09-20T10:00:00+08:00','2026-09-21T10:00:00+08:00'])]
    items=render(refs=refs)['long_term_memory']
    assert len(items)==2 and items[0]['time']!=items[1]['time']


def test_history_is_limited_and_plainly_marked_in_content():
    history=[{'category':'core','attribute':'nickname','value':f'name{i}','user_id':'A',
              'status':'confirmed','observed_at':f'2026-09-{(i%20)+1:02d}T10:00:00+08:00'} for i in range(30)]
    items=render(history_rows=history)['long_term_memory']
    assert len(items)==16 and all(item['content'].startswith('历史：') for item in items)


def test_endpoint_respects_reference_budget_without_extra_status_fields(harness):
    client,engine,_,backend,*_=harness
    engine.context_builder.max_reference_chars=100
    backend.search_results=[{'id':'big','memory':'x'*1000}]
    body=turn(client,'t1').json()
    assert body['long_term_memory']==[]
    assert set(body)=={'turn_id','current_user','long_term_memory','short_term_memory'}


def test_legacy_prefix_and_single_id_are_independent(harness):
    client,_,store,*_=harness
    legacy=client.post('/v1/memory/context',headers=dict(AUTH,**{'Idempotency-Key':'turn:t1'}),
        json={'user_id':'A','query':'same','previous_ai_reply':None})
    assert legacy.status_code==200 and 'messages' in legacy.json()
    single=turn(client,'t1',query='same')
    assert single.status_code==200 and set(single.json())=={
        'turn_id','current_user','long_term_memory','short_term_memory'}
    assert len(store.list_shared_messages('bot'))==2


def test_internal_separator_is_not_valid_external_id(harness):
    client,engine,*_=harness
    assert turn(client,TURN_REQUEST_PREFIX+'t1').status_code==422
    with pytest.raises(IdentityRequiredError):
        engine.process_context(user_id='A',agent_id='bot',query='q',previous_ai_reply=None,
                               idempotency_key=TURN_REQUEST_PREFIX+'x')


def test_wrong_cached_response_type_is_rejected(harness):
    client,_,store,*_=harness
    assert turn(client,'t1').status_code==200
    with store._connect(write=True) as conn:
        conn.execute('UPDATE context_requests SET response_json=?',
                     (json.dumps({'turn_id':'t1','current_user':'A','long_term_memory':'bad','short_term_memory':[]}),))
    response=turn(client,'t1')
    assert response.status_code==409 and response.json()['detail']['code']=='request_protocol_conflict'


def _set_schema(store,version):
    with sqlite3.connect(store.path) as conn:
        conn.execute(f'PRAGMA user_version={version}')


def test_v8_string_response_rebuilds_structured_without_duplicate(tmp_path):
    engine,store,backend,*_=make_engine(tmp_path)
    result=engine.process_turn_context(turn_id='t1',user_id='A',agent_id='bot',query='original',previous_ai_reply=None)
    assert set(result)=={'turn_id','current_user','long_term_memory','short_term_memory'}
    with store._connect(write=True) as conn:
        row=conn.execute("SELECT request_id,turn_id FROM context_requests WHERE request_kind='single'").fetchone()
        conn.execute("UPDATE context_requests SET response_json=?,state='ready' WHERE request_id=?",
                     (json.dumps({'turn_id':'t1','prompt':'old string'}),row['request_id']))
        conn.execute("UPDATE interaction_turns SET prompt_json=? WHERE turn_id=?",
                     (json.dumps({'prompt_text':'old string'}),row['turn_id']))
    engine.close();_set_schema(store,8)
    new,upgraded,_,*_=make_engine(tmp_path,long_term=backend)
    rebuilt=new.process_turn_context(turn_id='t1',user_id='A',agent_id='bot',query='original',previous_ai_reply=None)
    assert set(rebuilt)=={'turn_id','current_user','long_term_memory','short_term_memory'}
    assert len(upgraded.list_shared_messages('bot'))==1
    with upgraded._connect() as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0]==11
    new.close()


def test_v9_migration_failure_rolls_back(tmp_path,monkeypatch):
    from bumi_mem.session_store import SQLiteSessionStore
    engine,store,backend,*_=make_engine(tmp_path)
    engine.process_turn_context(turn_id='t1',user_id='A',agent_id='bot',query='source',previous_ai_reply=None)
    with store._connect(write=True) as conn:
        row=conn.execute("SELECT request_id,turn_id FROM context_requests WHERE request_kind='single'").fetchone()
        conn.execute("UPDATE context_requests SET response_json=? WHERE request_id=?",
                     (json.dumps({'turn_id':'t1','prompt':'old'}),row['request_id']))
    engine.close();_set_schema(store,8)
    monkeypatch.setattr(SQLiteSessionStore,'_bump_context_epoch',staticmethod(
        lambda *a: (_ for _ in ()).throw(RuntimeError('migration failure'))))
    with pytest.raises(RuntimeError,match='migration failure'):
        SQLiteSessionStore(store.path)
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0]==8
        assert json.loads(conn.execute('SELECT response_json FROM context_requests').fetchone()[0])['prompt']=='old'
        assert conn.execute('SELECT content FROM messages').fetchall()==[('source',)]


def test_concurrent_retries_write_exactly_once(harness):
    client,_,store,backend,*_=harness
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:turn(client,'concurrent',query='same').json(),range(4)))
    assert all(result==results[0] for result in results)
    assert len(store.list_shared_messages('bot'))==1 and len(backend.search_calls)==1


def test_settle_only_is_structured_empty_response(harness):
    client,_,store,*_=harness
    assert turn(client,'t1',query='hello').status_code==200
    first=turn(client,'flush',query=None,previous='bye')
    replay=turn(client,'flush',query=None,previous='bye')
    expected={'turn_id':'flush','current_user':'A','long_term_memory':[],'short_term_memory':[]}
    assert first.status_code==200 and replay.json()==first.json()==expected
    assert [m.content for m in store.list_shared_messages('bot')]==['hello','bye']
