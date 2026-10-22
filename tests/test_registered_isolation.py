# 本次改动（2026-10-21）：新增规则、注册、未知身份、共享短期与私有长期、缓存失效及 API 权限测试。
"""Registration boundaries, shared assertions and shared L1/private L3 contracts."""
import json
import time
from dataclasses import replace
from types import SimpleNamespace
import asyncio
import pytest
from bumi_mem.schemas import MessageRecord
from bumi_mem.shared_facts import joint_assertions
from bumi_mem.agent_protocol import AgentPrepareRequest
from test_graphiti_runtime import graph_engine, adapter
from test_agent_api import harness, AUTH

USERS=[dict(user_id='user1',display_name='小王',aliases=['王同学']),dict(user_id='user2',display_name='小李',aliases=[])]

def evidence(text,owner='user2',mid='source'):
    return MessageRecord(message_id=mid,session_id='s',sequence_no=1,role='user',content=text,
        token_count=1,context_only=False,user_id=owner,agent_id='bot',created_at='2026-10-21T10:00:00+08:00').to_extraction_message()

@pytest.mark.parametrize('text', ['我和小王是朋友','我和user1昨天一起去了图书馆','我与王同学是大学同学','我和小王在同一所大学上学'])
def test_explicit_shared_relationship_or_event(text):
    rows=joint_assertions([evidence(text)],owner_id='user2',users=USERS)
    assert len(rows)==1 and rows[0]['participants']==['user1','user2']
    assert rows[0]['owner_id']=='user2' and rows[0]['source_id']=='source'

@pytest.mark.parametrize('text', ['小王昨天去了博物馆','我们昨天一起去了图书馆','小王和小李在同一所大学',
    '如果我和小王是朋友','听说我和小王是朋友','我和小王是朋友？','他说“我和小王是朋友”','我和小王不是朋友',
    '我和小王是朋友，他住在北京','我和张老师是朋友'])
def test_no_guessed_participants_or_private_clause_leak(text):
    assert joint_assertions([evidence(text)],owner_id='user2',users=USERS)==[]


def test_ambiguous_alias_and_guest_cannot_create_shared_assertions():
    users=USERS+[dict(user_id='user3',display_name='小王')]
    assert joint_assertions([evidence('我和小王是朋友')],owner_id='user2',users=users)==[]
    assert joint_assertions([evidence('我和小王是朋友',owner='__guest__')],owner_id='__guest__',users=USERS)==[]


def isolated_engine(path):
    engine,store,memory,chat,summary=graph_engine(path)
    store.enable_isolation()
    original_search = memory.search
    memory.search = lambda **kw: original_search(**{k:v for k,v in kw.items() if k not in ('registered_isolation','registered_users')})
    for u in USERS:store.register_user(agent_id='bot',**u)
    return engine,store,memory,chat,summary


def test_guest_ids_are_collapsed_before_sqlite_and_have_no_long_term(tmp_path):
    engine,store,memory,*_=isolated_engine(tmp_path)
    first=engine.process_turn_context(turn_id='one',user_id='random-unregistered-one',agent_id='bot',query='你好')
    assert first['current_user']=='__guest__' and first['long_term_memory']==[]
    second=engine.process_turn_context(turn_id='two',user_id='random-unregistered-two',agent_id='bot',query='我来了',previous_ai_reply='欢迎')
    assert second['current_user']=='__guest__'
    assert len(second['short_term_memory'])==3
    with store._connect() as conn:
        assert {r[0] for r in conn.execute('SELECT user_id FROM sessions')}=={'__guest__'}
        for table in ('sessions','messages','context_requests','interaction_turns'):
            assert 'random-unregistered' not in json.dumps([tuple(r) for r in conn.execute('SELECT * FROM '+table)])
    assert memory.search_calls==[]
    assert store.submit_idle_snapshots(settings=engine.consolidator._settings(),now=time.time()+121)==0


def test_switch_retains_l1_but_not_other_users_long_term_refs(tmp_path):
    engine,store,memory,*_=isolated_engine(tmp_path)
    engine.remember(user_id='user1',agent_id='bot',contents=['USER1_PRIVATE_FACT'])
    first=engine.process_turn_context(turn_id='a',user_id='user1',agent_id='bot',query='你好')
    assert 'USER1_PRIVATE_FACT' in json.dumps(first)
    second=engine.process_turn_context(turn_id='b',user_id='user2',agent_id='bot',query='我来了',previous_ai_reply='共享现场回复')
    assert 'USER1_PRIVATE_FACT' not in json.dumps(second)
    assert any(r['content']=='你好' for r in second['short_term_memory'])
    assert any(r['content']=='共享现场回复' for r in second['short_term_memory'])
    guest=engine.process_turn_context(turn_id='c',user_id='unknown',agent_id='bot',query='你好')
    assert guest['long_term_memory']==[] and len(guest['short_term_memory'])>1


def test_registration_and_aliases_persist_and_can_disable(tmp_path):
    engine,store,*_=isolated_engine(tmp_path)
    store.register_user(agent_id='bot',user_id='user1',display_name='小王',active=False)
    assert store.resolve_registered_user('user1','bot')=='__guest__'
    engine.close()
    from bumi_mem.session_store import SQLiteSessionStore
    restored=SQLiteSessionStore(tmp_path/'bumi.sqlite3')
    assert restored.isolation_enabled()
    assert restored.resolve_registered_user('user2','bot')=='user2'
    assert restored.resolve_registered_user('user1','bot')=='__guest__'


def test_prepare_protocol_also_normalizes_unknown_identity(tmp_path):
    engine,store,memory,*_=isolated_engine(tmp_path)
    request=AgentPrepareRequest(operation='prepare',client_turn_id='a',query='hello',user_id='unregistered-user')
    result=engine.prepare_agent(request=request,agent_id='bot')
    assert result['user_id']=='__guest__'
    with store._connect() as conn:
        assert 'unregistered-user' not in json.dumps([tuple(r) for r in conn.execute('SELECT * FROM agent_turn_requests')])


def test_admin_registration_requires_authentication_and_shared_guest_cannot_write(harness):
    client,engine,store,*_=harness
    store.enable_isolation()
    assert client.get('/v2/admin/users').status_code==401
    assert client.put('/v2/admin/users',headers=AUTH,json=dict(user_id='user1',display_name='小王')).status_code==200
    assert client.put('/v2/admin/users',headers=AUTH,json=dict(user_id='__guest__',display_name='陌生人')).status_code==422
    assert client.get('/v2/memory/shared?user_id=unknown',headers=AUTH).json()=={'items':[]}
    assert client.patch('/v2/memory/shared/anything',headers=AUTH,json=dict(user_id='unknown',version=0,action='withdraw')).status_code==403


def test_isolated_graph_search_uses_edges_only_and_retains_private_scope(tmp_path):
    from bumi_mem.graphiti_memory import graph_group
    group=graph_group('user1','bot')
    class Item:
        def __init__(self,mid,text,scope=group):self.uuid=mid;self.text=text;self.group_id=scope
        def model_dump(self,**kwargs):return dict(fact=self.text)
    class Client:
        async def search_(self,query,*,group_ids,config,query_vector):
            assert query_vector==[1.0,0.0]
            assert group_ids==[group]
            assert config.edge_config is not None
            assert config.node_config is config.episode_config is config.community_config is None
            return SimpleNamespace(edges=[Item('mine','自己的事实'),Item('other','朋友的私人行程',graph_group('user2','bot'))],
                nodes=[Item('node','朋友完整摘要')],episodes=[Item('ep','别人的原话')],communities=[Item('community','其他人的综合信息')])
    client=Client()
    async def embedding(text):return [1.0,0.0]
    client.embedder=SimpleNamespace(create=embedding)
    memory=adapter(tmp_path,client)
    async def shared(**kwargs):return []
    memory.isolation_shared_search=shared
    rows=asyncio.run(memory.search(user_id='user1',agent_id='bot',query='test',registered_isolation=True,registered_users=USERS))
    assert [r['id'] for r in rows]==['edge:mine']
    assert asyncio.run(memory.search(user_id='unknown',agent_id='bot',query='test',registered_isolation=True,registered_users=USERS))==[]


def test_unregistered_user_cannot_use_mutation_entry_points(tmp_path):
    engine,store,*_=isolated_engine(tmp_path)
    from bumi_mem.schemas import IdentityRequiredError
    with pytest.raises(IdentityRequiredError):engine.forget_all(user_id='unknown',agent_id='bot')
    with pytest.raises(IdentityRequiredError):engine.remember(user_id='unknown',agent_id='bot',contents=['禁止写入'])


def test_matched_but_unregistered_speaker_is_not_persisted(tmp_path):
    engine,store,*_=isolated_engine(tmp_path)
    request=AgentPrepareRequest(operation='prepare',client_turn_id='a',query='hello',
        speaker=dict(status='matched',profile_id='unregistered-profile',label='UnknownPerson'))
    result=engine.prepare_agent(request=request,agent_id='bot')
    assert result['user_id']=='__guest__'
    with store._connect() as conn:
        for table in ('sessions','messages','agent_turn_requests'):
            data=json.dumps([tuple(r) for r in conn.execute('SELECT * FROM '+table)])
            assert 'unregistered-profile' not in data and 'UnknownPerson' not in data


def test_unregistering_owner_stops_queued_legacy_extraction(tmp_path):
    engine,store,memory,*_=graph_engine(tmp_path)
    session=store.get_or_create_session('old-user','bot')
    store.append_message(session.session_id,role='user',content='旧排队来源',token_count=1)
    store.submit(session.session_id,reason='test',settings=engine.consolidator._settings(),carryover_messages=0,max_attempts=3)
    store.enable_isolation()
    assert engine.worker.run_once()
    assert memory.add_calls==[]
    assert store.list_jobs(user_id='old-user',agent_id='bot')[0].status=='failed'


def test_registry_change_invalidates_cached_context_and_prefetch(tmp_path):
    engine,store,memory,*_=isolated_engine(tmp_path)
    engine.process_turn_context(turn_id='a',user_id='user1',agent_id='bot',query='hello')
    epoch=store.context_epoch('bot')
    store.register_user(agent_id='bot',user_id='user2',display_name='小李',active=False)
    assert store.context_epoch('bot')>epoch
    from bumi_mem.turn_store import TurnConflict
    with pytest.raises(TurnConflict):
        engine.process_turn_context(turn_id='a',user_id='user1',agent_id='bot',query='hello')
