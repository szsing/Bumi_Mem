"""Administrator API and override projection; no real user mutations."""
import asyncio
import types
import pytest
from test_agent_api import harness,AUTH
from test_graphiti_runtime import adapter
from bumi_mem.graphiti_memory import graph_group


def test_admin_page_and_authentication(harness):
    client,engine,*_=harness
    page=client.get('/admin')
    assert page.status_code==200 and '记忆管理' in page.text
    assert client.get('/v1/admin/groups').status_code==401
    assert client.get('/v1/admin/groups',headers=AUTH).status_code==409


def test_admin_routes_validate_versions_and_pass_scope(harness):
    client,engine,*_=harness
    engine.long_term.backend_name='graphiti'
    calls=[]
    def call(method,kwargs,**options):
        calls.append((method,kwargs))
        if method=='admin_change' and kwargs['version']!=3:raise ValueError('version_conflict')
        if method=='admin_groups':return [{'group_id':'g'}]
        return {'version':4,'content':'修订'}
    engine.long_term._call=call
    assert client.get('/v1/admin/groups',headers=AUTH).json()['items'][0]['group_id']=='g'
    assert client.get('/v1/admin/memories?limit=101',headers=AUTH).status_code==422
    assert client.patch('/v1/admin/memories/x',headers=AUTH,json={'group_id':'g','version':0,'action':'edit','content':'新内容'}).status_code==409
    response=client.patch('/v1/admin/memories/x',headers=AUTH,json={'group_id':'g','version':3,'action':'edit','content':'新内容'})
    assert response.status_code==200 and calls[-1][1]['memory_id']=='x'


def test_admin_override_hides_deleted_facts_and_stale_summaries(tmp_path):
    group=graph_group('u','a')
    class Item:
        group_id=group
        def __init__(self,id,text):self.uuid=id;self.text=text
        def model_dump(self,**kw):return {'fact':self.text}
    class Client:
        async def search_(self,*a,**kw):
            return types.SimpleNamespace(edges=[Item('edited','旧内容'),Item('deleted','应删除'),Item('keep','保留事实')],nodes=[Item('n','旧摘要')],episodes=[Item('p','原始旧对话')],communities=[Item('c','旧社区')])
    memory=adapter(tmp_path,Client())
    async def overrides(g):
        return [{'id':'edited','content':'新内容','deleted':False,'version':1},
                {'id':'deleted','content':'应删除','deleted':True,'version':1}]
    memory._admin_recall_overrides=overrides
    rows=asyncio.run(memory.search(user_id='u',agent_id='a',query='内容'))
    assert [r['id'] for r in rows]==['edge:edited','edge:keep']
    assert '新内容' in rows[0]['memory'] and not any('旧' in r['memory'] for r in rows)


def test_graph_api_preserves_group_and_validates_limit(harness):
    client,engine,*_=harness
    engine.long_term.backend_name='graphiti'
    calls=[]
    def call(method,kwargs,**options):
        calls.append((method,kwargs))
        return {'nodes':[],'edges':[],'truncated':False,'limit':kwargs['limit']}
    engine.long_term._call=call
    assert client.get('/v1/admin/graph').status_code==401
    assert client.get('/v1/admin/graph?limit=801',headers=AUTH).status_code==422
    result=client.get('/v1/admin/graph?group_id=example&limit=200',headers=AUTH)
    assert result.status_code==200
    assert calls[-1]==('admin_graph',{'group_id':'example','limit':200})
