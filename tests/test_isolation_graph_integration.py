# 本次改动（2026-10-21）：新增真实 Neo4j 临时作用域测试，覆盖参与者可见性、版本、隐藏、异议和来源撤回。
"""Explicit opt-in Neo4j authorization contracts, no remote model calls."""
import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4
import pytest
from bumi_mem.graphiti_memory import GraphitiLongTermMemory
from test_registered_isolation import USERS,evidence

@pytest.mark.skipif(os.getenv('BUMI_RUN_NEO4J_ISOLATION_TEST')!='1',reason='requires explicitly configured Neo4j')
def test_shared_visibility_owner_feedback_and_withdrawal():
    from neo4j import AsyncGraphDatabase
    async def run():
        driver=AsyncGraphDatabase.driver(os.environ.get('BUMI_NEO4J_URI','bolt://127.0.0.1:7687'),
            auth=(os.environ.get('BUMI_NEO4J_USER','neo4j'),os.environ['BUMI_NEO4J_PASSWORD']))
        async def embedding(text):return [1.0,0.2,0.1]
        memory=GraphitiLongTermMemory.__new__(GraphitiLongTermMemory)
        memory.client=SimpleNamespace(driver=driver,embedder=SimpleNamespace(create=embedding))
        memory._write_lock=asyncio.Lock()
        agent='isolation_verify_'+uuid4().hex
        group=memory._isolation_group('user2',agent)
        try:
            await memory.isolation_prepare()
            await memory._admin_query('''CREATE (a:Entity {uuid:$a,group_id:$g,name:'小王'}),
                (b:Entity {uuid:$b,group_id:$g,name:'小李'}) RETURN a.uuid AS id''',
                a=uuid4().hex,b=uuid4().hex,g=group)
            # Only the exact explicit joint clause is published, not the private next sentence.
            messages=[evidence('我和小王昨天一起去了图书馆。小王已经退学了。'),
                      evidence('我昨天去了博物馆',mid='private')]
            await memory.isolation_index(user_id='user2',agent_id=agent,messages=messages,users=USERS)
            await memory.isolation_index(user_id='user2',agent_id=agent,messages=messages,users=USERS)
            first=await memory.isolation_shared_search(user_id='user1',agent_id=agent,query='一起去了哪里',users=USERS)
            assert len(first)==1
            assert '图书馆' in first[0]['memory'] and '博物馆' not in first[0]['memory'] and '退学' not in first[0]['memory']
            assert first[0]['metadata']['owner_id']=='user2'
            assert await memory.isolation_shared_search(user_id='__guest__',agent_id=agent,query='图书馆',users=USERS)==[]
            assert await memory.isolation_shared_search(user_id='outsider',agent_id=agent,query='图书馆',users=USERS)==[]
            mid=first[0]['id'].split(':',1)[1]
            bindings=await memory._admin_query('MATCH (i:BumiRegisteredIdentity {agent_id:$a})-[:IDENTIFIES]->(e) RETURN i.user_id AS user',a=agent)
            assert {r['user'] for r in bindings}=={'user1','user2'}
            with pytest.raises(PermissionError,match='owner_required'):
                await memory.isolation_shared_change(user_id='user1',agent_id=agent,memory_id=mid,version=0,action='withdraw')
            await memory.isolation_shared_change(user_id='user1',agent_id=agent,memory_id=mid,version=0,action='hide')
            assert await memory.isolation_shared_search(user_id='user1',agent_id=agent,query='图书馆',users=USERS)==[]
            assert len(await memory.isolation_shared_search(user_id='user2',agent_id=agent,query='图书馆',users=USERS))==1
            await memory.isolation_shared_change(user_id='user1',agent_id=agent,memory_id=mid,version=0,action='unhide')
            await memory.isolation_shared_change(user_id='user1',agent_id=agent,memory_id=mid,version=0,action='dispute')
            listed=await memory.isolation_shared_list(user_id='user1',agent_id=agent,users=USERS)
            assert listed[0]['disputed'] and not listed[0]['hidden']
            with pytest.raises(ValueError,match='invalid_shared_content'):
                await memory.isolation_shared_change(user_id='user2',agent_id=agent,memory_id=mid,version=0,action='edit',content='小王退学了',registered_users=USERS)
            await memory.isolation_shared_change(user_id='user2',agent_id=agent,memory_id=mid,version=0,action='edit',content='我和小王前天一起去了图书馆',registered_users=USERS)
            with pytest.raises(ValueError,match='version_conflict'):
                await memory.isolation_shared_change(user_id='user2',agent_id=agent,memory_id=mid,version=0,action='withdraw')
            await memory.isolation_shared_change(user_id='user2',agent_id=agent,memory_id=mid,version=1,action='withdraw')
            await memory.isolation_index(user_id='user2',agent_id=agent,messages=messages,users=USERS)
            assert await memory.isolation_shared_search(user_id='user1',agent_id=agent,query='图书馆',users=USERS)==[]
            # Independent testimony survives the other owner's withdrawal.
            other=[evidence('我和小李是朋友',owner='user1',mid='independent')]
            await memory.isolation_index(user_id='user1',agent_id=agent,messages=other,users=USERS)
            assert len(await memory.isolation_shared_search(user_id='user2',agent_id=agent,query='朋友',users=USERS))==1
            from bumi_mem.graphiti_memory import graph_group
            await memory._admin_query('''CREATE (b:BumiControlBlock {key:$key,group_id:$g,source_id:'independent'}) RETURN b.key AS key''',
                key=graph_group('user1',agent)+'::independent',g=graph_group('user1',agent))
            assert await memory.isolation_shared_search(user_id='user2',agent_id=agent,query='朋友',users=USERS)==[]
        finally:
            await memory._admin_query('MATCH (n {agent_id:$a}) DETACH DELETE n',a=agent)
            for user in ('user1','user2'):
                await memory._admin_query('MATCH (n {group_id:$g}) DETACH DELETE n',g=memory._isolation_group(user,agent))
            await driver.close()
    asyncio.run(run())
