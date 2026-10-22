# 本次改动（2026-10-21）：增加新来源证据断言，验证撤回旧证据后新独立来源可以重新支持事实。
"""Opt-in Neo4j contract; writes only a unique disposable verification namespace."""
import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4
import pytest
from bumi_mem.graphiti_memory import GraphitiLongTermMemory

@pytest.mark.skipif(os.getenv('BUMI_RUN_NEO4J_CONTROL_TEST')!='1',reason='requires explicitly configured Neo4j')
def test_control_graph_atomic_deletion_and_new_evidence():
    from neo4j import AsyncGraphDatabase
    async def check():
        driver=AsyncGraphDatabase.driver(os.environ.get('BUMI_NEO4J_URI','bolt://127.0.0.1:7687'),
            auth=(os.environ.get('BUMI_NEO4J_USER','neo4j'),os.environ['BUMI_NEO4J_PASSWORD']))
        memory=GraphitiLongTermMemory.__new__(GraphitiLongTermMemory)
        memory.client=SimpleNamespace(driver=driver);memory._write_lock=asyncio.Lock()
        group='control_verify_'+uuid4().hex
        try:
            await memory._admin_query('''CREATE (a:Entity {uuid:$a,group_id:$g,name:'测试用户'}),
                (b:Entity {uuid:$b,group_id:$g,name:'茶'}),
                (e:Episodic {uuid:$e,group_id:$g,name:'batch',content:'verification'}),
                (a)-[:RELATES_TO {uuid:'fact',group_id:$g,fact:'喜欢茶',episodes:[$e]}]->(b)
                RETURN a.uuid AS id''',a=uuid4().hex,b=uuid4().hex,e=uuid4().hex,g=group)
            await memory.control_record_batch(group_id=group,batch_id='batch',source_ids=['s1','s2'])
            with pytest.raises(ValueError,match='source_scope_confirmation_required'):
                await memory.control_block_sources(group_id=group,source_ids=['s1'])
            candidate=(await memory.control_candidates(group_id=group,query='茶'))[0]
            assert candidate['affected_facts']==1 and candidate['reliable']
            receipt=await memory.control_withdraw(group_id=group,operation_id='op',candidate=candidate)
            assert receipt==await memory.control_withdraw(group_id=group,operation_id='op',candidate=candidate)
            assert await memory.control_candidates(group_id=group,query='茶')==[]
            assert await memory.control_blocks(group_id=group)=={'s1','s2'}
            assert (await memory.admin_detail(group_id=group,memory_id='fact'))['version']==1
            assert await memory.control_fresh_edges(group_id=group)==set()
            await memory.control_refresh_edges(group_id=group,edge_ids=['fact'],source_ids=['s1'])
            assert (await memory.admin_detail(group_id=group,memory_id='fact'))['deleted']
            await memory.control_refresh_edges(group_id=group,edge_ids=['fact'],source_ids=['new-source'])
            assert await memory.control_fresh_edges(group_id=group)=={'fact'}
            detail=await memory.admin_detail(group_id=group,memory_id='fact')
            assert not detail['deleted'] and detail['version']==2
            # Original blocked lineage remains immutable even if a chunk is revisited.
            await memory.control_record_batch(group_id=group,batch_id='batch',source_ids=[])
            assert len((await memory.control_suppressed(group_id=group))[0])==1
        finally:
            await memory._admin_query('MATCH (n {group_id:$g}) DETACH DELETE n',g=group)
            await driver.close()
    asyncio.run(check())
