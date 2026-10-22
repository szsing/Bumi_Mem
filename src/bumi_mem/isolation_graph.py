# 本次改动（2026-10-21）：实现人物账号绑定、共同事实参与者检索、版本校验、反馈、编辑和来源撤回。
"""Owner-private Graphiti facts plus explicitly sourced joint assertions."""
import json
from uuid import uuid4
from bumi_mem.shared_facts import joint_assertions, unique_names


class IsolationGraphMixin:
    async def isolation_prepare(self):
        for label in ('BumiRegisteredIdentity','BumiSharedFact','BumiSharedFeedback','BumiControlEvidence'):
            await self._admin_query(f'CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE')
        await self._admin_query('CREATE INDEX bumi_shared_agent IF NOT EXISTS FOR (f:BumiSharedFact) ON (f.agent_id)')

    async def isolation_index(self, *, user_id, agent_id, messages, users):
        active={u['user_id'] for u in users if u.get('active',True)}
        if user_id not in active:return
        assertions=joint_assertions(messages,owner_id=user_id,users=users)
        names=unique_names(users)
        # Entity UUIDs remain Graphiti-scoped; the account binding is separately managed.
        await self._admin_query('''UNWIND $names AS n
            MATCH (e:Entity {group_id:$g,name:n.name})
            WITH n,collect(e) AS entities WHERE size(entities)=1
            WITH n,head(entities) AS e
            MERGE (i:BumiRegisteredIdentity {key:$agent+'::'+n.user_id})
            SET i.agent_id=$agent,i.user_id=n.user_id
            MERGE (i)-[:IDENTIFIES]->(e) RETURN count(e) AS n''',
            names=[dict(name=n,user_id=uid) for n,uid in names.items()],g=self._isolation_group(user_id,agent_id),agent=agent_id)
        for fact in assertions:
            embedding=await self.client.embedder.create(fact['content'])
            await self._admin_query('''MERGE (f:BumiSharedFact {key:$agent+'::'+$id})
                ON CREATE SET f.id=$id,f.agent_id=$agent,f.owner_id=$owner,f.participants=$participants,
                    f.content=$content,f.source_id=$source,f.original_source_id=$source,f.observed_at=$observed,f.kind=$kind,
                    f.embedding=$embedding,f.version=0,f.deleted=false
                WITH f UNWIND f.participants AS participant
                MERGE (i:BumiRegisteredIdentity {key:$agent+'::'+participant})
                SET i.agent_id=$agent,i.user_id=participant
                MERGE (i)-[:PARTICIPATES_IN]->(f) RETURN f.id AS id''',
                agent=agent_id,id=fact['id'],owner=user_id,participants=fact['participants'],content=fact['content'],
                source=fact['source_id'],observed=fact['observed_at'],kind=fact['kind'],embedding=embedding)

    @staticmethod
    def _isolation_group(user,agent):
        from bumi_mem.graphiti_memory import graph_group
        return graph_group(user,agent)

    async def isolation_shared_search(self, *, user_id, agent_id, query, users, query_vector=None):
        active={u['user_id'] for u in users if u.get('active',True)}
        if user_id not in active:return []
        groups=[dict(owner=owner,group=self._isolation_group(owner,agent_id)) for owner in active]
        vector=query_vector if query_vector is not None else await self.client.embedder.create(query)
        facts=await self._admin_query('''MATCH (f:BumiSharedFact {agent_id:$agent})
            WHERE $viewer IN f.participants AND f.owner_id IN $active AND NOT f.deleted
                AND all(p IN f.participants WHERE p IN $active)
            OPTIONAL MATCH (h:BumiSharedFeedback {key:f.key+'::'+$viewer})
            WITH f,h WHERE NOT coalesce(h.hidden,false)
            UNWIND $groups AS scope WITH f,h,scope WHERE scope.owner=f.owner_id
            OPTIONAL MATCH (b:BumiControlBlock {key:scope.group+'::'+f.source_id})
            WITH f,b WHERE b IS NULL
            RETURN f.id AS id,f.content AS content,f.owner_id AS owner_id,f.participants AS participants,
                f.source_id AS source_id,f.observed_at AS observed_at,f.version AS version,f.kind AS kind,
                EXISTS { MATCH (d:BumiSharedFeedback {agent_id:$agent,memory_id:f.id}) WHERE d.disputed=true } AS disputed,
                vector.similarity.cosine(f.embedding,$vector) AS score ORDER BY score DESC,f.id LIMIT 10''',
            agent=agent_id,viewer=user_id,active=sorted(active),groups=groups,vector=vector)
        return [dict(id='shared:'+f['id'],user_id=user_id,agent_id=agent_id,
            memory=(' [存在异议]' if f.get('disputed') else '')+'[单方陈述，来源用户 '+f['owner_id']+'] '+f['content'],fact_status='unconfirmed_joint_assertion',
            metadata=dict(backend='graphiti',kind='edge',shared_fact=True,owner_id=f['owner_id'],
                participants=f['participants'],source_message_ids=[f['source_id']],observed_at=f['observed_at'],version=f['version'],disputed=f.get('disputed',False))) for f in facts]

    async def isolation_shared_list(self, *, user_id, agent_id, users):
        active=[u['user_id'] for u in users if u.get('active',True)]
        if user_id not in active:return []
        groups=[dict(owner=u,group=self._isolation_group(u,agent_id)) for u in active]
        return await self._admin_query('''MATCH (f:BumiSharedFact {agent_id:$agent})
            WHERE $user IN f.participants AND f.owner_id IN $active AND NOT f.deleted
                AND all(p IN f.participants WHERE p IN $active)
            UNWIND $groups AS scope WITH f,scope WHERE scope.owner=f.owner_id
            OPTIONAL MATCH (b:BumiControlBlock {key:scope.group+'::'+f.source_id})
            WITH f,b WHERE b IS NULL
            OPTIONAL MATCH (h:BumiSharedFeedback {key:f.key+'::'+$user})
            RETURN f.id AS id,f.content AS content,f.owner_id AS owner_id,f.participants AS participants,
                f.version AS version,f.observed_at AS observed_at,coalesce(h.hidden,false) AS hidden,
                coalesce(h.disputed,false) AS disputed ORDER BY f.id LIMIT 100''',
            agent=agent_id,user=user_id,active=active,groups=groups)

    async def isolation_shared_detail(self, *, user_id, agent_id, memory_id, administrator=False):
        rows=await self._admin_query('''MATCH (f:BumiSharedFact {key:$key}) WHERE $admin OR $user IN f.participants
            RETURN f.id AS id,f.content AS content,f.owner_id AS owner_id,f.participants AS participants,
                f.version AS version,f.deleted AS deleted,f.source_id AS source_id,coalesce(f.original_source_id,f.source_id) AS original_source_id''',key=agent_id+'::'+memory_id,user=user_id,admin=administrator)
        if not rows:raise LookupError('shared_fact_not_found')
        return rows[0]

    async def isolation_shared_change(self, *, user_id, agent_id, memory_id, version, action, content=None, administrator=False, registered_users=None):
        if action not in ('withdraw','edit','hide','unhide','dispute'):raise ValueError('invalid_action')
        async with self._write_lock:
            old=await self.isolation_shared_detail(user_id=user_id,agent_id=agent_id,memory_id=memory_id,administrator=administrator)
            if old['version']!=version:raise ValueError('version_conflict')
            if action in ('withdraw','edit') and old['owner_id']!=user_id and not administrator:raise PermissionError('owner_required')
            if action=='edit' and (not content or not content.strip() or len(content)>1000):raise ValueError('invalid_content')
            if action=='edit' and not administrator:
                source=dict(user_id=old['owner_id'],source_message_id='edit-validation',original_content=content.strip())
                candidates=joint_assertions([dict(role='user',content='[source '+json.dumps(source,ensure_ascii=False)+']')],
                    owner_id=old['owner_id'],users=registered_users or [])
                if len(candidates)!=1 or set(candidates[0]['participants'])!=set(old['participants']):
                    raise ValueError('invalid_shared_content')
            embedding=await self.client.embedder.create(content.strip()) if action=='edit' else None
            rows=await self._admin_query('''MATCH (f:BumiSharedFact {key:$key})
                SET f.lock=coalesce(f.lock,0)+1 WITH f WHERE f.version=$version AND ($admin OR $user IN f.participants)
                AND ($admin OR $action IN ['hide','unhide','dispute'] OR f.owner_id=$user)
                FOREACH (sid IN CASE WHEN $action IN ['withdraw','edit'] THEN $sources ELSE [] END |
                    MERGE (b:BumiControlBlock {key:$group+'::'+sid}) SET b.group_id=$group,b.source_id=sid)
                FOREACH (_ IN CASE WHEN $action='withdraw' THEN [1] ELSE [] END | SET f.deleted=true,f.version=f.version+1)
                FOREACH (_ IN CASE WHEN $action='edit' THEN [1] ELSE [] END |
                    SET f.content=$content,f.embedding=$embedding,f.source_id=$revision,f.version=f.version+1)
                MERGE (h:BumiSharedFeedback {key:$key+'::'+$user})
                SET h.agent_id=$agent,h.user_id=$user,h.memory_id=f.id
                FOREACH (_ IN CASE WHEN $action IN ['hide','unhide'] THEN [1] ELSE [] END | SET h.hidden=$action='hide')
                FOREACH (_ IN CASE WHEN $action='dispute' THEN [1] ELSE [] END | SET h.disputed=true)
                CREATE (a:BumiSharedAudit {agent_id:$agent,memory_id:f.id,actor_id:$user,
                    action:$action,version:f.version,at:datetime(),before:$before,after:f.content})
                RETURN f.id AS id,f.version AS version,f.deleted AS deleted,h.hidden AS hidden,h.disputed AS disputed,$sources AS withdrawn_sources,$owner AS owner_id''',
                key=agent_id+'::'+memory_id,user=user_id,agent=agent_id,version=version,action=action,
                content=content.strip() if content else None,embedding=embedding,before=old['content'],admin=administrator,
                sources=sorted({old['source_id'],old['original_source_id']}),owner=old['owner_id'],
                group=self._isolation_group(old['owner_id'],agent_id),revision='manual:'+uuid4().hex)
            if not rows:raise ValueError('version_conflict')
            return rows[0]
