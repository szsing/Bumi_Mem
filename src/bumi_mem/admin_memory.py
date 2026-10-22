"""Administrator fact management. Overrides survive re-extraction of the same edge."""
import json
import uuid
from datetime import datetime, timezone


def plain(value):
    if isinstance(value, dict):return {k:plain(v) for k,v in value.items() if 'embedding' not in k}
    if isinstance(value, (list, tuple)):return [plain(v) for v in value]
    if hasattr(value,'iso_format'):return value.iso_format()
    return value


class AdminMemoryMixin:
    async def _admin_query(self, cypher, **params):
        records,_,_=await self.client.driver.execute_query(cypher,**params)
        return [plain(dict(r)) for r in records]

    async def admin_groups(self):
        groups=await self._admin_query('''MATCH (n:Entity) WITH n.group_id AS group_id, count(*) AS entities
            WHERE group_id IS NOT NULL
            OPTIONAL MATCH ()-[r:RELATES_TO {group_id:group_id}]->()
            WITH group_id,entities,count(r) AS facts
            OPTIONAL MATCH (e:Episodic {group_id:group_id})
            RETURN group_id,entities,facts,head(collect(e.content)) AS sample ORDER BY group_id''')
        for g in groups:
            # Machine source JSON carries stable owner IDs. Never infer identity from prose.
            sample=g.pop('sample',None) or ''
            g['user_id']=None;g['agent_id']=None
            for line in sample.splitlines():
                marker='[source '
                if marker not in line:continue
                try:
                    data,_=json.JSONDecoder().raw_decode(line.split(marker,1)[1])
                    if data.get('user_id'):
                        g['user_id']=data['user_id'];g['agent_id']=data.get('agent_id');break
                except (ValueError,TypeError):continue
        return groups

    async def admin_list(self, *, group_id=None, query='', status='active', offset=0, limit=30):
        where='''($group_id IS NULL OR r.group_id=$group_id)
          AND ($query='' OR toLower(coalesce(o.content,r.fact,'')) CONTAINS toLower($query))
          AND ($status='all' OR ($status='deleted' AND coalesce(o.deleted,r.admin_deleted,false))
               OR ($status='active' AND NOT coalesce(o.deleted,r.admin_deleted,false)))'''
        prefix='''MATCH (a)-[r:RELATES_TO]->(b)
            OPTIONAL MATCH (o:BumiAdminMemory {key:r.group_id+'::'+r.uuid}) WITH a,r,b,o WHERE '''+where
        args=dict(group_id=group_id,query=query,status=status)
        total=await self._admin_query(prefix+' RETURN count(r) AS total',**args)
        rows=await self._admin_query(prefix+''' RETURN r.uuid AS id,r.group_id AS group_id,
            coalesce(o.content,r.fact) AS content,a.name AS source,b.name AS target,
            r.name AS relation,r.valid_at AS valid_at,r.invalid_at AS invalid_at,
            coalesce(o.version,r.admin_version,0) AS version,
            coalesce(o.deleted,r.admin_deleted,false) AS deleted,
            o.updated_at AS edited_at ORDER BY group_id,id SKIP $offset LIMIT $limit''',**args,offset=offset,limit=limit)
        return dict(items=rows,total=total[0]['total'],offset=offset,limit=limit)

    async def admin_detail(self, *, group_id, memory_id):
        rows=await self._admin_query('''MATCH (a)-[r:RELATES_TO {group_id:$group_id,uuid:$id}]->(b)
            OPTIONAL MATCH (o:BumiAdminMemory {key:$group_id+'::'+$id})
            RETURN r.uuid AS id,r.group_id AS group_id,coalesce(o.content,r.fact) AS content,
              r.fact AS graph_content,a.name AS source,b.name AS target,r.name AS relation,
              r.valid_at AS valid_at,r.invalid_at AS invalid_at,r.expired_at AS expired_at,
              r.episodes AS episodes,coalesce(o.version,r.admin_version,0) AS version,
              coalesce(o.deleted,r.admin_deleted,false) AS deleted,o.updated_at AS edited_at''',group_id=group_id,id=memory_id)
        if not rows:raise LookupError('memory_not_found')
        record=rows[0]
        record['sources']=await self._admin_query('''MATCH (e:Episodic) WHERE e.group_id=$group_id AND e.uuid IN $ids
            RETURN e.uuid AS id,e.content AS content,e.valid_at AS valid_at''',group_id=group_id,ids=record.pop('episodes') or [])
        record['history']=await self._admin_query('''MATCH (h:BumiAdminRevision {group_id:$group_id,memory_id:$id})
            RETURN h.version AS version,h.action AS action,h.before AS before,h.after AS after,
              h.at AS at ORDER BY version DESC LIMIT 50''',group_id=group_id,id=memory_id)
        return record

    async def admin_change(self, *, group_id, memory_id, version, action, content=None):
        if action not in ('edit','delete','restore'):raise ValueError('invalid_action')
        if action=='edit' and (not isinstance(content,str) or not content.strip() or len(content)>8000):
            raise ValueError('invalid_content')
        async with self._write_lock:
            old=await self.admin_detail(group_id=group_id,memory_id=memory_id)
            if old['version']!=version:raise ValueError('version_conflict')
            text=content.strip() if action=='edit' else old['content']
            deleted=action=='delete' or (action=='edit' and old['deleted'])
            embedding=await self.client.embedder.create(text) if action=='edit' else None
            now=datetime.now(timezone.utc).isoformat()
            rows=await self._admin_query('''MATCH ()-[r:RELATES_TO {group_id:$group_id,uuid:$id}]->()
                SET r.admin_version=coalesce(r.admin_version,0)
                WITH r OPTIONAL MATCH (existing:BumiAdminMemory {key:$group_id+'::'+$id})
                WITH r,existing WHERE coalesce(existing.version,r.admin_version,0)=$version
                SET r.admin_version=$version+1,r.admin_deleted=$deleted,r.fact=$content
                FOREACH (_ IN CASE WHEN $embedding IS NULL THEN [] ELSE [1] END |
                    SET r.fact_embedding=$embedding)
                MERGE (o:BumiAdminMemory {key:$group_id+'::'+$id})
                SET o.group_id=$group_id,o.memory_id=$id,o.content=$content,o.deleted=$deleted,
                    o.version=$version+1,o.updated_at=$now,o.control_deleted=false,o.control_relearned=false
                CREATE (h:BumiAdminRevision {uuid:$revision,group_id:$group_id,memory_id:$id,
                    version:$version+1,action:$action,before:$before,after:$content,at:$now})
                RETURN r.uuid AS id''',group_id=group_id,id=memory_id,version=version,
                content=text,deleted=deleted,embedding=embedding,now=now,revision=uuid.uuid4().hex,
                action=action,before=old['content'])
            if not rows:raise ValueError('version_conflict')
        return await self.admin_detail(group_id=group_id,memory_id=memory_id)

    async def _admin_recall_overrides(self, group):
        return await self._admin_query('''MATCH (o:BumiAdminMemory {group_id:$group})
            RETURN o.memory_id AS id,o.content AS content,o.deleted AS deleted,
              o.version AS version,o.updated_at AS updated_at,o.control_relearned AS control_relearned,
              o.control_source_ids AS control_source_ids ORDER BY updated_at DESC''',group=group)

    async def admin_graph(self, *, group_id=None, limit=350):
        args=dict(group_id=group_id,limit=limit+1)
        nodes=await self._admin_query('''MATCH (n) WHERE ($group_id IS NULL OR n.group_id=$group_id)
            AND (n:Entity OR n:Episodic OR n:Community)
            RETURN n.uuid AS id,n.group_id AS group_id,n.name AS name,
              CASE WHEN n:Entity THEN 'entity' WHEN n:Episodic THEN 'episode' ELSE 'community' END AS kind,
              substring(coalesce(n.summary,n.content,''),0,2000) AS description,
              n.created_at AS created_at ORDER BY group_id,kind,id LIMIT $limit''',**args)
        truncated=len(nodes)>limit;nodes=nodes[:limit]
        if not nodes:return dict(nodes=[],edges=[],truncated=False,limit=limit)
        ids=[n['id'] for n in nodes]
        edges=await self._admin_query('''MATCH (a)-[r]->(b)
            WHERE a.uuid IN $ids AND b.uuid IN $ids AND a.group_id=b.group_id
              AND ($group_id IS NULL OR a.group_id=$group_id)
              AND type(r) IN ['RELATES_TO','MENTIONS','HAS_MEMBER']
            OPTIONAL MATCH (o:BumiAdminMemory {key:a.group_id+'::'+r.uuid})
            WITH a,r,b,o WHERE NOT coalesce(o.deleted,r.admin_deleted,false)
            RETURN r.uuid AS id,a.group_id AS group_id,a.uuid AS source,b.uuid AS target,
              type(r) AS kind,coalesce(r.name,type(r)) AS name,coalesce(o.content,r.fact,'') AS content,
              coalesce(o.version,r.admin_version,0) AS version
            ORDER BY group_id,kind,id LIMIT 1501''',ids=ids,group_id=group_id)
        return dict(nodes=nodes,edges=edges[:1500],truncated=truncated or len(edges)>1500,limit=limit)
