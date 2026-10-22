# 本次改动（2026-10-21）：新增独立新来源证据账本，允许新原话重新支持事实，同时继续屏蔽已撤回旧来源。
"""Graph-side source ledger and owner-scoped, idempotent recall withdrawal."""
import hashlib
import json
import time


class ControlGraphMixin:
    async def control_blocks(self, *, group_id):
        rows = await self._admin_query('MATCH (b:BumiControlBlock {group_id:$g}) RETURN b.source_id AS id', g=group_id)
        return {r['id'] for r in rows}

    async def control_block_sources(self, *, group_id, source_ids):
        async with self._write_lock:
            batches = await self._admin_query('''MATCH (b:BumiControlBatch {group_id:$g})
                WHERE any(id IN b.source_ids WHERE id IN $ids)
                RETURN b.source_ids AS ids''',g=group_id,ids=source_ids)
            if any(set(r['ids']) - set(source_ids) for r in batches):
                raise ValueError('source_scope_confirmation_required')
            await self._admin_query('''UNWIND $ids AS sid MERGE (b:BumiControlBlock {key:$g+'::'+sid})
                SET b.group_id=$g,b.source_id=sid RETURN b.key AS key''',g=group_id,ids=source_ids)
        return {'source_ids':source_ids}

    async def control_record_batch(self, *, group_id, batch_id, source_ids):
        await self._admin_query('''MERGE (b:BumiControlBatch {key:$key})
            ON CREATE SET b.group_id=$g,b.batch_id=$batch,b.source_ids=$ids RETURN b.key AS key''',
            key=group_id+'::'+batch_id,g=group_id,batch=batch_id,ids=sorted(set(source_ids)))

    async def control_suppressed(self, *, group_id):
        blocked = await self.control_blocks(group_id=group_id)
        if not blocked:
            return set(), False
        rows = await self._admin_query('''MATCH (e:Episodic {group_id:$g})
            OPTIONAL MATCH (b:BumiControlBatch {key:$g+'::'+e.name})
            RETURN e.uuid AS id,b.source_ids AS sources''', g=group_id)
        return {r['id'] for r in rows if r['sources'] and set(r['sources']) & blocked}, True

    async def control_candidates(self, *, group_id, query):
        if hasattr(self,'journal'):
            with self._db() as db:
                batches = db.execute("SELECT id,payload FROM ingestions WHERE group_id=? AND state='complete'", (group_id,)).fetchall()
            ledger = []
            for key,payload in batches:
                messages = json.loads(payload).get('messages',[])
                ids = sorted({m['source_message_id'] for m in messages if m.get('source_message_id')})
                if ids:
                    ledger.append(dict(key=group_id+'::'+key,batch=key,ids=ids))
            if ledger:
                await self._admin_query('''UNWIND $rows AS row MERGE (b:BumiControlBatch {key:row.key})
                    ON CREATE SET b.group_id=$g,b.batch_id=row.batch,b.source_ids=row.ids RETURN count(b) AS n''',g=group_id,rows=ledger)
        result = await self.admin_list(group_id=group_id,query=query,limit=10)
        items = []
        blocked = await self.control_blocks(group_id=group_id)
        for row in result['items']:
            detail = await self.admin_detail(group_id=group_id,memory_id=row['id'])
            lineage = await self._admin_query('''MATCH (e:Episodic {group_id:$g}) WHERE e.uuid IN $episodes
                OPTIONAL MATCH (b:BumiControlBatch {key:$g+'::'+e.name})
                RETURN e.uuid AS episode,b.source_ids AS ids''',
                g=group_id,episodes=[s['id'] for s in detail['sources']])
            source_ids = sorted({mid for r in lineage for mid in (r['ids'] or [])})
            if source_ids and set(source_ids) <= blocked:
                continue
            reliable = bool(lineage) and all(r['ids'] for r in lineage)
            affected = await self._admin_query('''MATCH ()-[r:RELATES_TO {group_id:$g}]->()
                WHERE any(id IN r.episodes WHERE id IN $episodes) RETURN count(r) AS n''',
                g=group_id,episodes=[s['id'] for s in detail['sources']])
            item = dict(id=row['id'],version=row['version'],content=row['content'],source_ids=source_ids,
                        episode_ids=sorted(s['id'] for s in detail['sources']),
                        reliable=reliable,affected_facts=affected[0]['n'] if affected else 0)
            item['fingerprint'] = hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
            items.append(item)
        return items

    async def control_refresh_edges(self, *, group_id, edge_ids, source_ids):
        # A new original utterance can independently support a previously withdrawn fact.
        # Replays from the blocked original evidence never reach this branch.
        blocked = await self.control_blocks(group_id=group_id)
        if not source_ids or set(source_ids) & blocked:
            return
        await self._admin_query('''UNWIND $ids AS id
            MERGE (e:BumiControlEvidence {key:$g+'::'+id})
            SET e.group_id=$g,e.edge_id=id,e.source_ids=$sources RETURN e.edge_id AS id''',
            g=group_id,ids=edge_ids,sources=source_ids)
        await self._admin_query('''UNWIND $ids AS id
            MATCH ()-[r:RELATES_TO {group_id:$g,uuid:id}]->()
            MATCH (o:BumiAdminMemory {key:$g+'::'+id}) WHERE o.control_deleted=true
            SET o.content=r.fact,o.deleted=false,o.control_deleted=false,o.control_relearned=true,
                o.control_source_ids=$sources,o.version=o.version+1,
                o.updated_at=datetime(),r.admin_version=o.version,r.admin_deleted=false
            RETURN r.uuid AS id''',g=group_id,ids=edge_ids,sources=source_ids)

    async def control_fresh_edges(self, *, group_id):
        rows=await self._admin_query('''MATCH (e:BumiControlEvidence {group_id:$g})
            WHERE size(e.source_ids)>0 AND NOT EXISTS {
                MATCH (b:BumiControlBlock {group_id:$g}) WHERE b.source_id IN e.source_ids }
            RETURN e.edge_id AS id''',g=group_id)
        return {r['id'] for r in rows}

    async def control_receipt(self, *, group_id, operation_id):
        rows = await self._admin_query('MATCH (o:BumiControlApplied {key:$key}) RETURN o.result AS result',key=group_id+'::'+operation_id)
        return json.loads(rows[0]['result']) if rows else None

    async def control_withdraw(self, *, group_id, operation_id, candidate, expires_at=None):
        """A single Cypher commit: source blocks + immutable operation receipt + CAS."""
        async with self._write_lock:
            previous = await self._admin_query('MATCH (o:BumiControlApplied {key:$key}) RETURN o.result AS result',key=group_id+'::'+operation_id)
            if previous:
                return json.loads(previous[0]['result'])
            if expires_at is not None and time.time()>expires_at:
                raise ValueError('selection_expired')
            current = await self.control_candidates(group_id=group_id,query=candidate['content'])
            match = next((r for r in current if r['id']==candidate['id']), None)
            if not match or match['fingerprint']!=candidate['fingerprint']:
                raise ValueError('candidate_changed')
            if not candidate['reliable']:
                raise ValueError('source_lineage_unavailable')
            result = dict(source_ids=candidate['source_ids'],deleted_id=candidate['id'],
                          affected_facts=candidate['affected_facts'],physical_erasure=False)
            rows = await self._admin_query('''MATCH ()-[r:RELATES_TO {group_id:$g,uuid:$id}]->()
                SET r.control_lock=coalesce(r.control_lock,0)+1
                WITH r OPTIONAL MATCH (a:BumiAdminMemory {key:$g+'::'+$id})
                WITH r,a WHERE coalesce(a.version,r.admin_version,0)=$version
                AND size(r.episodes)=size($episodes) AND all(e IN r.episodes WHERE e IN $episodes)
                MERGE (o1:BumiAdminMemory {key:$g+'::'+$id})
                SET o1.group_id=$g,o1.memory_id=$id,o1.content=coalesce(a.content,r.fact),
                    o1.version=$version+1,o1.deleted=true,o1.control_deleted=true,o1.control_relearned=false,o1.updated_at=datetime()
                CREATE (h:BumiAdminRevision {group_id:$g,memory_id:$id,version:$version+1,
                    action:'delete',before:r.fact,after:r.fact,at:datetime()})
                SET r.admin_version=$version+1,r.admin_deleted=true
                FOREACH (sid IN $ids | MERGE (b:BumiControlBlock {key:$g+'::'+sid})
                    SET b.group_id=$g,b.source_id=sid)
                MERGE (o:BumiControlApplied {key:$op}) SET o.group_id=$g,o.result=$result
                RETURN o.result AS result''',g=group_id,id=candidate['id'],version=candidate['version'],
                ids=candidate['source_ids'],episodes=candidate['episode_ids'],op=group_id+'::'+operation_id,result=json.dumps(result))
            if not rows:
                raise ValueError('candidate_changed')
            return json.loads(rows[0]['result'])
