"""Instance-scoped fix for oscillating upstream community label propagation.

The native synchronous algorithm is retained for convergent inputs. A repeated
state switches to asynchronous, strictly improving weighted label moves; ties
keep the current label. Community prompts, summaries and persistence stay native.
"""
from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
import json
import os
from pathlib import Path
import uuid

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.nodes import EntityNode


def community_driver(driver, diagnostic_dir):
    """Scope the override to community calls; bulk ingestion has no fallback."""
    scoped=copy.copy(driver)
    if driver.graph_operations_interface is not None:
        raise ValueError('Unexpected pre-existing graph operation override')
    scoped.graph_operations_interface=ConvergentCommunityOperations(diagnostic_dir=diagnostic_dir)
    return scoped


def stable_label_propagation(projection):
    adjacency={u:{n['node_uuid']:n['edge_count'] for n in ns if n['node_uuid']!=u}
               for u,ns in projection.items()}
    for u,neighbors in adjacency.items():
        for v,w in neighbors.items():
            if w<=0 or v not in adjacency or adjacency[v].get(u)!=w:
                raise ValueError('Community projection must have symmetric positive edge counts')
    labels={u:i for i,u in enumerate(projection)}
    seen=set()
    fallback=False
    for step in range(max(32,4*len(labels))):
        state=tuple(labels.values())
        if state in seen:
            fallback=True
            break
        seen.add(state)
        new={}
        for u,neighbors in projection.items():
            votes=defaultdict(int)
            for n in neighbors:
                votes[labels[n['node_uuid']]]+=n['edge_count']
            options=sorted(((count,label) for label,count in votes.items()),reverse=True)
            weight,candidate=options[0] if options else (0,-1)
            new[u]=candidate if candidate!=-1 and weight>1 else max(candidate,labels[u])
        if new==labels:
            break
        labels=new
    else:
        fallback=True
    if fallback:
        # For an undirected integer-weighted graph every changed label strictly
        # raises the total within-community edge weight. That potential is bounded
        # by total edge weight, so this phase cannot oscillate.
        bound=sum(sum(ns.values()) for ns in adjacency.values())+1
        for _ in range(bound):
            changed=False
            for u,neighbors in adjacency.items():
                votes=defaultdict(int)
                for v,w in neighbors.items():votes[labels[v]]+=w
                if not votes:continue
                best=max(votes,key=lambda label:(votes[label],label))
                if votes[best]>votes[labels[u]]:
                    labels[u]=best
                    changed=True
            if not changed:break
        else:
            raise RuntimeError('Community potential bound exceeded')
    clusters=defaultdict(list)
    for u,label in labels.items():clusters[label].append(u)
    return list(clusters.values()), {'fallback_used':fallback,'native_states':len(seen)}


class ConvergentCommunityOperations(GraphOperationsInterface):
    diagnostic_dir: Path

    async def get_community_clusters(self, driver, group_ids):
        if not group_ids:
            raise ValueError('Bumi community construction requires explicit group IDs')
        result=[]
        for group in group_ids:
            nodes=await EntityNode.get_by_group_ids(driver,[group])
            projection={n.uuid:[] for n in nodes}
            records,_,_=await driver.execute_query('''
                MATCH (n:Entity {group_id:$group})
                OPTIONAL MATCH (n)-[e:RELATES_TO]-(m:Entity {group_id:$group})
                WITH n.uuid AS uuid, m.uuid AS neighbor, count(e) AS weight
                RETURN uuid, collect({node_uuid:neighbor,edge_count:weight}) AS neighbors
            ''',group=group)
            for record in records:
                projection[record['uuid']]=[dict(n) for n in record['neighbors'] if n['node_uuid'] is not None]
            clusters,detail=await asyncio.to_thread(stable_label_propagation,projection)
            by_uuid={n.uuid:n for n in nodes}
            result.extend([[by_uuid[u] for u in cluster] for cluster in clusters])
            self.diagnostic_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
            path=self.diagnostic_dir/(uuid.uuid4().hex+'.json')
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'w') as f:
                json.dump(dict(group_id=group,projection=projection,clusters=clusters,**detail),f)
                f.write('\n')
        return result
