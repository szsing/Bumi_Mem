"""Convergence regression from a real 50-turn graph, without GPU/Neo4j."""
import json
from pathlib import Path

import pytest
pytest.importorskip('graphiti_core')
from graphiti_core.utils.maintenance.community_operations import label_propagation, Neighbor
from bumi_mem.graphiti_community import stable_label_propagation


def graph(edges):
    result={}
    for a,b,weight in edges:
        result.setdefault(a,[]).append(dict(node_uuid=b,edge_count=weight))
        result.setdefault(b,[]).append(dict(node_uuid=a,edge_count=weight))
    return result


def canonical(clusters):return {frozenset(c) for c in clusters}


@pytest.mark.parametrize('edges', [[], [('a','b',1)], [('a','b',1),('b','c',1),('c','a',1)]])
def test_convergent_cases_keep_native_partition(edges):
    projection=graph(edges)
    native=label_propagation({u:[Neighbor(**n) for n in ns] for u,ns in projection.items()})
    actual,detail=stable_label_propagation(projection)
    assert not detail['fallback_used']
    assert canonical(actual)==canonical(native)


def test_parallel_edge_two_node_oscillation_converges():
    clusters,detail=stable_label_propagation(graph([('a','b',2)]))
    assert detail['fallback_used']
    assert canonical(clusters)=={frozenset(('a','b'))}


def test_real_fifth_batch_projection_converges_without_losing_nodes():
    projection=json.loads((Path(__file__).parent/'fixtures/graphiti_label_cycle.json').read_text())
    clusters,detail=stable_label_propagation(projection)
    assert detail['fallback_used']
    flattened=[u for c in clusters for u in c]
    assert len(flattened)==len(set(flattened))==len(projection)
    assert set(flattened)==set(projection)
    assert stable_label_propagation(projection)==(clusters,detail)


def test_asymmetric_or_foreign_projection_is_rejected():
    with pytest.raises(ValueError):stable_label_propagation({'a':[dict(node_uuid='foreign',edge_count=2)]})


def test_community_override_never_changes_ingestion_driver(tmp_path):
    from types import SimpleNamespace
    from bumi_mem.graphiti_community import community_driver
    connection=object()
    driver=SimpleNamespace(graph_operations_interface=None,client=connection)
    scoped=community_driver(driver,tmp_path)
    assert scoped is not driver and scoped.client is connection
    assert driver.graph_operations_interface is None
    assert scoped.graph_operations_interface is not None
