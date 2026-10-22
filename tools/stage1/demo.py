#!/usr/bin/env python3
"""Stage-1 手动验收演示：不依赖 API、模型、GPU 或旧版数据库。"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bumi_mem.typed import (TypedMemoryStore, Scope, ObservationInput, SpanInput,
                           MemoryDraft, GrantInput, ViewDraft, AccessDenied, StaleRead)


def context(person, *, audience=None, purpose='conversation_continuity'):
    return Scope(space_id='synthetic-home', actor_id=person, audience_ids=audience or [person],
                 identity_status='verified_by_upstream', purpose=purpose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    allowed = (ROOT.parent / 'mem0-runtime/stage1/runs').resolve()
    if run == allowed or not run.is_relative_to(allowed):
        raise ValueError('use an isolated child of the Stage-1 runs directory')
    run.mkdir(parents=True, exist_ok=True)
    db = run / 'typed_demo.sqlite3'
    if db.exists():
        raise FileExistsError('demo never overwrites an existing database; use a new run directory')
    store = TypedMemoryStore(db)
    for person, label in [('dad','合成爸爸'),('mom','合成妈妈'),('child','合成孩子'),('visitor','合成访客')]:
        store.create_principal(person,label)
    dad, mom, child = context('dad'), context('mom'), context('child',purpose='coordination.availability')
    store.create_space('synthetic-home','dad')
    for person in ('mom','child','visitor'):store.add_member(dad,person)
    store.add_relation(dad,'dad','child','parent_of')
    store.add_relation(dad,'mom','child','parent_of')
    store.create_conversation(dad,'parents-private')
    raw='妈妈和我今晚打算去合成餐厅甲吃饭。'
    source=ObservationInput(observation_id='synthetic-plan-source',conversation_id='parents-private',
                            text=raw,source_timestamp=datetime.now(timezone.utc).isoformat(),storage_allowed=True)
    store.record_observation(dad,source)
    sid=store.add_span(dad,SpanInput(observation_id=source.observation_id,revision=1,start=0,end=len(raw),quote=raw))
    draft=MemoryDraft(memory_id='synthetic-plan',kind='episode',subject_ids=['dad','mom'],
                      assertion_kind='reported_statement',payload={'event_type':'activity_plan','activity':'meal',
                      'event_status':'planned','availability':'busy','time':{'precision':'day_part',
                      'local_date':datetime.now().date().isoformat(),'day_part':'evening'},'location':'合成餐厅甲'},span_ids=[sid])
    store.create_memory(dad,draft)
    private=store.create_view(dad,ViewDraft(memory_id=draft.memory_id,source_version=1,projection_type='private',
                              attributes=['location','availability','time','subject_ids'],audience_ids=['dad'],
                              purposes=['conversation_continuity','explicit_memory_review']))
    view_input=ViewDraft(memory_id=draft.memory_id,source_version=1,projection_type='availability',
                         attributes=['availability','time','subject_ids'],audience_ids=['child'],
                         purposes=['coordination.availability'])
    blocked_before_grants=False
    try:store.create_view(dad,view_input)
    except AccessDenied:blocked_before_grants=True
    grants=[]
    for person,ctx in [('dad',dad),('mom',mom)]:
        grants.append(store.grant(ctx,GrantInput(subject_id=person,audience_ids=['child'],effect='allow',
                      category='coordination.availability',attributes=['availability','time','subject_ids'],
                      purposes=['coordination.availability'])))
    shared=store.create_view(dad,view_input)
    result=store.read_view(child,shared)
    assert result is not None and '合成餐厅甲' not in result.model_dump_json()
    assert result.projected_fields['availability']=='possibly_busy'
    ticket=store.prepare_read(child)
    child_private=store.read_view(child,private)
    child_raw=store.read_observation(context('child',purpose='explicit_memory_review'),source.observation_id,1)
    visitor=store.read_view(context('child',audience=['child','visitor'],purpose='coordination.availability'),shared)
    store.revoke_grant(dad,grants[0])
    cached_rejected=False
    try:store.materialize(child,ticket)
    except StaleRead:cached_rejected=True
    reopened=TypedMemoryStore(db)
    after_restart=reopened.read_view(child,shared)
    report={'synthetic':True,'database':str(db),'models_or_api_called':False,
            'sharing_without_all_subject_grants_blocked':blocked_before_grants,
            'parent_private_view':store.read_view(dad,private).model_dump(),
            'child_shared_view_before_revoke':result.model_dump(),
            'child_private_view':None if child_private is None else child_private.model_dump(),
            'child_raw_source':child_raw,'view_with_visitor':None if visitor is None else visitor.model_dump(),
            'old_read_ticket_rejected_after_revoke':cached_rejected,
            'shared_view_after_reopen':None if after_restart is None else after_restart.model_dump(),
            'scope':'manual data/permission/version foundation, NOT automatic extraction, sharing inference or chat integration'}
    assert blocked_before_grants and cached_rejected and child_private is None and child_raw is None and visitor is None and after_restart is None
    (run/'demo_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()