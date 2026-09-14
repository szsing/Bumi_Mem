import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import pytest
from pydantic import ValidationError
from bumi_mem.typed import (TypedMemoryStore, Scope, ObservationInput, SpanInput, MemoryDraft,
                           MemoryRef, GrantInput, ViewDraft, AccessDenied, ConflictError,
                           EvidenceError, SchemaMismatch, StaleRead)
from bumi_mem.typed.schema import APPLICATION_ID, SCHEMA_VERSION
from .conftest import scope, ALL_PURPOSES


def child():return scope('child',purpose='coordination.availability')


def test_schema_wal_foreign_keys_and_reopen(fx):
    with sqlite3.connect(fx.store.path) as c:
        assert c.execute('PRAGMA application_id').fetchone()[0]==APPLICATION_ID
        assert c.execute('PRAGMA user_version').fetchone()[0]==SCHEMA_VERSION
        assert c.execute('PRAGMA journal_mode').fetchone()[0]=='wal'
        assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
    reopened=TypedMemoryStore(fx.store.path,clock=fx.clock)
    assert reopened.epochs(scope())==fx.store.epochs(scope())


def test_refuses_legacy_and_legacy_refuses_typed(tmp_path,fx):
    from bumi_mem.session_store import SQLiteSessionStore
    legacy=tmp_path/'legacy.sqlite3';SQLiteSessionStore(legacy)
    before=hashlib.sha256(legacy.read_bytes()).hexdigest()
    with pytest.raises(SchemaMismatch):TypedMemoryStore(legacy)
    assert hashlib.sha256(legacy.read_bytes()).hexdigest()==before
    with pytest.raises(RuntimeError):SQLiteSessionStore(fx.store.path)


def test_newer_schema_not_silently_downgraded(fx):
    with sqlite3.connect(fx.store.path) as c:c.execute(f'PRAGMA user_version={SCHEMA_VERSION+1}')
    with pytest.raises(SchemaMismatch):TypedMemoryStore(fx.store.path)
    with sqlite3.connect(fx.store.path) as c:assert c.execute('PRAGMA user_version').fetchone()[0]==SCHEMA_VERSION+1


def test_foreign_database_is_not_modified(tmp_path):
    p=tmp_path/'foreign.sqlite'
    with sqlite3.connect(p) as c:c.execute('CREATE TABLE unrelated(value TEXT)');c.execute("INSERT INTO unrelated VALUES('keep')")
    before=p.read_bytes()
    with pytest.raises(SchemaMismatch):TypedMemoryStore(p)
    assert p.read_bytes()==before


def test_unknown_identity_not_persisted(fx):
    obs=ObservationInput(observation_id='unknown',conversation_id='private',text='我喜欢某件事',source_timestamp=fx.clock.iso(),storage_allowed=True)
    assert fx.store.record_observation(scope(None),obs)['persisted'] is False
    assert fx.store.list_views(scope(None))==[]
    with sqlite3.connect(fx.store.path) as c:assert c.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==0


@pytest.mark.parametrize('kwargs',[{'storage_allowed':False},{'is_final':False}])
def test_unapproved_or_partial_text_is_transient(fx,kwargs):
    values=dict(observation_id='o',conversation_id='private',text='内容',source_timestamp=fx.clock.iso(),storage_allowed=True)
    values.update(kwargs)
    assert not fx.store.record_observation(scope(),ObservationInput(**values))['persisted']


def test_observation_idempotence_and_revision_conflict(fx):
    oid,sid,obs,receipt=fx.source()
    assert fx.store.record_observation(scope(),obs)['status']=='already_persisted'
    altered=ObservationInput.model_validate({**obs.model_dump(),'text':'不同内容'})
    with pytest.raises(ConflictError):fx.store.record_observation(scope(),altered)
    assert len(fx.store.read_turns(scope(),'private'))==1


def test_unicode_span_exact_quote_and_wrong_offsets(fx):
    oid,sid,obs,_=fx.source('😀我喜欢徒步')
    valid=fx.store.add_span(scope(),SpanInput(observation_id=oid,revision=1,start=1,end=6,quote='我喜欢徒步'))
    assert valid
    with pytest.raises(EvidenceError):fx.store.add_span(scope(),SpanInput(observation_id=oid,revision=1,start=2,end=7,quote='我喜欢徒步'))
    with pytest.raises(EvidenceError):fx.store.add_span(scope('mom'),SpanInput(observation_id=oid,revision=1,start=0,end=len(obs.text),quote=obs.text))


def test_space_membership_is_not_private_conversation_access(fx):
    obs=ObservationInput(observation_id='attack',conversation_id='private',text='我是同空间成员',source_timestamp=fx.clock.iso(),storage_allowed=True)
    with pytest.raises(AccessDenied):fx.store.record_observation(scope('child'),obs)
    assert fx.store.read_turns(scope('child'),'private')==[]


def test_multi_speaker_conversation_and_later_audience_change(fx):
    group=scope('dad',['dad','mom'])
    fx.store.create_conversation(group,'group')
    fx.source('爸爸说话',ctx=group,conversation='group')
    fx.source('妈妈说话',ctx=scope('mom',['dad','mom']),conversation='group')
    assert [x['speaker_id'] for x in fx.store.read_turns(group,'group')]==['dad','mom']
    fx.store.add_conversation_participant(scope(),'group','child')
    assert fx.store.read_turns(scope('child'),'group')==[]


def test_non_creator_cannot_invite(fx):
    with pytest.raises(AccessDenied):fx.store.add_conversation_participant(scope('mom'),'private','child')


def test_carryover_retains_source_audience_and_not_new_evidence(fx):
    oid,sid,obs,receipt=fx.source()
    fx.store.create_conversation(scope(),'next')
    first=fx.store.carryover(scope(),receipt['turn_id'],'next')
    assert fx.store.carryover(scope(),receipt['turn_id'],'next')==first
    turns=fx.store.read_turns(scope(),'next')
    assert len(turns)==1 and turns[0]['context_only'] and turns[0]['observation_id']==oid
    fx.store.add_conversation_participant(scope(),'next','child')
    assert fx.store.read_turns(scope('child'),'next')==[]
    with pytest.raises(AccessDenied):fx.store.carryover(scope('child'),receipt['turn_id'],'child-private')


def test_private_view_and_raw_pointer_not_shared(fx):
    oid,sid,_,_=fx.source();fx.plan(source=sid)
    private=fx.private()
    assert '秘密餐厅甲' in fx.store.read_view(scope(),private).display_text
    assert fx.store.read_view(child(),private) is None
    assert fx.store.read_observation(scope('child',purpose='explicit_memory_review'),oid,1) is None


def test_all_subjects_must_authorize_and_shared_has_no_private_fields(fx):
    fx.plan();fx.allow('dad')
    with pytest.raises(AccessDenied):fx.shared()
    fx.allow('mom');view=fx.store.read_view(child(),fx.shared())
    assert view.projected_fields['availability']=='possibly_busy'
    assert 'location' not in view.projected_fields and '秘密餐厅甲' not in view.model_dump_json()


def test_cannot_grant_for_another_person(fx):
    g=GrantInput(subject_id='mom',audience_ids=['child'],effect='allow',category='coordination.availability',attributes=['time'],purposes=['coordination.availability'])
    with pytest.raises(AccessDenied):fx.store.grant(scope(),g)


def test_location_deny_does_not_block_only_availability(fx):
    fx.plan();fx.allow('dad');fx.allow('mom');fx.deny(attrs=['location'])
    assert fx.store.read_view(child(),fx.shared()) is not None
    with pytest.raises(AccessDenied):fx.shared(attrs=['location'])


def test_explicit_deny_blocks_existing_view_even_with_allow(fx):
    fx.plan();fx.allow('dad');fx.allow('mom');vid=fx.shared()
    fx.deny(attrs=['availability'])
    assert fx.store.read_view(child(),vid) is None


def test_inference_block_is_conservative_across_attributes(fx):
    fx.plan();fx.allow('dad');fx.allow('mom');vid=fx.shared()
    fx.deny(attrs=['location'],mid='plan',block=True)
    assert fx.store.read_view(child(),vid) is None


def test_audience_and_purpose_are_checked_including_requester(fx):
    fx.plan();fx.allow('dad');fx.allow('mom');vid=fx.shared()
    assert fx.store.read_view(scope('child',['child','visitor'],purpose='coordination.availability'),vid) is None
    assert fx.store.read_view(scope('visitor',['child'],purpose='coordination.availability'),vid) is None
    assert fx.store.read_view(scope('child',purpose='conversation_continuity'),vid) is None
    assert fx.store.read_view(scope('child',confirmed=False,purpose='coordination.availability'),vid) is None


def test_cross_space_read_by_known_actor_is_denied(fx):
    fx.plan();private=fx.private()
    assert fx.store.read_view(scope('dad',space='other'),private) is None


def test_parent_relation_does_not_grant_access(fx):
    fx.store.add_relation(scope(),'dad','child','parent_of')
    oid,sid,obs,_=fx.source('孩子自己的内容',ctx=scope('child'),conversation='child-private')
    assert fx.store.relations_for(scope(),'dad')
    assert fx.store.read_observation(scope('dad',purpose='explicit_memory_review'),oid,1) is None


def test_publication_cannot_accept_arbitrary_display_text(fx):
    with pytest.raises(ValidationError):ViewDraft(memory_id='plan',source_version=1,projection_type='availability',attributes=['time'],audience_ids=['child'],purposes=['coordination.availability'],display_text='秘密餐厅甲')


def test_revoke_and_restart_block_cached_ticket_and_old_view(fx):
    fx.plan();gid=fx.allow('dad');fx.allow('mom');vid=fx.shared();ticket=fx.store.prepare_read(child())
    assert fx.store.materialize(child(),ticket)
    fx.store.revoke_grant(scope(),gid)
    with pytest.raises(StaleRead):fx.store.materialize(child(),ticket)
    reopened=TypedMemoryStore(fx.store.path,clock=fx.clock)
    assert reopened.read_view(child(),vid) is None
    fx.allow('dad')
    assert fx.store.read_view(child(),vid) is None
    assert fx.store.read_view(child(),fx.shared()) is not None


def test_ticket_does_not_cross_user_or_purpose(fx):
    fx.plan();fx.allow('dad');fx.allow('mom');fx.shared();ticket=fx.store.prepare_read(child())
    with pytest.raises(StaleRead):fx.store.materialize(scope('visitor',purpose='coordination.availability'),ticket)
    with pytest.raises(StaleRead):fx.store.materialize(scope('child',purpose='explicit_memory_review'),ticket)


def test_expired_grant_not_restored_by_historical_query(fx):
    fx.plan();fx.allow('dad',expires_at='2026-09-14T10:00:00+00:00');fx.allow('mom')
    vid=fx.shared(snapshot='historical');ticket=fx.store.prepare_read(child(),temporal_mode='historical',as_of=fx.clock.iso())
    fx.clock.advance(7200)
    assert fx.store.read_view(child(),vid,temporal_mode='historical',as_of='2026-09-14T09:00:00+00:00') is None
    with pytest.raises(StaleRead):fx.store.materialize(child(),ticket)


def test_future_grant_not_usable_now(fx):
    fx.plan();fx.allow('dad',starts_at='2026-09-15T00:00:00+00:00');fx.allow('mom')
    with pytest.raises(AccessDenied):fx.shared()


def test_removed_member_rejoin_does_not_restore_sharing(fx):
    fx.plan();fx.allow('dad');fx.allow('mom');vid=fx.shared()
    fx.store.remove_member(scope(),'child');assert fx.store.read_view(child(),vid) is None
    fx.store.add_member(scope(),'child');assert fx.store.read_view(child(),vid) is None
    with pytest.raises(AccessDenied):fx.shared()


def identity(fx, value, *, mid='job', valid_from='2026-07-01T00:00:00+00:00'):
    oid,sid,obs,_=fx.source(f'我从所述时间开始在{value}工作。')
    draft=MemoryDraft(memory_id=mid,kind='identity',subject_ids=['dad'],assertion_kind='explicit_statement',payload={'predicate':'occupation','value':value},span_ids=[sid],valid_from=valid_from)
    return oid,draft


def test_temporal_update_preserves_true_historical_value(fx):
    oid,a=identity(fx,'公司甲');fx.store.create_memory(scope(),a)
    old_current=fx.private('job',attrs=['value'])
    _,b=identity(fx,'公司乙',valid_from='2026-09-01T00:00:00+00:00')
    assert fx.store.revise_memory(scope(),b,expected_version=1,reason='temporal_update')==2
    assert fx.store.read_view(scope(),old_current) is None
    history=fx.private('job',1,attrs=['value'],snapshot='historical')
    current=fx.private('job',2,attrs=['value'])
    assert fx.store.read_view(scope(),history,temporal_mode='historical',as_of='2026-08-15T00:00:00+00:00').projected_fields['value']=='公司甲'
    assert fx.store.read_view(scope(),history,temporal_mode='historical',as_of=fx.clock.iso()) is None
    assert fx.store.read_view(scope(),current).projected_fields['value']=='公司乙'


def test_correction_does_not_revive_old_error_as_history(fx):
    _,a=identity(fx,'错误公司');fx.store.create_memory(scope(),a)
    _,b=identity(fx,'真实公司')
    fx.store.revise_memory(scope(),b,expected_version=1,reason='correction')
    with pytest.raises(EvidenceError):fx.private('job',1,attrs=['value'],snapshot='historical')
    assert fx.store.read_view(scope(),fx.private('job',2,attrs=['value'])).projected_fields['value']=='真实公司'


def test_old_source_withdrawal_preserves_independent_current_version(fx):
    old_obs,a=identity(fx,'公司甲');fx.store.create_memory(scope(),a)
    _,b=identity(fx,'公司乙',valid_from='2026-09-01T00:00:00+00:00')
    fx.store.revise_memory(scope(),b,expected_version=1,reason='temporal_update')
    current=fx.private('job',2,attrs=['value'])
    fx.store.withdraw_observation(scope(),old_obs)
    assert fx.store.read_view(scope(),current).projected_fields['value']=='公司乙'


def test_duplicate_identity_field_requires_explicit_revision(fx):
    _,a=identity(fx,'公司甲');fx.store.create_memory(scope(),a)
    _,b=identity(fx,'公司乙',mid='other-job')
    with pytest.raises(ConflictError):fx.store.create_memory(scope(),b)


def test_stale_revision_rejected_and_atomic(fx):
    _,a=identity(fx,'公司甲');fx.store.create_memory(scope(),a)
    _,b=identity(fx,'公司乙');fx.store.revise_memory(scope(),b,expected_version=1,reason='correction')
    before=fx.store.epochs(scope())
    with pytest.raises(ConflictError):fx.store.revise_memory(scope(),b,expected_version=1,reason='correction')
    assert fx.store.epochs(scope())==before


def test_two_writers_expected_version_only_one_succeeds(fx):
    _,a=identity(fx,'公司甲');fx.store.create_memory(scope(),a)
    _,b=identity(fx,'公司乙');_,c=identity(fx,'公司丙')
    def update(draft):
        try:return fx.store.revise_memory(scope(),draft,expected_version=1,reason='correction')
        except ConflictError:return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(update,[b,c]))
    assert results.count(2)==1 and results.count('conflict')==1


def test_plan_cancelled_not_current_busy_or_automatically_free(fx):
    draft=fx.plan();fx.allow('dad');fx.allow('mom');vid=fx.shared();fx.clock.advance(1)
    _,sid,_,_=fx.source('刚才说的计划取消了。')
    values=draft.model_dump();values.update(span_ids=[sid]);values['payload'].update(event_status='cancelled',availability='unknown')
    fx.store.revise_memory(scope(),MemoryDraft.model_validate(values),expected_version=1,reason='temporal_update')
    assert fx.store.read_view(child(),vid) is None
    assert fx.store.read_view(child(),fx.shared(version=2)).projected_fields['availability']=='unknown'


def test_source_revision_invalidates_old_span_view_and_replay(fx):
    oid,sid,obs,_=fx.source();draft=fx.plan(source=sid);vid=fx.private()
    revised=ObservationInput.model_validate({**obs.model_dump(),'revision':2,'text':'修正后的原话'})
    fx.store.record_observation(scope(),revised)
    assert fx.store.read_view(scope(),vid) is None
    with pytest.raises(ConflictError):fx.store.record_observation(scope(),obs)
    with pytest.raises(EvidenceError):fx.store.add_span(scope(),SpanInput(observation_id=oid,revision=1,start=0,end=len(obs.text),quote=obs.text))


def test_missing_evidence_does_not_partially_insert(fx):
    d=MemoryDraft(memory_id='bad',kind='identity',subject_ids=['dad'],assertion_kind='explicit_statement',payload={'predicate':'occupation','value':'A'},span_ids=['missing'])
    before=fx.store.epochs(scope())
    with pytest.raises(EvidenceError):fx.store.create_memory(scope(),d)
    assert fx.store.epochs(scope())==before
    with sqlite3.connect(fx.store.path) as c:assert c.execute("SELECT COUNT(*) FROM memories WHERE memory_id='bad'").fetchone()[0]==0


def test_other_speaker_span_cannot_be_claimed_as_own(fx):
    _,sid,_,_=fx.source('妈妈自己的记录',ctx=scope('mom'),conversation='mom-private')
    d=MemoryDraft(memory_id='bad',kind='identity',subject_ids=['dad'],assertion_kind='explicit_statement',payload={'predicate':'occupation','value':'A'},span_ids=[sid])
    with pytest.raises(EvidenceError):fx.store.create_memory(scope(),d)


def test_assistant_suggestion_is_not_a_user_identity(fx):
    _,sid,_,_=fx.source('你也许可以做工程师。',role='assistant')
    d=MemoryDraft(memory_id='bad',kind='identity',subject_ids=['dad'],assertion_kind='explicit_statement',payload={'predicate':'occupation','value':'工程师'},span_ids=[sid])
    with pytest.raises(EvidenceError):fx.store.create_memory(scope(),d)


def test_candidate_behavior_is_review_only(fx):
    _,sid,_,_=fx.source('这次先听我说完。')
    values=dict(memory_id='pattern',kind='behavior',subject_ids=['dad'],assertion_kind='inferred',status='candidate',payload={'pattern_type':'interaction_pattern','preference':'先倾听'},span_ids=[sid])
    fx.store.create_memory(scope(),MemoryDraft(**values))
    with pytest.raises(AccessDenied):fx.private('pattern',attrs=['preference'])
    vid=fx.private('pattern',attrs=['preference'],purposes=['explicit_memory_review'])
    assert fx.store.read_view(scope(),vid) is None
    assert fx.store.read_view(scope(purpose='explicit_memory_review'),vid).status=='candidate'
    with pytest.raises(ValidationError):MemoryDraft(**{**values,'status':'active'})


def test_model_copy_cannot_bypass_validation_at_api_boundary(fx):
    _,sid,_,_=fx.source('先听我说完')
    d=MemoryDraft(memory_id='pattern',kind='behavior',subject_ids=['dad'],assertion_kind='inferred',status='candidate',payload={'pattern_type':'interaction_pattern','preference':'先倾听'},span_ids=[sid])
    poisoned=d.model_copy(update={'status':'active'})
    with pytest.raises(ValidationError):fx.store.create_memory(scope(),poisoned)


def test_derived_view_checks_source_attribute_permissions(fx):
    fx.plan('parent');fx.allow('dad');fx.allow('mom')
    fx.plan('derived',parents=[MemoryRef(memory_id='parent',version=1,attributes=['location','time','subject_ids'])])
    with pytest.raises(AccessDenied):fx.shared(mid='derived')


def test_deny_on_ancestor_cannot_be_laundered_through_new_memory(fx):
    fx.plan('parent');fx.allow('dad');fx.allow('mom')
    fx.plan('derived',parents=[MemoryRef(memory_id='parent',version=1,attributes=['availability','time','subject_ids'])])
    vid=fx.shared(mid='derived')
    fx.deny(attrs=['availability'],mid='parent')
    assert fx.store.read_view(child(),vid) is None


def test_cannot_drop_source_subjects(fx):
    fx.plan('parent')
    with pytest.raises(EvidenceError):fx.plan('derived',subjects=['dad'],parents=[MemoryRef(memory_id='parent',version=1,attributes=['availability'])])


def test_no_self_reinforcing_evidence_cycle(fx):
    a=fx.plan('a')
    fx.plan('b',parents=[MemoryRef(memory_id='a',version=1,attributes=['availability'])])
    values=a.model_dump();values.update(span_ids=[],parent_memories=[{'memory_id':'b','version':1,'attributes':['availability']}])
    with pytest.raises(EvidenceError):fx.store.revise_memory(scope(),MemoryDraft.model_validate(values),expected_version=1,reason='correction')


def test_delete_parent_revokes_derived_views_and_prevents_replay(fx):
    fx.plan('parent');fx.allow('dad');fx.allow('mom')
    fx.plan('derived',parents=[MemoryRef(memory_id='parent',version=1,attributes=['availability','time','subject_ids'])])
    vid=fx.shared('derived');ticket=fx.store.prepare_read(child())
    fx.store.delete_memory(scope(),'parent')
    assert fx.store.read_view(child(),vid) is None
    with pytest.raises(StaleRead):fx.store.materialize(child(),ticket)
    with pytest.raises(EvidenceError):fx.plan('replay',parents=[MemoryRef(memory_id='parent',version=1,attributes=['availability'])])


def test_raw_withdrawal_blocks_all_routes_after_restart(fx):
    oid,sid,obs,receipt=fx.source();fx.plan(source=sid);fx.allow('dad');fx.allow('mom');vid=fx.shared()
    fx.store.create_conversation(scope(),'next');fx.store.carryover(scope(),receipt['turn_id'],'next')
    fx.store.withdraw_observation(scope(),oid)
    reopened=TypedMemoryStore(fx.store.path,clock=fx.clock)
    assert reopened.read_view(child(),vid) is None
    assert reopened.read_turns(scope(),'private')==[] and reopened.read_turns(scope(),'next')==[]
    assert reopened.read_observation(scope(purpose='explicit_memory_review'),oid,1) is None


def test_deny_is_not_bypassed_by_raw_or_carryover(fx):
    group=scope('dad',['dad','child']);fx.store.create_conversation(group,'group')
    oid,sid,obs,receipt=fx.source(ctx=group,conversation='group');fx.plan(source=sid)
    assert fx.store.read_turns(child(),'group')
    fx.deny(attrs=['availability'],mid='plan',block=True)
    assert fx.store.read_turns(child(),'group')==[]
    with pytest.raises(AccessDenied):fx.store.carryover(child(),receipt['turn_id'],'child-private')


def test_global_raw_text_deny_works_without_a_memory_yet(fx):
    group=scope('dad',['dad','child']);fx.store.create_conversation(group,'group')
    fx.source(ctx=group,conversation='group')
    fx.deny(attrs=['text'],category='private')
    assert fx.store.read_turns(scope('child'),'group')==[]


def test_retention_expiry_rechecked_on_cached_ticket(fx):
    fx.plan();vid=fx.private();ticket=fx.store.prepare_read(scope())
    fx.clock.advance(14*24*3600+1)
    assert fx.store.read_view(scope(),vid) is None
    with pytest.raises(StaleRead):fx.store.materialize(scope(),ticket)


def test_forged_ticket_cannot_fetch_private_view(fx):
    fx.plan();private=fx.private();fx.allow('dad');fx.allow('mom');fx.shared()
    ticket=fx.store.prepare_read(child()).model_copy(update={'view_ids':[private]})
    with pytest.raises(StaleRead):fx.store.materialize(child(),ticket)


def test_returned_data_mutation_does_not_mutate_store(fx):
    fx.plan();vid=fx.private();result=fx.store.read_view(scope(),vid)
    result.projected_fields['location']='被改动'
    assert fx.store.read_view(scope(),vid).projected_fields['location']=='秘密餐厅甲'


def test_audit_has_ids_not_private_payload(fx):
    fx.plan();fx.private()
    with sqlite3.connect(fx.store.path) as c:rows=c.execute('SELECT * FROM audit_events').fetchall()
    assert '秘密餐厅甲' not in json.dumps(rows,ensure_ascii=False)


@pytest.mark.parametrize('bad',[
    {'precision':'unknown','start_at':'2026-09-14T18:00:00+08:00'},
    {'precision':'interval','start_at':'2026-09-14T20:00:00+08:00','end_at':'2026-09-14T18:00:00+08:00'},
    {'precision':'interval','start_at':'2026-09-14T18:00:00','end_at':'2026-09-14T20:00:00'},
    {'precision':'day_part','local_date':'2026-09-14','day_part':'秘密餐厅甲'},
])
def test_time_fields_do_not_accept_guessed_precision_or_hidden_free_text(bad):
    from bumi_mem.typed import EventTime
    with pytest.raises(ValidationError):EventTime.model_validate(bad)


def test_assistant_only_record_cannot_launder_into_user_fact(fx):
    _,sid,_,_=fx.source('建议你成为工程师。',role='assistant')
    record=MemoryDraft(memory_id='assistant-record',kind='record_index',subject_ids=['dad'],assertion_kind='explicit_statement',payload={'gist':'助手建议'},span_ids=[sid])
    fx.store.create_memory(scope(),record)
    fake=MemoryDraft(memory_id='fake-fact',kind='identity',subject_ids=['dad'],assertion_kind='explicit_statement',payload={'predicate':'occupation','value':'工程师'},parent_memories=[MemoryRef(memory_id='assistant-record',version=1,attributes=['gist'])])
    with pytest.raises(EvidenceError):fx.store.create_memory(scope(),fake)


def test_assistant_only_event_requires_commitment_type(fx):
    _,sid,_,_=fx.source('我答应记住这次交流。',role='assistant')
    with pytest.raises(EvidenceError):fx.plan(source=sid)
    promise=MemoryDraft(memory_id='promise',kind='episode',subject_ids=['dad'],assertion_kind='assistant_commitment',payload={'event_type':'assistant_commitment','event_status':'planned'},span_ids=[sid])
    fx.store.create_memory(scope(),promise)


def test_read_view_preserves_assertion_and_plan_uncertainty(fx):
    fx.plan();fx.allow('dad');fx.allow('mom')
    result=fx.store.read_view(child(),fx.shared())
    assert result.assertion_kind=='reported_statement'
    assert result.projected_fields['availability']=='possibly_busy'


def test_revision_failure_rolls_back_old_value_and_epoch(fx):
    _,a=identity(fx,'公司甲');fx.store.create_memory(scope(),a)
    vid=fx.private('job',attrs=['value']);before=fx.store.epochs(scope())
    _,b=identity(fx,'公司乙');before=fx.store.epochs(scope())
    invalid=MemoryDraft.model_validate({**b.model_dump(),'valid_from':None,'valid_until':'2026-01-01T00:00:00+00:00'})
    with pytest.raises(ConflictError):fx.store.revise_memory(scope(),invalid,expected_version=1,reason='correction')
    assert fx.store.epochs(scope())==before
    assert fx.store.read_view(scope(),vid).projected_fields['value']=='公司甲'


def test_incomplete_tagged_schema_is_refused(tmp_path):
    p=tmp_path/'incomplete.sqlite'
    with sqlite3.connect(p) as c:
        c.execute('CREATE TABLE scope_epochs(x)');c.execute('CREATE TABLE memory_views(x)')
        c.execute(f'PRAGMA application_id={APPLICATION_ID}');c.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
    with pytest.raises(SchemaMismatch):TypedMemoryStore(p)


def test_duplicate_revocation_does_not_invalidate_new_explicit_grant(fx):
    fx.plan();old=fx.allow('dad');fx.allow('mom');fx.shared()
    assert fx.store.revoke_grant(scope(),old) is True
    fx.allow('dad');new_view=fx.shared();before=fx.store.epochs(scope())
    assert fx.store.revoke_grant(scope(),old) is False
    assert fx.store.epochs(scope())==before
    assert fx.store.read_view(child(),new_view) is not None


def test_deletion_and_source_withdrawal_are_idempotent(fx):
    oid,sid,_,_=fx.source();fx.plan(source=sid)
    assert fx.store.delete_memory(scope(),'plan') is True
    before=fx.store.epochs(scope())
    assert fx.store.delete_memory(scope(),'plan') is False
    assert fx.store.epochs(scope())==before
    assert fx.store.withdraw_observation(scope(),oid) is True
    before=fx.store.epochs(scope())
    assert fx.store.withdraw_observation(scope(),oid) is False
    assert fx.store.epochs(scope())==before


def test_owner_can_logically_delete_an_expired_object(fx):
    fx.plan();fx.clock.advance(14*24*3600+1)
    assert fx.store.delete_memory(scope(),'plan') is True


def test_time_output_uses_canonical_iso_not_arbitrary_separator():
    from bumi_mem.typed import EventTime
    t=EventTime(precision='interval',start_at='2026-09-14秘18:00:00+08:00',end_at='2026-09-14密20:00:00+08:00')
    assert t.start_at=='2026-09-14T10:00:00+00:00'
    assert '秘' not in t.model_dump_json() and '密' not in t.model_dump_json()


def test_conversation_id_and_relation_registry_boundaries(fx):
    with pytest.raises(ValidationError):fx.store.create_conversation(scope(),'')
    with pytest.raises(ValidationError):fx.store.add_relation(scope(),'mom','child','arbitrary private sentence')
    fx.store.add_relation(scope(),'mom','child','parent_of')
    assert fx.store.relations_for(scope('child'),'mom')
    fx.store.remove_member(scope(),'mom')
    assert fx.store.relations_for(scope('child'),'mom')==[]


def test_real_new_process_observes_permission_revocation(fx):
    import subprocess,sys
    from pathlib import Path
    fx.plan();gid=fx.allow('dad');fx.allow('mom');vid=fx.shared()
    program = """
import sys,json,socket
sys.path.insert(0,sys.argv[1])
def forbidden(*a,**k):raise AssertionError('network forbidden in restart test')
socket.socket.connect=forbidden
from bumi_mem.typed import TypedMemoryStore,Scope
store=TypedMemoryStore(sys.argv[2],clock=lambda:float(sys.argv[4]))
ctx=Scope(space_id='home',actor_id='child',audience_ids=['child'],identity_status='verified_by_upstream',purpose='coordination.availability')
v=store.read_view(ctx,sys.argv[3])
print(json.dumps(None if v is None else v.projected_fields))
"""
    args=[sys.executable,'-c',program,str(Path(__file__).resolve().parents[2]/'src'),str(fx.store.path),vid,str(fx.clock.value)]
    before=json.loads(subprocess.check_output(args,text=True,timeout=20))
    assert before['availability']=='possibly_busy'
    fx.store.revoke_grant(scope(),gid)
    after=json.loads(subprocess.check_output(args,text=True,timeout=20))
    assert after is None
