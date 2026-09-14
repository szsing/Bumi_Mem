from datetime import datetime, timezone
import socket
import pytest
from bumi_mem.typed import TypedMemoryStore, Scope, ObservationInput, SpanInput, MemoryDraft, GrantInput, ViewDraft

ALL_PURPOSES = ['conversation_continuity','interaction_adaptation','coordination.availability','coordination.shared_activity','explicit_memory_review']

class Clock:
    def __init__(self): self.value = datetime(2026,9,14,9,tzinfo=timezone.utc).timestamp()
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds
    def iso(self): return datetime.fromtimestamp(self.value,timezone.utc).isoformat()


def scope(actor='dad', audience=None, purpose='conversation_continuity', space='home', confirmed=True):
    return Scope(space_id=space, actor_id=actor, audience_ids=audience if audience is not None else ([actor] if actor else []), identity_status='verified_by_upstream' if actor else 'unknown', purpose=purpose, audience_confirmed=confirmed)


class Fixture:
    def __init__(self, path):
        self.clock=Clock();self.store=TypedMemoryStore(path,clock=self.clock);self.n=0
        for p in ('dad','mom','child','visitor','outsider'):self.store.create_principal(p,p)
        self.store.create_space('home','dad')
        for p in ('mom','child','visitor'):self.store.add_member(scope(),p)
        self.store.create_space('other','outsider')
        self.store.add_member(scope('outsider',space='other'),'dad')
        self.store.create_conversation(scope(),'private')
        self.store.create_conversation(scope('mom'),'mom-private')
        self.store.create_conversation(scope('child'),'child-private')

    def source(self, text='妈妈和我今晚去秘密餐厅甲吃饭。', ctx=None, role='user', conversation='private'):
        self.n+=1;ctx=ctx or scope();oid=f'obs-{self.n}'
        observation=ObservationInput(observation_id=oid,conversation_id=conversation,text=text,source_timestamp=self.clock.iso(),role=role,storage_allowed=True)
        receipt=self.store.record_observation(ctx,observation)
        sid=self.store.add_span(ctx,SpanInput(observation_id=oid,revision=1,start=0,end=len(text),quote=text))
        return oid,sid,observation,receipt

    def plan(self, mid='plan', *, subjects=None, ctx=None, source=None, parents=None):
        ctx=ctx or scope();subjects=subjects or ['dad','mom']
        if source is None and not parents:source=self.source(ctx=ctx)[1]
        draft=MemoryDraft(memory_id=mid,kind='episode',subject_ids=subjects,assertion_kind='reported_statement',payload={'event_type':'activity_plan','activity':'meal','event_status':'planned','availability':'busy','time':{'precision':'interval','start_at':'2026-09-14T18:00:00+08:00','end_at':'2026-09-14T20:00:00+08:00'},'location':'秘密餐厅甲'},span_ids=[source] if source else [],parent_memories=parents or [])
        self.store.create_memory(ctx,draft);return draft

    def allow(self, subject='dad', attrs=None, recipients=None, **kw):
        return self.store.grant(scope(subject),GrantInput(subject_id=subject,audience_ids=recipients or ['child'],effect='allow',category='coordination.availability',attributes=attrs or ['availability','time','subject_ids'],purposes=['coordination.availability'],**kw))

    def shared(self, mid='plan', version=1, *, attrs=None, recipients=None, snapshot='current'):
        return self.store.create_view(scope(),ViewDraft(memory_id=mid,source_version=version,snapshot_kind=snapshot,projection_type='availability',attributes=attrs or ['availability','time','subject_ids'],audience_ids=recipients or ['child'],purposes=['coordination.availability']))

    def private(self, mid='plan', version=1, *, attrs=None, snapshot='current', purposes=None):
        return self.store.create_view(scope(),ViewDraft(memory_id=mid,source_version=version,snapshot_kind=snapshot,projection_type='private',attributes=attrs or ['availability','time','location','subject_ids'],audience_ids=['dad'],purposes=purposes or ALL_PURPOSES))

    def deny(self, subject='dad', attrs=None, *, recipients=None, mid=None, category='*', block=False, purposes=None, **kw):
        return self.store.grant(scope(subject),GrantInput(subject_id=subject,audience_ids=recipients or ['child'],effect='deny',category=category,attributes=attrs or ['location'],purposes=purposes or ALL_PURPOSES,memory_id=mid,block_inference=block,**kw))

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Stage 1 must not call models or network')
    monkeypatch.setattr(socket.socket,'connect',forbidden)

@pytest.fixture
def fx(tmp_path):return Fixture(tmp_path/'typed.sqlite3')
