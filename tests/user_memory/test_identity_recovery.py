"""Only user-ID isolation/confirmation checks; not the complete regression suite."""
import hashlib
import json
import threading
import time
from argparse import Namespace
from types import SimpleNamespace
import pytest
from test_bumi_memory import make_engine as make_fixture
from bumi_mem.identity import UserIdentity
from bumi_mem.long_term import Mem0LongTermMemory, FakeLongTermMemory
from bumi_mem.schemas import IdentityRequiredError, MemoryScopeError, MemoryWriteUncertainError
from bumi_mem.cli import run_interactive


def make_engine(*args,**kwargs):
    return make_fixture(*args,**kwargs)[0]


@pytest.mark.parametrize('uid',[None,'',' ',' a','a\n',0])
def test_missing_or_ambiguous_user_is_rejected_before_io(tmp_path,uid):
    engine=make_engine(tmp_path)
    for call in [lambda:engine.chat(user_id=uid,agent_id='bot',text='hello'),
                 lambda:engine.search_long_term(user_id=uid,agent_id='bot',query='q'),
                 lambda:engine.get_memories(user_id=uid,agent_id='bot'),
                 lambda:engine.remember(user_id=uid,agent_id='bot',contents=['fact']),
                 lambda:engine.read_memory(user_id=uid,agent_id='bot',memory_id='m')]:
        with pytest.raises(IdentityRequiredError):call()
    with engine.store._connect() as c:assert c.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]==0


def test_new_connection_is_unidentified_and_old_tickets_cannot_recur():
    a=UserIdentity();token=a.bind('A');b=UserIdentity()
    with pytest.raises(IdentityRequiredError):b.snapshot()
    b.bind('A');assert not b.accepts(token)


def test_late_response_is_not_delivered_to_new_user():
    state=UserIdentity();a=state.bind('A');out=[]
    b=state.bind('B')
    assert not state.deliver(a,out.append,'A_PRIVATE')
    assert state.deliver(b,out.append,'B_PRIVATE')
    state.clear();assert not state.deliver(b,out.append,'B_PRIVATE')
    assert out==['B_PRIVATE']


def test_confirmed_records_have_owner_ids_and_content_hashes(tmp_path):
    engine=make_engine(tmp_path)
    receipt=engine.remember(user_id='A',agent_id='bot',contents=[' A_COFFEE ','A_CITY'])
    assert receipt.confirmed and len(receipt.records)==2
    for record in receipt.records:
        actual=engine.read_memory(user_id='A',agent_id='bot',memory_id=record.memory_id)
        assert record.user_id=='A' and record.agent_id=='bot'
        assert record.content_sha256==hashlib.sha256(actual['memory'].encode()).hexdigest()
        assert engine.read_memory(user_id='B',agent_id='bot',memory_id=record.memory_id) is None


@pytest.mark.parametrize('fault',['wrong_user','wrong_text','missing_record','duplicate_id'])
def test_bad_write_result_is_never_confirmed(tmp_path,fault):
    class Bad(FakeLongTermMemory):
        def remember(self,**kw):
            rows=super().remember(**kw)
            if fault=='wrong_user':rows[0]['user_id']='B'
            if fault=='wrong_text':rows[0]['memory']='wrong'
            if fault=='missing_record':rows=rows[:1]
            if fault=='duplicate_id':rows[1]['id']=rows[0]['id']
            return rows
    engine=make_engine(tmp_path,long_term=Bad())
    with pytest.raises(MemoryWriteUncertainError):engine.remember(user_id='A',agent_id='bot',contents=['x','y'])


def test_real_adapter_rejects_mixed_owner_results_without_a_model():
    adapter=object.__new__(Mem0LongTermMemory)
    wrong={'id':'m','memory':'B_PRIVATE','user_id':'B','agent_id':'bot'}
    adapter.memory=SimpleNamespace(search=lambda **kw:{'results':[wrong]},get_all=lambda **kw:{'results':[wrong]},get=lambda mid:wrong)
    with pytest.raises(MemoryScopeError):adapter.search(user_id='A',agent_id='bot',query='q',top_k=5,threshold=.1)
    with pytest.raises(MemoryScopeError):adapter.get_all(user_id='A',agent_id='bot')
    assert adapter.get_memory(user_id='A',agent_id='bot',memory_id='m') is None


def test_metadata_cannot_override_user_binding():
    adapter=object.__new__(Mem0LongTermMemory)
    with pytest.raises(MemoryScopeError):adapter.add_session(user_id='A',agent_id='bot',messages=[],metadata={'user_id':'B'},prompt='')


def test_user_switch_shares_l1_but_keeps_scoped_memory(tmp_path):
    engine=make_engine(tmp_path,max_messages=100)
    engine.remember(user_id='A',agent_id='bot',contents=['A_ONLY_COFFEE'])
    engine.remember(user_id='B',agent_id='bot',contents=['B_ONLY_TEA'])
    a=engine.chat(user_id='A',agent_id='bot',text='A_ONLY_CONVERSATION')
    b=engine.chat(user_id='B',agent_id='bot',text='B_ONLY_CONVERSATION')
    assert a.user_id=='A' and b.user_id=='B' and a.session_id!=b.session_id
    joined='\n'.join(x['content'] for x in b.prompt_messages)
    assert 'A_ONLY_CONVERSATION' in joined and 'B_ONLY_CONVERSATION' in joined
    direct_context, shared_context = b.prompt_messages[0]['content'].split('[共享 L3 引用 / shared memory references]\n', 1)
    assert 'A_ONLY_COFFEE' not in direct_context
    refs = json.loads(shared_context.split('\n', 1)[0])
    assert any(r['owner_user_id'] == 'A' and r['memory'] == 'A_ONLY_COFFEE' for r in refs)
    assert any(r['owner_user_id'] == 'B' and r['memory'] == 'B_ONLY_TEA' for r in refs)
    assert [r['memory'] for r in b.recalled] == ['B_ONLY_TEA']
    assert '"user_id": "A"' in joined and '"user_id": "B"' in joined


def test_pending_job_is_not_a_confirmed_memory_receipt(tmp_path):
    engine=make_engine(tmp_path,auto_start=False)
    engine.chat(user_id='A',agent_id='bot',text='A fact')
    submission=engine.end_session(user_id='A',agent_id='bot')
    assert submission.accepted and not getattr(submission,'confirmed',False)
    assert submission.status=='queued'


def test_cli_no_default_user_and_no_implicit_ltm_access(tmp_path,monkeypatch,capsys):
    backend=FakeLongTermMemory();engine=make_engine(tmp_path,long_term=backend)
    args=Namespace(user_id=None,agent_id='bot',debug=False,chat_provider='deepseek',memory_provider='deepseek',max_messages=20,max_l1_tokens=8000)
    commands=iter(['hello','/memories','/user A','/remember A_ONLY','/user B','/memories','/logout','/memories','/quit'])
    monkeypatch.setattr('builtins.input',lambda *a:next(commands))
    run_interactive(engine,args)
    assert len(backend.memories.get(('A','bot'),[]))==1
    assert not backend.memories.get(('B','bot'))
    assert backend.warmup_calls == ['记忆系统预热']
    assert all(call['user_id'] for call in backend.search_calls)
    assert '尚未确认 user_id' in capsys.readouterr().out


def test_delayed_job_keeps_original_user_after_other_user_chats(tmp_path):
    started=threading.Event();release=threading.Event()
    class Slow(FakeLongTermMemory):
        def add_session(self,**kw):
            started.set();assert release.wait(5)
            return super().add_session(**kw)
    backend=Slow();engine=make_engine(tmp_path,long_term=backend,auto_start=True,max_messages=100)
    engine.chat(user_id='A',agent_id='bot',text='A_PENDING_FACT')
    job=engine.end_session(user_id='A',agent_id='bot')
    assert started.wait(3)
    try:
        b=engine.chat(user_id='B',agent_id='bot',text='B_CURRENT')
        assert b.user_id=='B' and any('A_PENDING_FACT' in m['content'] for m in b.prompt_messages)
    finally:release.set()
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        if engine.store.get_job(job.job_id).status=='succeeded':break
        time.sleep(.02)
    assert engine.store.get_job(job.job_id).user_id=='A'
    assert backend.add_calls[0]['user_id']=='A'


def test_session_id_cannot_be_reused_for_another_user(tmp_path):
    engine=make_engine(tmp_path);session=engine.get_or_create_session(user_id='A',agent_id='bot')
    with pytest.raises(MemoryScopeError):engine.record_user_message(session.session_id,'B text',user_id='B',agent_id='bot')
    with pytest.raises(MemoryScopeError):engine.build_context(session_id=session.session_id,recalled=[],user_id='B',agent_id='bot')
    assert engine.store.stats(session.session_id).effective_messages==0


def test_foreign_backend_memory_never_enters_prompt(tmp_path):
    backend=FakeLongTermMemory();backend.search_results=[{'id':'x','memory':'B_PRIVATE','user_id':'B','agent_id':'bot'}]
    engine=make_engine(tmp_path,long_term=backend)
    result=engine.chat(user_id='A',agent_id='bot',text='hello')
    assert result.recall_status=='unavailable'
    assert all('B_PRIVATE' not in m['content'] for m in result.prompt_messages)
