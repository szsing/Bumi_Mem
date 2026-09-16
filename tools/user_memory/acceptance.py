#!/usr/bin/env python3
"""Only three-user isolation and acknowledged-record recovery. No latency benchmark.

Kills only a Popen child owned by this tool. Never suspends, reboots or powers off a host.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[2]
RUNTIME=ROOT.parent/'mem0-runtime'
ALLOWED=(RUNTIME/'acceptance/user_memory/runs').resolve()
USERS={
 'user_A':['甲的饮料偏好是热拿铁。','甲住在杭州。','甲周末喜欢跑步。','甲喜欢简短回答。','甲的测试标记是A_ONLY。'],
 'user_B':['乙的饮料偏好是绿茶。','乙住在成都。','乙周末喜欢爬山。','乙喜欢详细回答。','乙的测试标记是B_ONLY。'],
 'user_C':['丙的饮料偏好是白水。','丙住在西安。','丙周末喜欢游泳。','丙喜欢先给结论。','丙的测试标记是C_ONLY。']}
AGENT='user_memory_acceptance'


def atomic_json(path,obj):
    temp=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    with temp.open('x',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def setup(path):
    path=Path(path).resolve()
    if path==ALLOWED or not path.is_relative_to(ALLOWED):raise ValueError('isolated acceptance run required')
    for name in ('home','cache','tmp','reports'):(path/name).mkdir(parents=True,exist_ok=True)
    for key in ('DEEPSEEK_API_KEY','OPENAI_API_KEY','OPENROUTER_API_KEY','BUMI_CHAT_API_KEY','BUMI_MEMORY_API_KEY'):os.environ.pop(key,None)
    os.environ.update(MEM0_TELEMETRY='False',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HOME=str(path/'home'),XDG_CACHE_HOME=str(path/'cache'),TMPDIR=str(path/'tmp'))
    sys.path[:0]=[str(ROOT/'src')]
    return path


def copy_model_cache(path):
    src=RUNTIME/'tmp/fastembed_cache/models--Qdrant--bm25'
    dst=path/'tmp/fastembed_cache/models--Qdrant--bm25'
    if src.is_dir() and not dst.exists():
        files=[p for p in src.rglob('*') if p.is_file()]
        if sum(p.stat().st_size for p in files)>32*1024**2:raise RuntimeError('unexpected large model cache')
        if not all(p.resolve().is_relative_to(src.resolve()) for p in files):raise RuntimeError('cache symlink escapes model directory')
        shutil.copytree(src,dst,symlinks=False)


def make_engine(path):
    import threading
    from bumi_mem.engine import BumiMemory,BumiMemoryConfig
    from bumi_mem.session_store import SQLiteSessionStore
    from bumi_mem.consolidator import SessionConsolidator
    from bumi_mem.long_term import Mem0AdapterConfig,Mem0LongTermMemory
    from bumi_mem.providers import FakeTextProvider,ProviderConfig
    from bumi_mem.memory_access import SerialMemoryAccess
    from bumi_mem.token_counter import FixedTokenCounter
    closed=threading.Event()
    class RealDiskNoLLM(Mem0LongTermMemory):
        def __init__(self):
            super().__init__(ProviderConfig('deepseek','unused-no-llm','https://api.deepseek.com/v1','unused-no-api'),Mem0AdapterConfig(data_dir=path/'real_ltm',embed_model=ROOT.parent/'models/Qwen3-Embedding-0.6B',embed_device=os.getenv('MEM0_EMBED_DEVICE','cpu'),collection_name='strict_user_acceptance'))
            def forbidden(*a,**kw):raise AssertionError('LLM calls forbidden in this acceptance')
            self.memory.llm.generate_response=forbidden
        def close(self):
            try:
                entity=getattr(self.memory,'_entity_store',None)
                if entity is not None and getattr(entity,'client',None):entity.client.close()
                super().close()
            finally:closed.set()
    store=SQLiteSessionStore(path/'sessions.sqlite3')
    memory=SerialMemoryAccess(RealDiskNoLLM,read_timeout=120.0)
    provider=FakeTextProvider('fixture-only')
    consolidator=SessionConsolidator(store=store,long_term=memory,summary_provider=provider,memory_provider_name='fixture-only',memory_token_counter=FixedTokenCounter())
    engine=BumiMemory(store=store,long_term=memory,chat_provider=provider,consolidator=consolidator,chat_token_counter=FixedTokenCounter(),config=BumiMemoryConfig(max_messages=100,top_k=10,search_threshold=0.0),auto_start=False)
    return engine,closed


def boot_id():
    p=Path('/proc/sys/kernel/random/boot_id')
    return p.read_text().strip() if p.exists() else None


def write_dataset(path):
    if (path/'manifest.json').exists():raise FileExistsError('do not replay writes into an existing acceptance run')
    engine,closed=make_engine(path)
    try:
        receipts=[]
        for uid,texts in USERS.items():
            receipt=engine.remember(user_id=uid,agent_id=AGENT,contents=texts)
            assert receipt.confirmed and len(receipt.records)==len(texts)
            receipts.append(asdict(receipt))
        manifest={'schema':'user-memory-acceptance-v1','writer_pid':os.getpid(),'boot_id_before':boot_id(),'created_at':time.time(),'users':USERS,'agent_id':AGENT,'receipts':receipts,'expected_count':15,'scope':'explicit confirmed records only; not LLM extraction or arbitrary query accuracy'}
        atomic_json(path/'manifest.json',manifest)
        print(json.dumps({'ready':True,'pid':os.getpid(),'confirmed_records':15}),flush=True)
        sys.stdin.readline()  # Parent may request a normal exit or SIGKILL this owned process.
    finally:
        engine.close();closed.wait(30)


def verify(path,event='unspecified'):
    from bumi_mem.identity import UserIdentity
    from bumi_mem.schemas import IdentityRequiredError
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['expected_count']==15 and len(manifest['receipts'])==3
    expected_records=[x for receipt in manifest['receipts'] for x in receipt['records']]
    assert len(expected_records)==15 and len({x['memory_id'] for x in expected_records})==15
    assert {x['user_id'] for x in manifest['receipts']}==set(USERS)
    for receipt in manifest['receipts']:
        assert receipt['confirmed'] is True and len(receipt['records'])==5 and receipt['agent_id']==AGENT
        assert all(x['user_id']==receipt['user_id'] and x['agent_id']==AGENT and len(x['content_sha256'])==64 for x in receipt['records'])
    engine,closed=make_engine(path)
    checks=[];pollution=[];missing=[];hash_mismatch=[];binding_mismatch=[];context_leaks=[]
    try:
        identity=UserIdentity()
        for method in (lambda:engine.get_memories(user_id=None,agent_id=AGENT),lambda:engine.remember(user_id=None,agent_id=AGENT,contents=['must not persist'])):
            try:method()
            except IdentityRequiredError:checks.append('unidentified_blocked')
            else:raise AssertionError('unidentified request reached long-term memory')
        for receipt in manifest['receipts']:
            uid=receipt['user_id'];identity.bind(uid)
            if engine.store.read_scope(uid,AGENT)[0]!=receipt['memory_generation']:
                raise RuntimeError('manifest invalidated by a later forget operation; do not resurrect data')
            for expected in receipt['records']:
                row=engine.read_memory(user_id=uid,agent_id=AGENT,memory_id=expected['memory_id'])
                if row is None:missing.append(expected['memory_id']);continue
                if row.get('user_id')!=uid or row.get('agent_id')!=AGENT or row.get('id')!=expected['memory_id']:
                    pollution.append('exact-read ownership');binding_mismatch.append(expected['memory_id'])
                if hashlib.sha256(row['memory'].encode('utf8')).hexdigest()!=expected['content_sha256']:hash_mismatch.append(expected['memory_id'])
                for other in USERS:
                    if other!=uid and engine.read_memory(user_id=other,agent_id=AGENT,memory_id=expected['memory_id']) is not None:pollution.append('foreign ID read')
            for query in ('我平时喝什么？','我住在哪里？','user_A喜欢什么？'):
                rows=engine.search_long_term(user_id=uid,agent_id=AGENT,query=query)
                assert rows,'empty retrieval is not counted as successful isolation'
                if any(x.get('user_id')!=uid or x.get('agent_id')!=AGENT for x in rows):pollution.append('search ownership')
            turn=engine.chat(user_id=uid,agent_id=AGENT,text=uid+'_PRIVATE_CURRENT')
            joined='\n'.join(m['content'] for m in turn.prompt_messages)
            for other in USERS:
                if other!=uid and any(t in turn.prompt_messages[0]['content'] for t in USERS[other]):context_leaks.append(uid)
            if turn.user_id!=uid:pollution.append('response binding')
        old=identity.snapshot();identity.bind('user_A' if old.user_id!='user_A' else 'user_B');delivered=[]
        assert not identity.deliver(old,delivered.append,'old private reply') and not delivered
        identity.clear()
        try:identity.snapshot()
        except IdentityRequiredError:checks.append('reconnect_requires_identity')
        else:raise AssertionError('identity was not cleared')
        matches=15-len(set(missing+hash_mismatch+binding_mismatch))
        report={'event_label':event,'reader_pid':os.getpid(),'independent_process':os.getpid()!=manifest['writer_pid'],'boot_id_before':manifest['boot_id_before'],'boot_id_after':boot_id(),'expected_confirmed_records':15,'restored_exact_records':matches,'recovery_rate':matches/15,'missing_ids':missing,'content_mismatch_ids':hash_mismatch,'binding_mismatch_ids':binding_mismatch,'cross_user_pollution_count':len(pollution)+len(context_leaks),'violations':pollution+context_leaks,'checks':checks,'frontend_reconnect':'simulated trusted user_id rebinding, not voice recognition','physical_suspend_tested':False,'physical_power_loss_tested':False,'latency_optimization_or_validation':False,'llm_calls':0,'data_directory':str(path)}
        atomic_json(path/'reports'/('verify-'+str(os.getpid())+'.json'),report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
        if matches!=15 or pollution or context_leaks:raise SystemExit(1)
        return report
    finally:
        engine.close();closed.wait(30)


def owned_scenario(parent,kind):
    path=setup(parent/kind);copy_model_cache(path)
    stderr=(path/'reports/writer.log').open('w',encoding='utf8')
    child=subprocess.Popen([sys.executable,__file__,'_writer','--run-dir',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
    try:
        deadline=time.monotonic()+180;ready=None
        while time.monotonic()<deadline and child.poll() is None:
            if not select.select([child.stdout],[],[],.2)[0]:continue
            line=child.stdout.readline()
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if value.get('ready'):ready=value;break
        if ready is None or ready['pid']!=child.pid:raise RuntimeError('writer never confirmed data; inspect isolated writer.log')
        if kind=='process_sigkill':
            assert child.poll() is None
            child.kill()  # Only this Popen-owned process; never a server/service PID.
        else:child.stdin.write('exit\n');child.stdin.flush()
        rc=child.wait(timeout=40)
        if kind=='process_sigkill':assert rc==-9
        else:assert rc==0
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=15)
        stderr.close()
    result=subprocess.run([sys.executable,__file__,'verify','--run-dir',str(path),'--event',kind],capture_output=True,text=True,timeout=180)
    (path/'reports/verify_stdout.log').write_text(result.stdout,encoding='utf8');(path/'reports/verify_stderr.log').write_text(result.stderr,encoding='utf8')
    if result.returncode:raise RuntimeError('restoration check failed; inspect verify logs')
    report=json.loads(result.stdout.strip().splitlines()[-1]);report['writer_exit_code']=rc
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','prepare-host','verify','_writer']);ap.add_argument('--run-dir',required=True);ap.add_argument('--event',default='operator-unspecified')
    a=ap.parse_args();path=setup(a.run_dir)
    if a.action=='_writer':return write_dataset(path)
    if a.action=='verify':return verify(path,a.event)
    if a.action=='prepare-host':
        report=owned_scenario(path,'host_baseline')
        print(json.dumps({'status':'awaiting_operator_host_event','data_baseline':report,'verify_command':f'{sys.executable} {__file__} verify --run-dir {path/"host_baseline"} --event operator-reported-host-event','note':'No host action performed. Boot-ID change shows a reboot, not proof that it was a power cut.'},ensure_ascii=False));return
    reports=[owned_scenario(path,kind) for kind in ('process_restart','process_sigkill')]
    summary={'scope':'two requested acceptance items only','scenarios':reports,'whole_host_suspend':'not performed','whole_host_power_loss':'not performed','voice_recognition':'not implemented; trusted user_id simulated','latency':'not changed or optimized','all_checks_passed':all(x['recovery_rate']==1 and x['cross_user_pollution_count']==0 for x in reports)}
    atomic_json(path/'reports/summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
