#!/usr/bin/env python3
"""Real, isolated Mem0 write/reopen/search benchmark. No chat answers or GPU repair."""
from __future__ import annotations
import argparse, datetime, functools, hashlib, importlib, importlib.metadata
import json, logging, os, random, stat, sys, threading, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNS=(ROOT.parent/'mem0-runtime/stage0/runs').resolve()
KEYFILE=ROOT.parent/'mem0-runtime/stage0/secrets/deepseek_api_key'
MODEL_PATH=ROOT.parent/'models/Qwen3-Embedding-0.6B'
API_MODEL='deepseek-flash'
INSTRUCTIONS='''这是完全虚构的检索测试用户，以下十项自述均供本次长期记忆测试。\n1. 我叫林川。\n2. 我的职业是嵌入式软件工程师。\n3. 我目前住在苏州。\n4. 我长期关注机器人与 AI Agent 技术。\n5. 我买咖啡时通常选择无糖热拿铁。\n6. 我希望回答先给结论，再列具体步骤。\n7. 我每周三晚上练钢琴。\n8. 我计划在 2026-09-19 上午去金鸡湖骑行。\n9. 我不喜欢食物里放香菜。\n10. 我周末喜欢徒步。'''
QUERIES=[
 {'id':'Q01','query':'我是做什么工作的？','groups':[['嵌入式']]},
 {'id':'Q02','query':'我平时生活在哪座城市？','groups':[['苏州']]},
 {'id':'Q03','query':'帮我买咖啡的话，我通常选哪一种？','groups':[['无糖','不加糖'],['拿铁']]},
 {'id':'Q04','query':'向我解释方案时，回答最好按什么顺序组织？','groups':[['结论'],['步骤']]},
 {'id':'Q05','query':'周三晚上我一般安排什么活动？','groups':[['钢琴']]},
 {'id':'Q06','query':'2026年9月19日上午我有什么计划？','groups':[['金鸡湖'],['骑行']]},
 {'id':'Q07','query':'给我做饭时，哪种配料最好不要放？','groups':[['香菜']]},
 {'id':'Q08','query':'我长期关注哪些技术方向？','groups':[['机器人'],['AI Agent','AI代理','AI 代理','智能体']]},
 {'id':'Q09','query':'周末休闲活动我有什么偏好？','groups':[['徒步']]},
 {'id':'Q10','query':'我叫什么名字？','groups':[['林川']]},
 {'id':'Q11','query':'我最喜欢的电影是哪一部？','groups':[],'negative_control':True},
 {'id':'Q12','query':'我的宠物叫什么名字？','groups':[],'negative_control':True},
]


def ms():return time.perf_counter_ns()/1_000_000

def stats(values):
    import numpy as np
    return {'n':len(values),**{f'p{p}_ms':round(float(np.percentile(values,p)),3) for p in (50,95,99)},'mean_ms':round(float(np.mean(values)),3),'max_ms':round(max(values),3)}

def dump(path,obj):
    if path.exists():raise FileExistsError('Refusing to overwrite an existing benchmark artifact')
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8')

def setup(raw):
    run=Path(raw).resolve()
    if run==RUNS or not run.is_relative_to(RUNS):raise ValueError('Isolated stage0 run directory required')
    for p in (run/'reports',run/'home',run/'cache',run/'tmp'):p.mkdir(parents=True,exist_ok=True)
    for name in ('OPENAI_API_KEY','OPENROUTER_API_KEY','OPENROUTER_API_BASE','OPENAI_BASE_URL','DEEPSEEK_API_KEY','BUMI_CHAT_API_KEY','BUMI_MEMORY_API_KEY'):
        os.environ.pop(name,None)
    os.environ.update(MEM0_TELEMETRY='False',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HOME=str(run/'home'),XDG_CACHE_HOME=str(run/'cache'),TMPDIR=str(run/'tmp'),MEM0_RUNTIME_ROOT=str(run/'legacy_runtime'),MEM0_MODEL_ROOT=str(MODEL_PATH.parent))
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/mem0-new')]
    logging.disable(logging.CRITICAL)
    return run


def read_secret():
    fd=os.open(KEYFILE,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'r',encoding='utf8') as f:
        s=os.fstat(f.fileno())
        if s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o600 or not stat.S_ISREG(s.st_mode):raise PermissionError('Dedicated key file must be owned by this user and mode 600')
        if not 0<s.st_size<=4096:raise ValueError('Invalid key file size')
        key=f.read().strip()
    if not key or any(c.isspace() for c in key):raise ValueError('Invalid key format; content withheld')
    return key


class ApiBudget:
    def __init__(self,allow):self.allow=allow;self.calls=[];self.attempts=0
    def bind(self,memory,key):
        from openai import OpenAI
        import httpx
        old=memory.llm.client
        # Explicit vendor configuration in this test process only. SDK extraction
        # prompt, JSON parsing, writes and searches are not replaced by mocks.
        self.client=OpenAI(api_key=key,base_url='https://api.deepseek.com/v1',max_retries=0,timeout=60,http_client=httpx.Client(follow_redirects=False,timeout=60))
        memory.llm.client=self.client
        old.close()
        original=self.client.chat.completions.create
        def guarded(*args,**kwargs):
            self.attempts+=1
            if not self.allow:raise RuntimeError('LLM calls are prohibited during retrieval')
            if len(self.calls)>=2:raise RuntimeError('Extraction request budget exhausted')
            kwargs['model']=API_MODEL;kwargs['stream']=False
            kwargs['max_tokens']=min(int(kwargs.get('max_tokens',2048)),2048)
            kwargs.pop('max_completion_tokens',None)
            kwargs['extra_body']={**(kwargs.get('extra_body') or {}),'thinking':{'type':'disabled'}}
            row={'model_requested':API_MODEL,'max_tokens':kwargs['max_tokens'],'thinking':'disabled','automatic_retries':0}
            self.calls.append(row);t=ms()
            try:
                response=original(*args,**kwargs)
                row.update(elapsed_ms=ms()-t,model_reported=response.model,usage=response.usage.model_dump() if response.usage else None,finish_reason=response.choices[0].finish_reason)
                content=response.choices[0].message.content or ''
                row['extraction_response']=content.replace(key,'[REDACTED]')[:16000]
                return response
            except Exception as exc:
                row.update(elapsed_ms=ms()-t,error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
                raise
        self.client.chat.completions.create=guarded


class Backend:
    def __init__(self,run,user,agent,allow_api=False,closed_event=None):
        from bumi_mem.long_term import Mem0AdapterConfig,Mem0LongTermMemory
        from bumi_mem.providers import ProviderConfig
        self.run=run;self.user=user;self.agent=agent;self.rows=[];self.current=None;self.closed_event=closed_event
        self.budget=ApiBudget(allow_api)
        key=read_secret() if allow_api else 'unused-readonly-placeholder'
        t=ms()
        self.adapter=Mem0LongTermMemory(ProviderConfig('deepseek',API_MODEL,'https://api.deepseek.com/v1',key,2048,0.0),Mem0AdapterConfig(data_dir=run/'legacy_mem0',embed_model=MODEL_PATH,embed_device='cpu',collection_name='stage0_real_ltm',embedding_dims=1024,memory_max_tokens=2048))
        self.budget.bind(self.adapter.memory,key)
        self.init_ms=ms()-t
        self.lock=threading.Lock();self.in_entity=False
        self._wrap(self.adapter.memory.embedding_model,'embed','single_embedding_ms')
        self._wrap(self.adapter.memory.embedding_model,'embed_batch','batch_embedding_ms')
        self._wrap(self.adapter.memory.vector_store,'search','dense_store_ms')
        self._wrap(self.adapter.memory.vector_store,'keyword_search','keyword_store_ms',capture_count=True)
        self._wrap(self.adapter.memory,'_compute_entity_boosts','entity_boost_ms')
        core=importlib.import_module('mem0.memory.main')
        self._wrap(core,'lemmatize_for_bm25','lemmatize_ms')
        self._wrap(core,'extract_entities','entity_extract_ms',capture_count=True)

    def _wrap(self,obj,name,label,capture_count=False):
        original=getattr(obj,name)
        @functools.wraps(original)
        def wrapped(*a,**kw):
            trace=self.current;t=ms();value=None
            try:value=original(*a,**kw);return value
            finally:
                if trace is not None:
                    record={'ms':ms()-t}
                    if capture_count:record['returned_count']=None if value is None else len(value)
                    with self.lock:trace['spans'].setdefault(label,[]).append(record)
        setattr(obj,name,wrapped)

    def search(self,**kwargs):
        trace={'query':kwargs['query'],'spans':{}};self.current=trace;t=ms()
        try:
            result=self.adapter.search(**kwargs)
            trace['results']=result
            return result
        finally:
            trace['backend_ms']=ms()-t;self.current=None;self.rows.append(trace)

    def get_all(self,**kwargs):return self.adapter.get_all(**kwargs)
    def close(self):
        try:
            entity=getattr(self.adapter.memory,'_entity_store',None)
            if entity is not None:
                client=getattr(entity,'client',None)
                if client is not None:client.close()
            self.adapter.close();self.budget.client.close()
        finally:
            if self.closed_event:self.closed_event.set()

    def features(self):
        store=self.adapter.memory.vector_store
        nlp=importlib.import_module('mem0.utils.spacy_models')
        model=self.adapter.memory.embedding_model.model
        return {'vector_backend':'Qdrant local persistent, not Qdrant Server','bm25_slot':getattr(store,'_has_bm25_slot',None),'bm25_encoder_loaded':bool(getattr(store,'_bm25_encoder',None)),'spacy_lemma_loaded':getattr(nlp,'_nlp_lemma',None) is not None,'spacy_full_loaded':getattr(nlp,'_nlp_full',None) is not None,'embedding_default_prompt_name':getattr(model,'default_prompt_name',None),'embedding_device':str(model.device),'embedding_dimension':model.get_sentence_embedding_dimension(),'entity_store_initialized':getattr(self.adapter.memory,'_entity_store',None) is not None}


def relevant(groups,results):
    joined='\n'.join(str(x.get('memory','')) for x in results).casefold()
    return bool(groups) and all(any(term.casefold() in joined for term in group) for group in groups)


def writer(run):
    if (run/'write_started.json').exists():raise RuntimeError('Existing write attempt; do not blindly replay a possible write')
    user='synthetic_linchuan_'+run.name;agent='stage0_retrieval'
    dump(run/'write_started.json',{'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'user_id':user,'agent_id':agent,'data':'synthetic only'})
    dump(run/'source_session.json',{'messages':[{'role':'user','content':INSTRUCTIONS}],'queries':QUERIES})
    b=None
    try:
        b=Backend(run,user,agent,True)
        from bumi_mem.prompts import MEMORY_EXTRACTION_INSTRUCTIONS
        t=ms()
        events=b.adapter.add_session(user_id=user,agent_id=agent,messages=[{'role':'user','content':INSTRUCTIONS}],metadata={'source':'stage0_synthetic_session','session_id':run.name,'consolidation_id':run.name+'-chunk0','memory_generation':0,'synthetic':True},prompt=MEMORY_EXTRACTION_INSTRUCTIONS)
        write_ms=ms()-t
        all_mem=b.get_all(user_id=user,agent_id=agent)
        if not all_mem:raise RuntimeError('Extraction persisted no memories; retrieval test cannot proceed')
        assert all(m.get('user_id')==user and m.get('agent_id')==agent for m in all_mem)
        report={'user_id':user,'agent_id':agent,'memory_count':len(all_mem),'memories':all_mem,'events':events,'init_ms':b.init_ms,'write_with_readback_ms':write_ms,'api_calls':b.budget.calls,'api_request_attempts':b.budget.attempts,'features':b.features(),'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'notes':['Real DeepSeek -> SDK extraction -> actual persistent Qdrant write/readback.','Thinking explicitly disabled for extraction in this isolated process.','No chat-answer generation.']}
        dump(run/'reports/write.json',report)
        print(json.dumps({'write_ok':True,'memory_count':len(all_mem),'api_calls':len(b.budget.calls),'write_ms':round(write_ms,2)},ensure_ascii=False),flush=True)
        for m in all_mem:print(json.dumps({'id':m['id'],'memory':m['memory']},ensure_ascii=False),flush=True)
    except Exception as exc:
        cause=exc;status=None
        for _ in range(8):
            if cause is None:break
            status=status or getattr(cause,'status_code',None);cause=getattr(cause,'__cause__',None)
        dump(run/'reports/write_error.json',{'error_type':type(exc).__name__,'http_status':status,'api_calls':b.budget.calls if b else [],'no_automatic_replay':True})
        print(json.dumps({'write_ok':False,'error_type':type(exc).__name__,'http_status':status}),flush=True)
        raise SystemExit(1)
    finally:
        if b:b.close()


def reader(run,repeats,wrapped=False):
    written=json.loads((run/'reports/write.json').read_text());user=written['user_id'];agent=written['agent_id']
    closed=threading.Event();holder={};executor=None
    def make():holder['backend']=Backend(run,user,agent,False,closed);return holder['backend']
    t=ms()
    if wrapped:
        from bumi_mem.memory_access import SerialMemoryAccess
        executor=SerialMemoryAccess(make,read_timeout=120.0)
        persisted=executor.get_all(user_id=user,agent_id=agent);b=holder['backend']; search=executor.search
    else:
        b=make();persisted=b.get_all(user_id=user,agent_id=agent);search=b.search
    reopened_ms=ms()-t
    if {m['id'] for m in persisted}!={m['id'] for m in written['memories']}:raise RuntimeError('Persistence verification failed after process restart')
    def ask(q):return search(user_id=user,agent_id=agent,query=q['query'],top_k=5,threshold=.1)
    t=ms();first=ask(QUERIES[0]);first_ms=ms()-t
    # All query shapes prewarmed once. A newly initialized process is NOT an OS-cold disk.
    for q in QUERIES:ask(q)
    if wrapped:executor.read_timeout=.2
    measured=[]; schedule=[q for q in QUERIES for _ in range(repeats)];random.Random(17).shuffle(schedule)
    for i,q in enumerate(schedule):
        t=ms();error=None;result=[]
        try:result=ask(q)
        except Exception as exc:error=type(exc).__name__
        elapsed=ms()-t
        row={'query_id':q['id'],'elapsed_ms':elapsed,'error_type':error,'results':result}
        measured.append(row)
        if error and wrapped:
            # Drain owned queued work before the next trial; do not misattribute
            # an old in-flight read to the following query's measured latency.
            executor.read_timeout=120;executor.get_all(user_id=user,agent_id=agent);executor.read_timeout=.2
        if (i+1)%100==0:print(json.dumps({'mode':'wrapped' if wrapped else 'backend','completed':i+1,'total':len(schedule)}),flush=True)
    wrong=search(user_id=user+'_WRONG',agent_id=agent,query='我住在哪座城市？',top_k=5,threshold=.1)
    per=[]
    for q in QUERIES:
        rows=[r for r in measured if r['query_id']==q['id']];ok=[r for r in rows if r['error_type'] is None]
        sample=ok[0]['results'] if ok else []
        per.append({**q,'latency':stats([r['elapsed_ms'] for r in rows]),'errors':len(rows)-len(ok),'stored_evidence_matches_markers':relevant(q['groups'],persisted) if q['groups'] else None,'top1_marker_hit':relevant(q['groups'],sample[:1]) if q['groups'] else None,'top5_marker_hit':relevant(q['groups'],sample) if q['groups'] else None,'result_count':len(sample),'sample_results':sample,'stable_result_ids':len({tuple(m['id'] for m in r['results']) for r in ok})<=1})
    features=b.features();api_attempts=b.budget.attempts
    versions={}
    for pkg in ['mem0ai','openai','torch','transformers','sentence-transformers','qdrant-client','spacy','fastembed']:
        try:versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:versions[pkg]=None
    # Preserve every raw timed result plus all nested SDK spans; nested spans are
    # not added to create a fictitious end-to-end total.
    report={'mode':'Bumi SerialMemoryAccess 200ms deadline' if wrapped else 'Bumi Mem0LongTermMemory.search backend','date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repeats_per_query':repeats,'query_count':len(QUERIES),'total_measured_searches':len(measured),'overall':stats([r['elapsed_ms'] for r in measured]),'errors':sum(r['error_type'] is not None for r in measured),'memory_count':len(persisted),'same_persisted_ids_after_reopen':True,'backend_initialization_ms':b.init_ms,'reopen_and_get_all_ms':reopened_ms,'first_search_ms':first_ms,'first_search_results':first,'per_query':per,'features':features,'versions':versions,'wrong_user_result_count':len(wrong),'llm_request_attempts_during_read':api_attempts,'full_memory_bundle_measured':False,'timed_rows':measured,'sdk_spans':b.rows,'notes':['New independent read-only process reopened persistent memories.','No answer-generation API calls. Read-time LLM calls prohibited.','Current SDK internally preprocesses words/entities, overfetches candidates and may use BM25/entity boosts; rerank=False does not mean a bare vector DB query.','Marker checks are transparent toy-evidence checks, not a standard benchmark or answer accuracy.','Small synthetic database, short Chinese queries; not large-scale or full MemoryBundle P95.']}
    dump(run/'reports'/('wrapped_search.json' if wrapped else 'backend_search.json'),report)
    print(json.dumps({'mode':report['mode'],'overall':report['overall'],'errors':report['errors'],'llm_calls_during_read':api_attempts,'top5_hits':sum(x['top5_marker_hit'] is True for x in per),'answerable_queries':sum(bool(q['groups']) for q in QUERIES)},ensure_ascii=False),flush=True)
    if executor:
        executor.close()
        if not closed.wait(30):raise RuntimeError('Owned memory worker did not close')
    else:b.close()
    if api_attempts or wrong:raise RuntimeError('Read-time API or scope invariant violated')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['write','read','wrapped']);ap.add_argument('--run-dir',required=True);ap.add_argument('--repeats',type=int,default=50)
    args=ap.parse_args()
    if not 1<=args.repeats<=100:ap.error('repeats must be in [1,100]')
    run=setup(args.run_dir)
    if args.phase=='write':writer(run)
    else:reader(run,args.repeats,args.phase=='wrapped')

if __name__=='__main__':main()
