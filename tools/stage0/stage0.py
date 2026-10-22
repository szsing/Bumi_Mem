#!/usr/bin/env python3
"""Stage-0 基础检查：不修改 Bumi 核心代码；除显式 api-smoke 外不发起付费请求。"""
from __future__ import annotations
import argparse, datetime, hashlib, importlib.metadata as metadata, json, os
from pathlib import Path
import platform, socket, sqlite3, subprocess, sys, time, uuid
import urllib.request, urllib.error

REPO = Path(__file__).resolve().parents[2]
ALLOWED_RUNS = (REPO.parent / "mem0-runtime/stage0/runs").resolve()
KEY_NAMES = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "BUMI_CHAT_API_KEY", "BUMI_MEMORY_API_KEY")


def run_path(raw):
    p = Path(raw).expanduser().resolve()
    if p == ALLOWED_RUNS or not p.is_relative_to(ALLOWED_RUNS):
        raise ValueError(f"run directory must be a child of {ALLOWED_RUNS}")
    for name in ("reports", "legacy", "typed", "home", "cache"):
        (p/name).mkdir(parents=True,exist_ok=True)
    return p


def save(run, name, obj):
    p = run/"reports"/name
    if p.exists(): raise FileExistsError(f"refusing to overwrite {p}; use a new run directory/report name")
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf8")
    return p


def safe_command(args):
    try:
        p=subprocess.run(args,cwd=REPO,capture_output=True,text=True,timeout=15)
        return {"exit_code":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr.strip()[:1000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code":None,"stdout":"","error_type":type(exc).__name__}


def clean_env(run):
    env=os.environ.copy()
    for key in KEY_NAMES: env.pop(key,None)
    env.update(MEM0_TELEMETRY="False",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",HOME=str(run/"home"),XDG_CACHE_HOME=str(run/"cache"),MEM0_RUNTIME_ROOT=str(run/"legacy/runtime"),MEM0_DATA_ROOT=str(run/"legacy/data"),MEM0_TMP_ROOT=str(run/"legacy/tmp"),MEM0_CACHE_ROOT=str(run/"cache"),MEM0_HOME_ROOT=str(run/"home"),MEM0_RESULTS_ROOT=str(run/"reports"),MEM0_LOG_ROOT=str(run/"reports/logs"),MEM0_SECRETS_ROOT=str(run/"legacy/empty-secrets"),MEM0_MODEL_ROOT=str(REPO.parent/"models"),MEM0_PYTHON=sys.executable)
    return env


def stats(values):
    import numpy as np
    return {"n":len(values),**{f"p{q}_ms":round(float(np.percentile(values,q)),3) for q in (50,95,99)},"max_ms":round(max(values),3)}


def doctor(run):
    packages={}
    for name in ("pytest","mem0ai","openai","transformers","torch","sentence-transformers","qdrant-client","pydantic","numpy"):
        try: packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError: packages[name]=None
    c=sqlite3.connect(":memory:")
    c.execute("CREATE VIRTUAL TABLE probe USING fts5(text)");c.close()
    ports={}
    for port in (6333,6334):
        try:
            with socket.create_connection(("127.0.0.1",port),timeout=.3): ports[str(port)]="listening"
        except OSError: ports[str(port)]="not listening"
    obj={"captured_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"repo":str(REPO),"python":sys.executable,"python_version":platform.python_version(),"sqlite_version":sqlite3.sqlite_version,"fts5_available":True,"packages":packages,"api_key_presence_only":{k:bool(os.environ.get(k)) for k in KEY_NAMES},"head":safe_command(["git","rev-parse","HEAD"]),"worktree":safe_command(["git","status","--short"]),"submodules":safe_command(["git","submodule","status"]),"gpu":safe_command(["nvidia-smi","--query-gpu=name,memory.total,memory.used,utilization.gpu","--format=csv,noheader"]),"qdrant_ports":ports,"isolation":{"run":str(run),"legacy":str(run/"legacy"),"typed":str(run/"typed")}}
    print(save(run,"environment.json",obj))
    print(json.dumps({"sqlite":sqlite3.sqlite_version,"packages":packages,"qdrant_ports":ports},ensure_ascii=False))


def validate(run):
    sys.path.insert(0,str(REPO))
    from tools.stage0.contracts import SeedCase, ALL_CONTRACTS
    items=json.loads((REPO/"tests/fixtures/stage0/cases.json").read_text(encoding="utf-8-sig"))
    cases=[SeedCase.model_validate(x) for x in items]
    ids=[c.case_id for c in cases]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate case_id")
    for case in cases:
        obs=case.observations+[case.query]
        if len({o.observation_id for o in obs})!=len(obs):raise ValueError(f"duplicate observation in {case.case_id}")
    schemas={cls.__name__:cls.model_json_schema() for cls in ALL_CONTRACTS}
    path=save(run,"contracts.schemas.json",schemas)
    result={"validated_fixture_count":len(cases),"behavioral_tests_run":False,"warning":"Fixture/schema validity does NOT prove sharing, inference, deletion or retrieval behavior.","cases":[{"case_id":c.case_id,"target_phase":c.expected.target_phase} for c in cases]}
    print(save(run,"fixtures_validation.json",result));print(path)


def pytest_run(run, contracts=False):
    name="stage0_contracts" if contracts else "baseline_after"
    temp=run/"legacy"/(name+"-"+uuid.uuid4().hex[:8])
    args=[sys.executable,"-m","pytest","-q","-p","no:cacheprovider","--basetemp",str(temp),"--junitxml",str(run/"reports"/(name+".xml"))]
    args += ["tests/stage0"] if contracts else ["tests/test_async_memory.py", "tests/test_bumi_memory.py", "tests/test_consolidation_and_context.py", "tests/test_memory_adapter_and_migration.py", "tests/test_provider_config.py"]
    log=run/"reports"/(name+".log")
    if log.exists():raise FileExistsError(log)
    with log.open("w",encoding="utf8") as stream:
        rc=subprocess.run(args,cwd=REPO,env=clean_env(run),stdout=stream,stderr=subprocess.STDOUT).returncode
    print(log.read_text(encoding="utf8")[-4000:]);return rc


def storage_probe(run):
    import numpy as np
    from qdrant_client import QdrantClient, models
    db=run/"legacy"/("storage_probe_"+uuid.uuid4().hex[:8]+".sqlite3")
    c=sqlite3.connect(db)
    mode=c.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    c.execute("PRAGMA synchronous=FULL")
    c.execute("CREATE TABLE probe_docs(id INTEGER PRIMARY KEY, scope TEXT, body TEXT)")
    c.execute("CREATE VIRTUAL TABLE probe_fts USING fts5(body)")
    for i in range(1000):
        text=f"家人 今晚 安排 电影 测试 {i}"
        c.execute("INSERT INTO probe_docs VALUES(?,?,?)",(i,"synthetic",text))
        c.execute("INSERT INTO probe_fts(rowid,body) VALUES(?,?)",(i+1,text))
    c.commit()
    sql=[];fts=[];commits=[]
    for i in range(50):
        t=time.perf_counter();c.execute("SELECT * FROM probe_docs WHERE scope=? LIMIT 24",("synthetic",)).fetchall();sql.append((time.perf_counter()-t)*1000)
        t=time.perf_counter();c.execute("SELECT rowid FROM probe_fts WHERE probe_fts MATCH ? ORDER BY bm25(probe_fts) LIMIT 24",("电影",)).fetchall();fts.append((time.perf_counter()-t)*1000)
        t=time.perf_counter();c.execute("INSERT INTO probe_docs VALUES(?,?,?)",(1000+i,"synthetic","新增 合成记录"));c.commit();commits.append((time.perf_counter()-t)*1000)
    c.close()
    rng=np.random.default_rng(7);vectors=rng.standard_normal((1000,1024)).astype("float32");vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    q=QdrantClient(location=":memory:")
    try:
        q.create_collection("stage0_synthetic",vectors_config=models.VectorParams(size=1024,distance=models.Distance.COSINE))
        q.upsert("stage0_synthetic",points=models.Batch(ids=list(range(1000)),vectors=vectors.tolist(),payloads=[{"scope":"synthetic"} for _ in range(1000)]))
        filt=models.Filter(must=[models.FieldCondition(key="scope",match=models.MatchValue(value="synthetic"))])
        timings=[]
        for i in range(55):
            t=time.perf_counter();r=q.query_points("stage0_synthetic",query=vectors[i].tolist(),query_filter=filt,limit=24)
            assert len(r.points)==24
            if i>=5:timings.append((time.perf_counter()-t)*1000)
    finally:q.close()
    obj={"data":"1000 synthetic rows and random normalized vectors; not semantic relevance evidence","sqlite_path":str(db),"sqlite_journal_mode":mode,"sqlite_synchronous":"FULL","sql_query":stats(sql),"fts_bm25":stats(fts),"durable_single_row_commit":stats(commits),"vector_backend":"qdrant local in-memory client; NOT planned Qdrant Server/HNSW benchmark","vector_query_preencoded":stats(timings),"full_bundle_latency_measured":False}
    print(save(run,"storage_probe.json",obj));print(json.dumps(obj,ensure_ascii=False))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None


def api_smoke(run, model):
    key=os.environ.get("DEEPSEEK_API_KEY","")
    if not key: print("DeepSeek key missing. Run scripts/run_stage0_deepseek_smoke.sh interactively.",file=sys.stderr);return 2
    # No redirects, no automatic retries, no private data, no persisted key/header.
    body={"model":model,"messages":[{"role":"user","content":"Reply with STAGE0_OK only."}],"thinking":{"type":"disabled"},"max_tokens":64,"stream":False}
    req=urllib.request.Request("https://api.deepseek.com/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+key},method="POST")
    t=time.perf_counter()
    try:
        with urllib.request.build_opener(NoRedirect).open(req,timeout=45) as response: payload=json.load(response)
    except urllib.error.HTTPError as exc:
        print(json.dumps({"api_smoke":"failed","http_status":exc.code,"details":"response body withheld to avoid accidental credential logging"}));return 1
    except Exception as exc:
        print(json.dumps({"api_smoke":"failed","error_type":type(exc).__name__}));return 1
    choices=payload.get("choices") or [{}]
    content=(choices[0].get("message") or {}).get("content")
    answer=(content if isinstance(content,str) else "").replace(key,"[REDACTED]")[:500]
    obj={"requested_model":model,"reported_model":payload.get("model"),"endpoint":"api.deepseek.com/chat/completions","thinking":"disabled","max_tokens":64,"automatic_retries":0,"elapsed_ms":round((time.perf_counter()-t)*1000,2),"answer":answer,"marker_ok":answer.strip()=="STAGE0_OK","usage":payload.get("usage"),"scope":"endpoint only; NOT Bumi memory extraction/retrieval validation"}
    print(save(run,"deepseek_smoke.json",obj));print(json.dumps(obj,ensure_ascii=False));return 0 if obj["marker_ok"] else 1


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-dir",required=True)
    ap.add_argument("command",choices=("doctor","validate","baseline","contract-tests","storage-probe","api-smoke"));ap.add_argument("--model",default="deepseek-flash")
    args=ap.parse_args();run=run_path(args.run_dir)
    os.environ.update(MEM0_TELEMETRY="False",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    if args.command=="doctor":doctor(run)
    elif args.command=="validate":validate(run)
    elif args.command=="baseline":return pytest_run(run)
    elif args.command=="contract-tests":return pytest_run(run,True)
    elif args.command=="storage-probe":storage_probe(run)
    else:return api_smoke(run,args.model)
    return 0

if __name__=="__main__":raise SystemExit(main())