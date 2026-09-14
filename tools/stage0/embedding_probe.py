#!/usr/bin/env python3
"""Offline component timing, NOT Memory Bundle latency or retrieval-quality evidence."""
from __future__ import annotations

import argparse, concurrent.futures, hashlib, importlib.metadata, json, os
from pathlib import Path
import statistics, threading, time

START = time.perf_counter()
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", MEM0_TELEMETRY="False")


def percentile(values, q):
    values = sorted(values)
    p = (len(values)-1)*q
    a = int(p); b = min(a+1, len(values)-1)
    return values[a] + (values[b]-values[a])*(p-a)


def summary(values):
    return {"n":len(values), "p50_ms":round(percentile(values,.5),3), "p95_ms":round(percentile(values,.95),3), "p99_ms":round(percentile(values,.99),3), "max_ms":round(max(values),3)}


def safe_output(path):
    path = path.expanduser().resolve()
    allowed = Path(__file__).resolve().parents[3] / "mem0-runtime/stage0/runs"
    # The repository is PROJECT/tools/stage0/file; PROJECT's parent owns runtime.
    if not path.is_relative_to(allowed.resolve()):
        raise ValueError(f"probe output must stay under {allowed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("use a new report path; existing measurements are not overwritten")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", choices=("cpu","cuda"), required=True)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--contention-iterations", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.iterations < 5 or args.warmup < 1 or args.contention_iterations < 1:
        ap.error("iterations>=5, warmup>=1 and contention-iterations>=1 required")
    out = safe_output(args.output)
    run = out.parent.parent
    for directory in (run/"home", run/"cache", run/"cache/cuda"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(HOME=str(run/"home"), XDG_CACHE_HOME=str(run/"cache"), HF_HOME=str(run/"cache/hf"), TORCH_HOME=str(run/"cache/torch"), CUDA_CACHE_PATH=str(run/"cache/cuda"))
    model_path = Path(args.model_path).expanduser().resolve()
    if not (model_path/"config.json").is_file():
        raise FileNotFoundError("local model config missing; downloads are disabled")
    import torch
    from transformers import AutoModel, AutoTokenizer
    import torch.nn.functional as F
    if args.threads: torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("this interpreter has no working CUDA")
        free, total = torch.cuda.mem_get_info()
        if free < 3*1024**3: raise RuntimeError("less than 3GiB free GPU memory; refusing probe")
        torch.cuda.set_per_process_memory_fraction(.25)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    prefix = "Instruct: Given a conversation, retrieve relevant facts, events, and interaction preferences.\nQuery: "
    text = prefix + "今晚家人计划外出，想确认一起看电影的时间。"*180
    state = {}

    def load():
        t = time.perf_counter()
        state["tokenizer"] = AutoTokenizer.from_pretrained(str(model_path), padding_side="left", local_files_only=True, trust_remote_code=False)
        state["model"] = AutoModel.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False, torch_dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
        if args.device == "cuda": torch.cuda.synchronize()
        return {"load_ms":(time.perf_counter()-t)*1000, "script_start_to_ready_ms":(time.perf_counter()-START)*1000}

    def encode(cap, batch, submitted_at, started=None):
        t0=time.perf_counter()
        if started is not None: started.set()
        tok=state["tokenizer"]([text]*batch, max_length=cap, truncation=True, padding=True, return_tensors="pt")
        actual=int(tok["input_ids"].shape[1])
        if actual != cap: raise RuntimeError(f"expected {cap} encoded tokens, got {actual}")
        tok={k:v.to(args.device) for k,v in tok.items()}
        if args.device == "cuda": torch.cuda.synchronize()
        t1=time.perf_counter()
        with torch.inference_mode():
            hidden=state["model"](**tok, use_cache=False).last_hidden_state
            # Left padding: final position is the final non-padding token.
            vectors=F.normalize(hidden[:, -1].float(), p=2, dim=1).cpu().numpy()
        if args.device == "cuda": torch.cuda.synchronize()
        t2=time.perf_counter()
        import numpy as np
        if vectors.shape != (batch,1024) or not np.isfinite(vectors).all():
            raise RuntimeError("invalid embedding shape or non-finite values")
        if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
            raise RuntimeError("embedding norm check failed")
        return {"tokens":actual,"batch":batch,"queue_ms":(t0-submitted_at)*1000,"tokenize_transfer_ms":(t1-t0)*1000,"forward_pool_copy_ms":(t2-t1)*1000,"worker_ms":(t2-t0)*1000,"submission_to_result_ms":(time.perf_counter()-submitted_at)*1000}

    result={"probe_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"measurement_kind":"embedding_component_only", "does_not_measure":"full MemoryBundle, live QdrantServer, retrieval quality or API latency", "os_cache_flushed":False, "device":args.device,"dtype":str(dtype),"python_executable":os.sys.executable,"torch_threads":torch.get_num_threads(),"cuda":torch.version.cuda,"model_path":str(model_path),"dimensions":1024,"query_instruction":prefix,"attention":"sdpa","use_cache":False,"versions":{},"isolated":[],"contention":{}}
    for pkg in ("torch","transformers","numpy","sentence-transformers"):
        try: result["versions"][pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: result["versions"][pkg]=None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        result["cold_process_load"] = ex.submit(load).result()
        for cap in (32,128,256):
            for _ in range(args.warmup): ex.submit(encode,cap,1,time.perf_counter()).result()
            rows=[ex.submit(encode,cap,1,time.perf_counter()).result() for _ in range(args.iterations)]
            entry={"encoded_tokens_including_instruction":cap,"rows":rows,"summary":summary([r["submission_to_result_ms"] for r in rows])}
            result["isolated"].append(entry)
            print(json.dumps({"device":args.device,"tokens":cap,**entry["summary"]}),flush=True)
        # Deliberately non-preemptive FIFO contention: one batch-4 indexing job
        # is already running when the query is submitted. Not the future scheduler.
        ex.submit(encode,256,4,time.perf_counter()).result()
        rows=[]
        for _ in range(args.contention_iterations):
            begun=threading.Event()
            bg=ex.submit(encode,256,4,time.perf_counter(),begun)
            if not begun.wait(10): raise RuntimeError("index probe did not start")
            query=ex.submit(encode,128,1,time.perf_counter())
            bg.result();rows.append(query.result())
        result["contention"]={"scenario":"query behind already-running batch4 x 256-token indexing in one FIFO worker","rows":rows,"summary":summary([r["submission_to_result_ms"] for r in rows]),"queue_summary":summary([r["queue_ms"] for r in rows])}
    if args.device == "cuda": result["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
    result["model_file_fingerprints"]={}
    for name in ("config.json","tokenizer_config.json","model.safetensors"):
        path=model_path/name
        if not path.exists():continue
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(4*1024**2),b""):h.update(block)
        result["model_file_fingerprints"][name]={"bytes":path.stat().st_size,"sha256":h.hexdigest()}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"output":str(out),"contention":result["contention"]["summary"]}),flush=True)

if __name__ == "__main__": main()
