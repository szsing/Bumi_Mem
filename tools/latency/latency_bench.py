#!/usr/bin/env python3
"""Bumi_Mem 时延基准：分阶段测量写入、转向量、语义检索、BM25、融合与前台召回。

计时点是真实调用链上的函数，不重新实现检索逻辑：
    mem0.Memory.search
      ├── lemmatize_for_bm25        (spaCy 词形还原)
      ├── extract_entities          (spaCy 实体抽取)
      ├── embedding_model.embed     (Qwen3-Embedding-0.6B 转向量)
      ├── vector_store.search       (Qdrant 语义检索)
      ├── vector_store._encode_bm25 (fastembed Qdrant/bm25 转向量)
      ├── vector_store.keyword_search (Qdrant BM25 检索)
      ├── _compute_entity_boosts    (实体加成：embed_batch + 实体库检索)
      └── score_and_rank            (融合排序，用总时长减去上面各项估算)

用法：
    bumi-latency-bench                        # 默认 300 条语料、每查询 20 次
    bumi-latency-bench --corpus 2000 --repeats 30
    bumi-latency-bench --with-engine           # 额外测引擎前台召回（0.2s 截止）
    bumi-latency-bench --with-chat             # 额外测端到端（含 DeepSeek 调用）
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPO = "/data1/zhangran/git/Bumi_Mem"
DEFAULT_QUERIES = [
    "咖啡",
    "我住在哪个城市？",
    "我早上几点起床？",
    "推荐一家安静、适合工作的咖啡馆，我下午三点到五点有空",
    "我朋友小王最近在忙什么？",
    "机械键盘",
    "What coffee does the user like?",
    "我周末想去西湖边跑步，顺便看看天气怎么样",
]


def default_repo() -> str:
    """优先环境变量；脚本被放进仓库 tools/ 下时按 pyproject 反推仓库根。"""
    if os.environ.get("BUMI_MEM_REPO"):
        return os.environ["BUMI_MEM_REPO"]
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "bumi_mem").is_dir():
            return str(parent)
    return DEFAULT_REPO


# --------------------------------------------------------------------------- #
# 运行环境
# --------------------------------------------------------------------------- #
def bootstrap() -> dict[str, str]:
    """解析仓库/工作区路径，必要时用项目 venv 重新执行自己。"""
    repo = Path(default_repo()).expanduser()
    workspace = Path(os.environ.get("MEM0_WORKSPACE_ROOT") or repo.parent.parent).expanduser()
    env = dict(os.environ)
    env["MEM0_WORKSPACE_ROOT"] = str(workspace)
    env.setdefault("MEM0_RUNTIME_ROOT", str(workspace / "mem0-runtime"))
    env.setdefault("MEM0_MODEL_ROOT", str(workspace / "models"))
    env.setdefault("MEM0_TELEMETRY", "False")

    python = env.get("MEM0_PYTHON")
    if not python or not os.access(python, os.X_OK):
        for candidate in (
            workspace / ".venvs" / "bumi" / "bin" / "python",
            workspace / ".venv" / "bin" / "python",
            workspace / "mem0-runtime" / "venv" / "bin" / "python",
        ):
            if candidate.exists() and os.access(candidate, os.X_OK):
                python = str(candidate)
                break
    # The venv interpreter is a symlink chain back to the system python, so
    # sys.executable cannot be compared directly. Re-exec exactly once instead.
    marker = env.get("BUMI_LATENCY_BENCH_PYTHON")
    if python and marker != python:
        env["MEM0_PYTHON"] = python
        env["BUMI_LATENCY_BENCH_PYTHON"] = python
        os.execve(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    env["MEM0_PYTHON"] = python or sys.executable
    for key, value in env.items():
        os.environ[key] = value
    return {"repo": str(repo), "workspace": str(workspace), "python": env["MEM0_PYTHON"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bumi_Mem 分阶段时延基准")
    parser.add_argument("--corpus", type=int, default=300, help="合成记忆条数（默认 300）")
    parser.add_argument("--repeats", type=int, default=20, help="每个查询重复次数（默认 20）")
    parser.add_argument("--top-k", type=int, default=5, help="检索 top_k（默认 5，与 CLI 一致）")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--embed-device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default=None, help="隔离的数据目录（默认 <runtime>/latency-bench）")
    parser.add_argument("--collection", default="bumi_latency_bench",
                        help="Qdrant 集合名；不同规模/后端用不同集合，避免互相污染")
    parser.add_argument("--queries", default=None, help="查询文件，每行一条")
    parser.add_argument("--with-engine", action="store_true", help="额外测引擎前台召回路径")
    parser.add_argument("--assembly", action="store_true",
                        help="只做前台一轮分解（检索 vs 串行拼装），复用已有语料，需独立进程运行")
    parser.add_argument("--scale-probe", default=None,
                        help="逗号分隔的向量条数，如 300,1000,3000,10000；随机向量测本地 Qdrant 稠密检索的规模伸缩")
    parser.add_argument("--with-chat", action="store_true", help="额外测端到端（含回答模型）")
    parser.add_argument("--read-timeout", type=float, default=0.2, help="前台检索截止（默认 0.2s）")
    parser.add_argument("--assembly-read-timeout", type=float, default=5.0,
                        help="分解计时时允许检索跑完的等待上限（默认 5s，只用于把检索与串行拼装分开）")
    parser.add_argument("--json", dest="json_out", default=None, help="原始数据导出路径")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# 计时工具
# --------------------------------------------------------------------------- #
class Recorder:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)

    @contextlib.contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append((time.perf_counter() - started) * 1000)

    def record(self, name: str, ms: float) -> None:
        self.samples[name].append(ms)

    def reset(self) -> None:
        self.samples.clear()
        self.counters.clear()


def wrap_function(module, name: str, recorder: Recorder, stage: str):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        with recorder.stage(stage):
            return original(*args, **kwargs)

    setattr(module, name, wrapper)
    return original


def wrap_method(obj, name: str, recorder: Recorder, stage: str):
    original = getattr(obj, name)

    def wrapper(*args, **kwargs):
        with recorder.stage(stage):
            return original(*args, **kwargs)

    setattr(obj, name, wrapper)
    return original


def pct(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(ratio * (len(ordered) - 1)))))
    return ordered[index]


def summarize(values: list[float]) -> tuple[float, float, float, float]:
    return statistics.fmean(values), statistics.median(values), pct(values, 0.9), max(values)


def fmt_row(label: str, values: list[float], width: int = 26) -> str:
    if not values:
        return f"{label:<{width}} {'-':>9} {'-':>9} {'-':>9} {'-':>9}"
    mean, median, p90, worst = summarize(values)
    return f"{label:<{width}} {mean:>9.1f} {median:>9.1f} {p90:>9.1f} {worst:>9.1f}"


def table_header(title: str, width: int = 26) -> None:
    print(f"\n{title}")
    print(f"{'阶段':<{width}} {'均值ms':>9} {'中位ms':>9} {'p90ms':>9} {'最大ms':>9}")
    print("-" * (width + 40))


# --------------------------------------------------------------------------- #
# 语料与查询
# --------------------------------------------------------------------------- #
SUBJECTS = ["我", "我朋友小王", "我妈妈", "同事小李", "我妹妹", "楼下的张阿姨", "我大学同学"]
VERBS = ["喜欢", "不太喜欢", "最近在学", "计划明年去", "已经买好了", "每周都会去", "正在研究"]
OBJECTS = [
    "手冲咖啡", "徒步旅行", "机械键盘", "钢琴", "成都的小吃", "日本料理", "晨跑",
    "摄影", "养多肉植物", "看纪录片", "老式胶片相机", "露营装备", "烘焙面包",
    "羽毛球", "古典音乐", "编程副业", "杭州西湖", "云南普洱", "猫粮品牌", "智能家居",
]
TAILS = [
    "周末经常聊起这件事", "为此专门做了笔记", "还推荐给了别的朋友",
    "下次见面可以细说", "已经坚持了半年", "打算长期保持", "上周刚讨论过",
]
POOL_SIZE = len(SUBJECTS) * len(VERBS) * len(OBJECTS) * len(TAILS)


def make_corpus(size: int) -> list[str]:
    rows: list[str] = []
    for index in range(size):
        subject = SUBJECTS[index % len(SUBJECTS)]
        verb = VERBS[(index // len(SUBJECTS)) % len(VERBS)]
        obj = OBJECTS[(index // (len(SUBJECTS) * len(VERBS))) % len(OBJECTS)]
        tail = TAILS[(index // (len(SUBJECTS) * len(VERBS) * len(OBJECTS))) % len(TAILS)]
        text = f"{subject}{verb}{obj}，{tail}。"
        # 组合池是 7×7×20×7=6860；超出后补序号，保证 corpus 条数等于真实向量条数。
        if index >= POOL_SIZE:
            text = f"{text[:-1]}（第{index}条）。"
        rows.append(text)
    if size:
        rows[0] = "我住在杭州，喜欢手冲咖啡，每天早上七点起床。"
        rows[1 % size] = "我朋友小王最近在准备一场半程马拉松，训练得很认真。"
    return rows


def point_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def load_secrets() -> None:
    """复用项目的外置密钥文件；没有就放占位符（仅用于不调用回答模型的路径）。"""
    secrets_root = Path(os.environ.get("MEM0_SECRETS_ROOT")
                        or Path(os.environ["MEM0_RUNTIME_ROOT"]) / "secrets")
    path = secrets_root / "deepseek.env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    # CLI 的 provider 优先读 OPENAI_API_KEY，用真实 DeepSeek key 顶上，避免被占位符抢占。
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = deepseek_key
    os.environ.setdefault("OPENAI_API_KEY", "benchmark-unused")


# --------------------------------------------------------------------------- #
# 各阶段测量
# --------------------------------------------------------------------------- #
def collect_environment(requested_device: str, resolved_device: str, data_dir: Path,
                        *, vector_backend: str | None = None, collection_name: str | None = None,
                        embed_model: str | None = None) -> dict[str, object]:
    """把验收需要的硬件/软件/后端证据固化成结构化字段，避免只留在 stdout。"""
    import platform

    if vector_backend is None:
        url = os.environ.get("BUMI_QDRANT_URL", "").strip()
        if url:
            vector_backend = f"qdrant-server({url})"
        elif (data_dir / "qdrant").exists():
            vector_backend = "qdrant-local-path"
        else:
            vector_backend = "unknown"
    env: dict[str, object] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "logical_cpus": os.cpu_count(),
        "requested_embed_device": requested_device,
        "resolved_embed_device": resolved_device,
        "MEM0_EMBED_DEVICE_env": os.environ.get("MEM0_EMBED_DEVICE"),
        "entity_boost": os.environ.get("BUMI_ENTITY_BOOST", "on"),
        "vector_backend": vector_backend,
        "data_dir": str(data_dir),
    }
    if collection_name:
        env["vector_store_collection"] = collection_name
    if embed_model:
        env["embed_model"] = embed_model
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["torch_cuda_build"] = torch.version.cuda
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            env["torch_cuda_devices"] = [torch.cuda.get_device_name(index)
                                         for index in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001 - 证据缺失也要留痕，不能静默
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import mem0

        env["mem0_version"] = getattr(mem0, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    return env


def print_environment(env: dict[str, object]) -> None:
    print("=" * 66)
    print("⓪ 环境证据（验收口径，同时写入 JSON 的 environment 字段）")
    for key in ("hostname", "platform", "python_version", "python_executable", "logical_cpus",
                "requested_embed_device", "resolved_embed_device", "MEM0_EMBED_DEVICE_env",
                "entity_boost",
                "torch_version", "torch_cuda_build", "torch_cuda_available",
                "torch_cuda_device_count", "torch_cuda_devices", "torch_error",
                "mem0_version", "embed_model", "vector_backend", "vector_store_collection",
                "data_dir"):
        if key in env:
            print(f"{key:<28}: {env[key]}")
    print("=" * 66)


def measure_cold_start(ltm, data_dir: Path, recorder: Recorder, query: str) -> dict[str, object]:
    memory = ltm.memory
    table_header("① 冷启动（一次性成本）")
    print(f"{'数据目录':<26} {str(data_dir)}")

    started = time.perf_counter()
    encoder = memory.vector_store._get_bm25_encoder()
    bm25_ms = (time.perf_counter() - started) * 1000
    print(fmt_row("BM25 编码器首次加载", [bm25_ms]))
    print(f"{'BM25 是否可用':<26} {'是' if encoder is not None else '否'}")

    started = time.perf_counter()
    warm_rows = memory.search(query=query, top_k=1, filters={"run_id": f"warmup_{uuid.uuid4().hex}"},
                              threshold=0.0, rerank=False)
    first_search_ms = (time.perf_counter() - started) * 1000
    print(fmt_row("首次 search（含模型预热）", [first_search_ms]))
    warm_rows = warm_rows.get("results", warm_rows) if isinstance(warm_rows, dict) else warm_rows
    return {
        "bm25_encoder_load_ms": bm25_ms,
        "bm25_available": encoder is not None,
        "first_search_ms": first_search_ms,
        "first_search_hits": len(warm_rows or []),
    }


def measure_write_path(ltm, corpus: list[str]) -> int:
    memory = ltm.memory
    user_id, agent_id = "bench_user", "bumi_companion"
    now = datetime.now(timezone.utc).isoformat()
    payloads = [
        {
            "data": text,
            "text_lemmatized": text,
            "user_id": user_id,
            "agent_id": agent_id,
            "created_at": now,
            "updated_at": now,
            "hash": hashlib.sha1(text.encode()).hexdigest(),
        }
        for text in corpus
    ]

    started = time.perf_counter()
    vectors = memory.embedding_model.embed_batch(corpus, "add")
    embed_ms = (time.perf_counter() - started) * 1000

    encoder = memory.vector_store._get_bm25_encoder()
    started = time.perf_counter()
    if encoder is not None:
        list(encoder.embed(corpus))
    bm25_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    # 分批 upsert：Qdrant Server 的 HTTP body 上限 32MB，整批提交会 400。
    ids = [point_id(text) for text in corpus]
    insert_batch = 500
    for start in range(0, len(corpus), insert_batch):
        memory.vector_store.insert(
            vectors=vectors[start:start + insert_batch],
            payloads=payloads[start:start + insert_batch],
            ids=ids[start:start + insert_batch],
        )
    insert_ms = (time.perf_counter() - started) * 1000

    table_header(f"② 写入路径（{len(corpus)} 条，一次性）")
    print(f"{'唯一文本/向量数':<26} {len(set(corpus))}（point_id 按文本去重）")
    print(fmt_row(f"稠密转向量（共 {len(corpus)} 条）", [embed_ms]))
    print(fmt_row("稠密转向量（每条）", [embed_ms / len(corpus)]))
    print(fmt_row(f"BM25 转向量（共 {len(corpus)} 条）", [bm25_ms]))
    print(fmt_row("BM25 转向量（每条）", [bm25_ms / len(corpus)]))
    print(fmt_row("Qdrant 写入（含 BM25 再编码）", [insert_ms]))
    print(f"{'写入吞吐':<26} {len(corpus) / (insert_ms / 1000):.1f} 条/秒")
    return len(vectors[0]) if vectors else 0


def measure_embedding_micro(memory, repeats: int) -> None:
    lengths = [
        ("短（8 字）", "杭州咖啡"),
        ("中（60 字）", "我住在杭州，喜欢手冲咖啡，" * 3),
        ("长（300 字）", "我最近在研究手冲咖啡的萃取参数和不同产地的风味差异，" * 8),
    ]
    table_header("③ 转向量微基准（单条）")
    for label, text in lengths:
        values = []
        for _ in range(max(3, repeats // 4)):
            started = time.perf_counter()
            memory.embedding_model.embed(text, "search")
            values.append((time.perf_counter() - started) * 1000)
        print(fmt_row(f"稠密 embed {label}", values))

    encoder = memory.vector_store._get_bm25_encoder()
    if encoder is not None:
        for label, text in lengths:
            values = []
            for _ in range(max(3, repeats // 4)):
                started = time.perf_counter()
                list(encoder.embed([text]))
                values.append((time.perf_counter() - started) * 1000)
            print(fmt_row(f"BM25 embed {label}", values))

    table_header("④ 转向量微基准（批量）")
    base = "我住在杭州，喜欢手冲咖啡，每天早上七点起床。"
    for batch in (1, 8, 32):
        texts = [base] * batch
        values = []
        for _ in range(max(2, repeats // 6)):
            started = time.perf_counter()
            memory.embedding_model.embed_batch(texts, "search")
            values.append((time.perf_counter() - started) * 1000)
        mean, median, p90, worst = summarize(values)
        print(f"{f'稠密 batch={batch}':<26} {mean:>9.1f} {median:>9.1f} {p90:>9.1f} {worst:>9.1f}"
              f"   （每条 {mean / batch:.1f}ms）")

    table_header("⑤ Qdrant 单路查询（不含编码）")
    query_vector = memory.embedding_model.embed(base, "search")
    filters = {"user_id": "bench_user", "agent_id": "bumi_companion"}
    for top_k in (5, 20, 60):
        values = []
        for _ in range(max(3, repeats // 2)):
            started = time.perf_counter()
            memory.vector_store.search(query=base, vectors=query_vector, top_k=top_k, filters=filters)
            values.append((time.perf_counter() - started) * 1000)
        print(fmt_row(f"语义路 Qdrant top_k={top_k}", values))

    if encoder is not None:
        for top_k in (5, 60):
            values = []
            for _ in range(max(3, repeats // 2)):
                started = time.perf_counter()
                memory.vector_store.keyword_search(query=base, top_k=top_k, filters=filters)
                values.append((time.perf_counter() - started) * 1000)
            print(fmt_row(f"BM25 路 Qdrant top_k={top_k}", values))


def measure_search_path(memory, recorder: Recorder, queries: list[str], repeats: int,
                        top_k: int, threshold: float) -> dict[str, dict]:
    import mem0.memory.main as mem0_main

    recorder.reset()
    wrap_function(mem0_main, "lemmatize_for_bm25", recorder, "lemmatize")
    wrap_function(mem0_main, "extract_entities", recorder, "entities")
    wrap_method(memory.embedding_model, "embed", recorder, "dense_embed")
    wrap_method(memory.embedding_model, "embed_batch", recorder, "entity_embed")
    wrap_method(memory.vector_store, "search", recorder, "dense_query")
    wrap_method(memory.vector_store, "_encode_bm25", recorder, "bm25_encode")
    wrap_method(memory.vector_store, "keyword_search", recorder, "bm25_query")
    wrap_method(memory, "_compute_entity_boosts", recorder, "entity_boost")

    filters = {"user_id": "bench_user", "agent_id": "bumi_companion"}
    per_query: dict[str, dict] = {}
    for query in queries:
        before = {name: len(values) for name, values in recorder.samples.items()}
        totals: list[float] = []
        hits: list[int] = []
        for _ in range(repeats):
            started = time.perf_counter()
            rows = memory.search(query=query, top_k=top_k, filters=filters,
                                 threshold=threshold, rerank=False)
            totals.append((time.perf_counter() - started) * 1000)
            rows = rows.get("results", rows) if isinstance(rows, dict) else rows
            hits.append(len(rows or []))
        per_query[query] = {
            "total": totals,
            "hits": hits,
            "stages": {
                name: values[before.get(name, 0):]
                for name, values in recorder.samples.items()
            },
        }

    total_values = [value for item in per_query.values() for value in item["total"]]
    table_header(f"⑥ 完整检索 memory.search（{len(queries)} 查询 × {repeats} 次，top_k={top_k}）")
    calls = len(queries) * repeats
    stage_order = [
        ("lemmatize", "spaCy 词形还原"),
        ("entities", "spaCy 实体抽取"),
        ("dense_embed", "转向量（稠密 Qwen3）"),
        ("dense_query", "语义路 Qdrant 检索"),
        ("bm25_query", "BM25 路检索（含稀疏编码）"),
        ("entity_boost", "实体加成检索"),
    ]
    accounted = 0.0
    for name, label in stage_order:
        values = [value for item in per_query.values() for value in item["stages"].get(name, [])]
        if values:
            accounted += sum(values) / calls
        print(fmt_row(label, values))
    nested = [value for item in per_query.values() for value in item["stages"].get("bm25_encode", [])]
    if nested:
        print(fmt_row("  └ 其中的稀疏编码", nested))
    mean_total, median_total, p90_total, worst_total = summarize(total_values)
    print("-" * 66)
    print(f"{'总计 memory.search':<26} {mean_total:>9.1f} {median_total:>9.1f} "
          f"{p90_total:>9.1f} {worst_total:>9.1f}")
    print(f"{'其中 融合/排序/其他':<26} {mean_total - accounted:>9.1f}")

    table_header("⑦ 分查询明细（每次调用，ms）")
    print(f"{'查询':<30} {'总中位':>8} {'语义路':>8} {'BM25路':>8} {'加成':>8} {'命中':>6}")
    print("-" * 76)
    for query, item in per_query.items():
        dense = item["stages"].get("dense_query", [])
        bm25 = item["stages"].get("bm25_query", [])
        boost = item["stages"].get("entity_boost", [])
        label = query if len(query) <= 28 else query[:27] + "…"
        print(f"{label:<30} {statistics.median(item['total']):>8.1f} "
              f"{(statistics.median(dense) if dense else 0):>8.1f} "
              f"{(statistics.median(bm25) if bm25 else 0):>8.1f} "
              f"{(statistics.median(boost) if boost else 0):>8.1f} "
              f"{statistics.fmean(item['hits']):>6.1f}")
    return per_query


def _build_engine(data_dir: Path, args: argparse.Namespace, read_timeout: float):
    """构造引擎；每轮用全新的基准 SQLite，上一轮残留的未结算轮次会以 agent_busy 拒绝新请求。"""
    from bumi_mem.cli import parse_args as cli_parse_args, build_engine

    load_secrets()
    db_path = data_dir.parent / "bench.sqlite3"
    for suffix in ("", "-wal", "-shm"):
        stale = Path(str(db_path) + suffix)
        if stale.exists():
            stale.unlink()
    cli_args = [
        "--user-id", "bench_user", "--agent-id", "bumi_companion",
        "--chat-provider", "deepseek", "--memory-provider", "deepseek",
        "--db-path", str(db_path),
        "--collection-name", args.collection,
        "--memory-read-timeout", str(read_timeout),
    ]
    if args.embed_device:
        cli_args += ["--embed-device", args.embed_device]

    original_argv = sys.argv
    try:
        sys.argv = ["bumi-latency-bench", *cli_args]
        cli_ns = cli_parse_args()
    finally:
        sys.argv = original_argv

    started = time.perf_counter()
    engine = build_engine(cli_ns)
    build_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    warmup = engine.start()
    return engine, {"build_ms": build_ms,
                    "warmup_ms": float(warmup) if warmup is not None else 0.0,
                    "db_path": str(db_path)}


def measure_engine(data_dir: Path, args: argparse.Namespace, queries: list[str]) -> dict:
    """前台召回：走 SerialMemoryAccess + 截止时间，正是 CLI 每轮走的路径。"""
    engine, boot = _build_engine(data_dir, args, args.read_timeout)
    table_header("⑧ 引擎前台路径")
    print(fmt_row("build_engine（构造对象）", [boot["build_ms"]]))
    print(fmt_row("引擎启动预热", [boot["warmup_ms"]]))

    statuses: dict[str, int] = defaultdict(int)
    foreground: list[float] = []
    chat_totals: list[float] = []
    for query in queries:
        started = time.perf_counter()
        prepared = engine.prepare_prompt(user_id="bench_user", agent_id="bumi_companion",
                                         query=query, memory_only=False)
        foreground.append((time.perf_counter() - started) * 1000)
        statuses[prepared.get("recall_status", "?")] += 1
        # 必须结算轮次，否则下一次 prepare 会以 agent_busy 拒绝（结算本身不计入上面的耗时）。
        engine.record_reply(turn_id=prepared["turn_id"], agent_id="bumi_companion",
                            reply="（时延基准占位回复）")
        if args.with_chat:
            started = time.perf_counter()
            engine.chat(user_id="bench_user", agent_id="bumi_companion", text=query)
            chat_totals.append((time.perf_counter() - started) * 1000)

    print(fmt_row(f"prepare_prompt（截止 {args.read_timeout}s）", foreground))
    over = [value for value in foreground if value > args.read_timeout * 1000]
    print(f"{'超过截止的轮次':<26} {len(over)}/{len(foreground)}")
    print(f"{'recall_status 分布':<26} {dict(statuses)}")
    if chat_totals:
        print(fmt_row("端到端 chat（含回答模型）", chat_totals))
    engine.close()
    return {**boot, "queries": queries, "foreground_ms": foreground,
            "chat_ms": chat_totals, "statuses": dict(statuses)}


def measure_engine_decomposition(data_dir: Path, args: argparse.Namespace, queries: list[str]) -> dict:
    """把前台一轮拆成“检索”与“检索之后的串行拼装”。

    检索可在流式识别前缀上提前发起，不占串行关键路径；真正占用的是总时长
    减去检索的那部分：排轮次、核心状态读取、引用落库、上下文拼装渲染与
    Prompt 持久化。
    """
    engine, boot = _build_engine(data_dir, args, args.assembly_read_timeout)
    recorder = Recorder()
    wrap_method(engine.long_term, "search", recorder, "retrieval")
    wrap_method(engine.long_term, "search_with_timeout", recorder, "retrieval")
    wrap_method(engine.store, "reserve_turn", recorder, "reserve_turn")
    wrap_method(engine.store, "semantic_context", recorder, "semantic_context")
    wrap_method(engine.store, "attach_memory_refs", recorder, "attach_refs")
    wrap_method(engine, "build_context", recorder, "build_context")
    wrap_method(engine.store, "save_turn_prompt", recorder, "save_turn_prompt")

    statuses: dict[str, int] = defaultdict(int)
    totals: list[float] = []
    serial: list[float] = []
    for query in queries:
        before = {name: len(values) for name, values in recorder.samples.items()}
        started = time.perf_counter()
        prepared = engine.prepare_prompt(user_id="bench_user", agent_id="bumi_companion",
                                         query=query, memory_only=False)
        totals.append((time.perf_counter() - started) * 1000)
        statuses[prepared.get("recall_status", "?")] += 1
        engine.record_reply(turn_id=prepared["turn_id"], agent_id="bumi_companion",
                            reply="（时延基准占位回复）")
        retrieval = recorder.samples["retrieval"][before.get("retrieval", 0):]
        if retrieval:
            serial.append(totals[-1] - sum(retrieval))

    table_header(f"⑨ 前台一轮分解（检索等待上限 {args.assembly_read_timeout}s，串行=总时长-检索）")
    print(fmt_row("完整 prepare_prompt", totals))
    for name, label in (
        ("retrieval", "L3 检索（可提前发起）"),
        ("reserve_turn", "排轮次/写当前问题"),
        ("semantic_context", "核心状态读取"),
        ("attach_refs", "召回引用落库"),
        ("build_context", "上下文拼装与渲染"),
        ("save_turn_prompt", "Prompt 持久化"),
    ):
        values = recorder.samples.get(name, [])
        if values:
            print(fmt_row(label, values))
    if serial:
        print("-" * 66)
        print(fmt_row("串行关键路径（总-检索）", serial))
    print(f"{'recall_status 分布':<26} {dict(statuses)}")
    engine.close()
    return {
        "read_timeout": args.assembly_read_timeout,
        "queries": queries,
        "total_ms": totals,
        "serial_ms": serial,
        "statuses": dict(statuses),
        "stages_ms": {name: list(values) for name, values in recorder.samples.items()},
        "boot": boot,
    }


def run_scale_probe(data_root: Path, sizes: list[int], json_out: str | None) -> int:
    """本地 Qdrant 稠密检索随向量条数的伸缩。随机归一化向量，只测存储层，不是语义质量证据。"""
    import numpy as np
    from mem0.vector_stores.qdrant import Qdrant

    probe_root = data_root / "qdrant-scale"
    probe_root.mkdir(parents=True, exist_ok=True)
    qdrant_url = os.environ.get("BUMI_QDRANT_URL", "").strip()
    environment = collect_environment("not-used", "not-used", probe_root)
    print_environment(environment)
    rng = np.random.default_rng(0)
    filters = {"user_id": "bench_user", "agent_id": "bumi_companion"}
    table_header("⑩ 本地 Qdrant 稠密检索规模伸缩（随机向量，非语义质量证据）")
    results: dict[str, dict] = {}
    for n in sizes:
        path = probe_root / f"n{n}"
        path.mkdir(parents=True, exist_ok=True)
        if qdrant_url:
            store = Qdrant(collection_name=f"scale_{n}", embedding_model_dims=1024, url=qdrant_url)
            store.client.delete_collection(f"scale_{n}")
            store.create_col(1024, on_disk=True)
        else:
            store = Qdrant(collection_name=f"scale_{n}", embedding_model_dims=1024, path=str(path))
        vectors = rng.normal(size=(n, 1024)).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        payloads = [{"user_id": "bench_user", "agent_id": "bumi_companion"} for _ in range(n)]
        ids = [str(uuid.UUID(int=index)) for index in range(n)]
        started = time.perf_counter()
        for start in range(0, n, 2000):
            store.insert(vectors=vectors[start:start + 2000].tolist(),
                         payloads=payloads[start:start + 2000], ids=ids[start:start + 2000])
        insert_ms = (time.perf_counter() - started) * 1000
        query_vector = vectors[0].tolist()
        values: list[float] = []
        for _ in range(30):
            started = time.perf_counter()
            store.search(query="scale-probe", vectors=query_vector, top_k=5, filters=filters)
            values.append((time.perf_counter() - started) * 1000)
        mean, median, p90, worst = summarize(values)
        print(f"{f'n={n} 稠密查询':<26} {mean:>9.2f} {median:>9.2f} {p90:>9.2f} {worst:>9.2f}")
        print(f"{f'n={n} 写入（本地client）':<26} {insert_ms:>9.1f} ms   "
              f"{n / (insert_ms / 1000):.0f} 条/秒")
        results[str(n)] = {"insert_ms": insert_ms, "query_ms": values,
                           "throughput_per_s": n / (insert_ms / 1000)}
        store.client.close()
    if json_out:
        payload = {"config": {"mode": "qdrant-scale-probe", "sizes": sizes,
                              "collection": "scale_<n>", "dim": 1024, "top_k": 5,
                              "data_dir": str(probe_root)},
                   "environment": environment,
                   "scale": results}
        Path(json_out).expanduser().write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        print(f"\n原始数据：{json_out}")
    return 0


def main() -> int:
    logging.getLogger().setLevel(logging.ERROR)
    for name in ("mem0", "transformers", "sentence_transformers", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)

    paths = bootstrap()
    args = parse_args()

    runtime_root = Path(os.environ["MEM0_RUNTIME_ROOT"])
    data_root = Path(args.data_root).expanduser() if args.data_root else runtime_root / "latency-bench"
    # 与 bumi_mem.paths.DATA_ROOT 对齐：引擎和基准共用同一份语料，且完全不碰应用真实数据目录。
    os.environ["MEM0_DATA_ROOT"] = str(data_root / "data")
    data_dir = data_root / "data" / "bumi_mem_l3"
    data_dir.mkdir(parents=True, exist_ok=True)
    # 先加载外置 secrets，让环境证据记录的是实际生效的 MEM0_EMBED_DEVICE，而不是默认值。
    load_secrets()

    if args.scale_probe:
        sizes = [int(part) for part in args.scale_probe.split(",") if part.strip()]
        return run_scale_probe(data_root, sizes, args.json_out)

    from bumi_mem.long_term import Mem0AdapterConfig, Mem0LongTermMemory, resolve_embed_device
    from bumi_mem.paths import MODEL_ROOT
    from bumi_mem.providers import ProviderConfig

    device = args.embed_device or os.environ.get("MEM0_EMBED_DEVICE", "auto")
    resolved_device = resolve_embed_device(device)
    embed_model = MODEL_ROOT / "Qwen3-Embedding-0.6B"
    # 后端由 BUMI_QDRANT_URL 决定：未设置=本地 path client，设置=Qdrant Server。
    environment = collect_environment(device, resolved_device, data_dir,
                                      collection_name=args.collection,
                                      embed_model=str(embed_model))
    print("=" * 66)
    print("Bumi_Mem 时延基准")
    print(f"仓库      : {paths['repo']}")
    print(f"Python    : {paths['python']}")
    print(f"数据目录  : {data_dir}")
    print(f"embed 设备: {resolved_device}（请求 {device}）")
    print(f"语料条数  : {args.corpus}    每查询重复: {args.repeats}    top_k: {args.top_k}")
    print("=" * 66)
    print_environment(environment)

    queries = DEFAULT_QUERIES
    if args.queries:
        queries = [line.strip() for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
                   if line.strip()]

    if args.assembly:
        # 本地 Qdrant 是单实例锁，分解必须独立进程跑，且数据目录里要已有语料；
        # 用 Qdrant Server 时语料在服务端，不检查本地 qdrant 目录。
        if not os.environ.get("BUMI_QDRANT_URL", "").strip() and not (data_dir / "qdrant").exists():
            print(f"\n--assembly 需要已有语料：{data_dir} 下没有 qdrant 目录，请先跑一次完整基准")
            return 2
        decomposition = measure_engine_decomposition(data_dir, args, queries)
        if args.json_out:
            payload = {
                "config": {"corpus": args.corpus, "repeats": args.repeats, "top_k": args.top_k,
                           "device": resolved_device, "data_dir": str(data_dir),
                           "collection": args.collection, "mode": "assembly-only"},
                "environment": {**environment, "warmup": decomposition.get("boot", {})},
                "engine": {"decomposition": decomposition},
            }
            Path(args.json_out).expanduser().write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
            print(f"\n原始数据：{args.json_out}")
        return 0

    started = time.perf_counter()
    ltm = Mem0LongTermMemory(
        ProviderConfig(provider="deepseek", model="deepseek-chat",
                       base_url="https://api.deepseek.com/v1", api_key="benchmark-unused"),
        Mem0AdapterConfig(data_dir=data_dir, embed_model=embed_model,
                          embed_device=device, collection_name=args.collection),
    )
    construct_ms = (time.perf_counter() - started) * 1000

    cold_start = measure_cold_start(ltm, data_dir, Recorder(), queries[0])
    print(f"{'Memory.from_config 构造':<26} {construct_ms:>9.1f}   （含 embedding 模型加载）")
    environment["warmup"] = {"memory_construct_ms": construct_ms, **cold_start}

    corpus = make_corpus(args.corpus)
    unique_corpus = len(set(corpus))
    dimension = measure_write_path(ltm, corpus)
    print(f"{'向量维度':<26} {dimension}")
    measure_embedding_micro(ltm.memory, args.repeats)
    recorder = Recorder()
    per_query = measure_search_path(ltm.memory, recorder, queries, args.repeats,
                                    args.top_k, args.threshold)

    engine_payload: dict[str, dict] = {}
    if args.with_engine or args.with_chat:
        # 本地 Qdrant 是单实例锁：先释放基准用的句柄，引擎才能打开同一个数据目录。
        try:
            ltm.close()
        except Exception as exc:  # noqa: BLE001 - 释放失败时下面会给出可读提示
            print(f"\n释放基准数据目录失败：{exc}")
        try:
            engine_payload["foreground"] = measure_engine(data_dir, args, queries[:4])
        except Exception as exc:  # noqa: BLE001
            print(f"\n引擎前台路径未能测量：{type(exc).__name__}: {exc}")
        if "foreground" in engine_payload:
            environment.setdefault("warmup", {})
            environment["warmup"]["engine_warmup_ms"] = engine_payload["foreground"]["warmup_ms"]

    if args.json_out:
        payload = {
            "config": {
                "corpus": args.corpus, "unique_corpus": unique_corpus,
                "repeats": args.repeats, "top_k": args.top_k,
                "threshold": args.threshold, "device": resolved_device,
                "collection": args.collection,
                "data_dir": str(data_dir),
            },
            "environment": environment,
            "search": {query: {"total_ms": item["total"], "hits": item["hits"],
                               "stages_ms": item["stages"]}
                       for query, item in per_query.items()},
            "engine": engine_payload,
        }
        Path(args.json_out).expanduser().write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
        print(f"\n原始数据：{args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
