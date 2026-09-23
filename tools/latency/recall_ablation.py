#!/usr/bin/env python3
"""Bumi_Mem 检索对照实验：语义路 / 语义+BM25 / 完整（含实体加成）。

回答的问题是：BM25 路和实体加成这两条路，在中文语料上到底有没有带来召回收益？

做法：同一批语料、同一批带标准答案的查询，只切换 mem0 的两处行为——
    * 关掉 keyword_search  -> 语义路 only
    * 再关掉 _compute_entity_boosts -> 语义 + BM25
    * 都不关 -> 完整（mem0 默认混合检索）
比较 Recall@k、MRR 和时延。计时与召回都走 mem0 真实的 memory.search。

用法：
    bumi-retrieval-ablation
    bumi-retrieval-ablation --top-k 5 --repeats 5 --json /path/out.json
"""

from __future__ import annotations

import argparse
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

_HERE = Path(__file__).resolve()
DEFAULT_REPO = (str(_HERE.parents[2]) if (_HERE.parents[2] / "pyproject.toml").is_file()
                else "/data1/zhangran/git/Bumi_Mem")

# 标准答案：查询 → 期望命中的记忆原文
PROBES: list[tuple[str, str, str]] = [
    # (查询, 期望命中的记忆, 分组)
    ("我早上几点起床？", "我住在杭州，喜欢手冲咖啡，每天早上七点起床。", "中文·问句"),
    ("我住在哪个城市？", "我住在杭州，喜欢手冲咖啡，每天早上七点起床。", "中文·问句"),
    ("我妹妹在哪里读大学？", "我妹妹在成都读大学，主修钢琴。", "中文·问句"),
    ("我家猫叫什么名字？", "我家猫叫团子，是一只英国短毛猫。", "中文·问句"),
    ("小王最近在忙什么？", "我朋友小王最近在准备半程马拉松，训练得很认真。", "中文·问句"),
    ("咖啡", "我住在杭州，喜欢手冲咖啡，每天早上七点起床。", "中文·关键词"),
    ("钢琴", "我妹妹在成都读大学，主修钢琴。", "中文·关键词"),
    ("马拉松", "我朋友小王最近在准备半程马拉松，训练得很认真。", "中文·关键词"),
    ("AIC2049", "我的工牌号是 AIC2049。", "中文·编号"),
    ("我的工牌号是多少？", "我的工牌号是 AIC2049。", "中文·编号"),
    ("能帮我找一家安静、适合工作的咖啡馆吗？我下午三点到五点有空",
     "我最近在找适合办公的咖啡馆，常去的那家下午人很少。", "中文·长句"),
    ("Who plays the cello?", "The user's colleague Alice plays the cello.", "英文·对照"),
    ("cello", "The user's colleague Alice plays the cello.", "英文·对照"),
]

# 干扰记忆：故意和标准答案共享关键词，用来检验排序而不只是"能不能捞到"
DISTRACTORS = [
    "同事小李喜欢喝美式咖啡，不加糖。",
    "我妈妈每天早上六点起床做早饭。",
    "我妹妹喜欢听钢琴协奏曲，但不会弹。",
    "邻居家的狗叫豆豆，是一只柯基。",
    "同事小张上个月刚跑完十公里。",
    "我的门禁密码是 8842，别记错了。",
    "我上个月换了新工牌，颜色是蓝色的。",
    "楼下的咖啡店早上七点半开门。",
    "The user's colleague Bob plays the guitar.",
    "The user's manager lives in Shanghai.",
]

SUBJECTS = ["我", "我朋友小王", "我妈妈", "同事小李", "我妹妹", "楼下的张阿姨", "我大学同学"]
VERBS = ["喜欢", "不太喜欢", "最近在学", "计划明年去", "已经买好了", "每周都会去", "正在研究"]
OBJECTS = ["徒步旅行", "机械键盘", "成都的小吃", "日本料理", "晨跑", "摄影", "养多肉植物",
           "看纪录片", "老式胶片相机", "露营装备", "烘焙面包", "羽毛球", "古典音乐", "编程副业",
           "云南普洱", "猫粮品牌", "智能家居", "水彩画", "木工", "天文观测"]


def bootstrap() -> dict[str, str]:
    repo = Path(os.environ.get("BUMI_MEM_REPO", DEFAULT_REPO)).expanduser()
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
    marker = env.get("BUMI_RETRIEVAL_ABLATION_PYTHON")
    if python and marker != python:
        env["MEM0_PYTHON"] = python
        env["BUMI_RETRIEVAL_ABLATION_PYTHON"] = python
        os.execve(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    for key, value in env.items():
        os.environ[key] = value
    return {"repo": str(repo), "workspace": str(workspace), "python": python or sys.executable}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bumi_Mem 检索对照实验")
    parser.add_argument("--fillers", type=int, default=None,
                        help="干扰性填充记忆条数（默认：内置语料 200，--dataset 时 0）")
    parser.add_argument("--dataset", default=None,
                        help='真实/脱敏语料 JSON：{"memories": [...], '
                             '"probes": [{"query","expected","group"}, ...]}')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--collection", default="bumi_ablation",
                        help="Qdrant 集合名；跑真实语料时用独立集合，避免与内置语料混在一起")
    parser.add_argument("--json", dest="json_out", default=None)
    return parser.parse_args()


def load_secrets() -> None:
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
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = deepseek_key
    os.environ.setdefault("OPENAI_API_KEY", "benchmark-unused")


def point_id(text: str) -> str:
    return str(uuid.UUID(hashlib.sha1(text.encode("utf-8")).hexdigest()[:32]))


def load_dataset(path: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """读取外部召回回归语料：memories 是记忆原文，probes 是 (query, expected, group)。"""
    import json as _json

    payload = _json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    memories = [str(item) for item in payload.get("memories", [])]
    probes: list[tuple[str, str, str]] = []
    for item in payload.get("probes", []):
        probes.append((str(item["query"]), str(item["expected"]),
                       str(item.get("group", "未分组"))))
    if not memories or not probes:
        raise ValueError("dataset 必须同时包含非空的 memories 与 probes")
    missing = sorted({expected for _, expected, _ in probes if expected not in set(memories)})
    if missing:
        raise ValueError(
            f"{len(missing)} 条标准答案不在 memories 里，无法计算 Recall；例如：{missing[0][:40]}")
    return probes, memories


def build_corpus(fillers: int, probes: list[tuple[str, str, str]],
                 base_memories: list[str]) -> list[str]:
    rows = [text for _, text, _ in probes] + DISTRACTORS + base_memories
    for index in range(fillers):
        subject = SUBJECTS[index % len(SUBJECTS)]
        verb = VERBS[(index // len(SUBJECTS)) % len(VERBS)]
        obj = OBJECTS[(index // (len(SUBJECTS) * len(VERBS))) % len(OBJECTS)]
        rows.append(f"{subject}{verb}{obj}，最近经常聊起这件事。")
    # 去重但保持顺序
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            unique.append(row)
    return unique


def main() -> int:
    logging.getLogger().setLevel(logging.ERROR)
    for name in ("mem0", "transformers", "sentence_transformers", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)

    paths = bootstrap()
    args = parse_args()
    load_secrets()

    runtime_root = Path(os.environ["MEM0_RUNTIME_ROOT"])
    data_root = Path(args.data_root).expanduser() if args.data_root else runtime_root / "retrieval-ablation"
    os.environ["MEM0_DATA_ROOT"] = str(data_root / "data")
    data_dir = data_root / "data" / "bumi_mem_l3"
    data_dir.mkdir(parents=True, exist_ok=True)

    from bumi_mem.long_term import Mem0AdapterConfig, Mem0LongTermMemory, resolve_embed_device
    from bumi_mem.paths import MODEL_ROOT
    from bumi_mem.providers import ProviderConfig

    device = os.environ.get("MEM0_EMBED_DEVICE", "auto")
    print("=" * 74)
    print("Bumi_Mem 检索对照实验：语义路 / +BM25 / +实体加成")
    print(f"数据目录: {data_dir}")
    print(f"embed 设备: {resolve_embed_device(device)}    top_k={args.top_k}    repeats={args.repeats}")
    print("=" * 74)

    ltm = Mem0LongTermMemory(
        ProviderConfig(provider="deepseek", model="deepseek-chat",
                       base_url="https://api.deepseek.com/v1", api_key="benchmark-unused"),
        Mem0AdapterConfig(data_dir=data_dir, embed_model=MODEL_ROOT / "Qwen3-Embedding-0.6B",
                          embed_device=device, collection_name=args.collection),
    )
    memory = ltm.memory
    user_id, agent_id = "ablation_user", "bumi_companion"
    filters = {"user_id": user_id, "agent_id": agent_id}

    probes = PROBES
    base_memories: list[str] = []
    if args.dataset:
        probes, base_memories = load_dataset(args.dataset)
    fillers = args.fillers if args.fillers is not None else (0 if args.dataset else 200)
    print(f"语料来源: {'内置合成语料' if not args.dataset else args.dataset}"
          f"（记忆 {len(base_memories)} 条 + 干扰 {fillers} 条，查询 {len(probes)} 条）")

    corpus = build_corpus(fillers, probes, base_memories)
    now = datetime.now(timezone.utc).isoformat()
    payloads = [{"data": text, "text_lemmatized": text, "user_id": user_id, "agent_id": agent_id,
                 "created_at": now, "updated_at": now, "hash": hashlib.sha1(text.encode()).hexdigest()}
                for text in corpus]
    started = time.perf_counter()
    vectors = memory.embedding_model.embed_batch(corpus, "add")
    memory.vector_store.insert(vectors=vectors, payloads=payloads,
                               ids=[point_id(text) for text in corpus])
    print(f"\n写入 {len(corpus)} 条记忆，用时 {(time.perf_counter() - started):.1f}s")

    # 按 mem0 写入记忆时的真实逻辑建实体库：从每条记忆抽实体，编码后写入 <collection>_entities
    from mem0.utils.entity_extraction import extract_entities

    entity_texts: list[str] = []
    entity_types: list[str] = []
    entity_links: dict[str, set[str]] = defaultdict(set)
    chinese_memories_with_entities = 0
    english_memories_with_entities = 0
    for text in corpus:
        found = extract_entities(text)
        if found:
            if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                chinese_memories_with_entities += 1
            else:
                english_memories_with_entities += 1
        for entity_type, entity_text in found:
            key = entity_text.strip().lower()
            entity_links[key] |= {point_id(text)}
            if key not in [t.lower() for t in entity_texts]:
                entity_texts.append(entity_text)
                entity_types.append(entity_type)

    started = time.perf_counter()
    if entity_texts:
        entity_vectors = memory.embedding_model.embed_batch(entity_texts, "add")
        memory.entity_store.insert(
            vectors=entity_vectors,
            ids=[str(uuid.uuid4()) for _ in entity_texts],
            payloads=[{
                "data": text, "entity_type": etype,
                "linked_memory_ids": sorted(entity_links[text.lower()]),
                "user_id": user_id, "agent_id": agent_id,
            } for text, etype in zip(entity_texts, entity_types)],
        )
    build_entity_ms = (time.perf_counter() - started) * 1000
    print(f"实体库：从 {len(corpus)} 条记忆抽出 {len(entity_texts)} 条实体记录"
          f"（中文记忆贡献 {chinese_memories_with_entities} 条，英文 {english_memories_with_entities} 条），"
          f"用时 {build_entity_ms:.1f}ms")

    # 两条旁路的活性：BM25 能否捞出东西、实体抽取出什么、实体加成能否命中
    print("\n① 两条旁路的活性（直接调用内部函数，不经过融合）")
    print(f"{'查询':<30}{'BM25命中':>9}{'抽取实体':>9}{'加成命中':>9}")
    print("-" * 58)
    bm25_hits: dict[str, int] = {}
    entity_counts: dict[str, int] = {}
    boost_counts: dict[str, int] = {}
    for query, _, _ in probes:
        rows = memory.vector_store.keyword_search(query=query, top_k=args.top_k, filters=filters)
        bm25_hits[query] = len(rows or [])
        found = extract_entities(query)
        boosts = memory._compute_entity_boosts(found, filters)
        entity_counts[query] = len(found)
        boost_counts[query] = len(boosts)
        label = query if len(query) <= 28 else query[:27] + "…"
        print(f"{label:<30}{len(rows or []):>9}{len(found):>9}{len(boosts):>9}")

    original_search = memory._compute_entity_boosts
    original_keyword = memory.vector_store.keyword_search
    configs = [
        ("语义路 only", lambda: (setattr(memory, "_compute_entity_boosts", lambda *a, **k: {}),
                                setattr(memory.vector_store, "keyword_search", lambda *a, **k: None))),
        ("语义 + BM25", lambda: (setattr(memory, "_compute_entity_boosts", lambda *a, **k: {}),
                                setattr(memory.vector_store, "keyword_search", original_keyword))),
        ("完整（含实体加成）", lambda: (setattr(memory, "_compute_entity_boosts", original_search),
                                setattr(memory.vector_store, "keyword_search", original_keyword))),
    ]

    results: dict[str, dict] = {}
    for label, apply in configs:
        apply()
        per_group: dict[str, list[int]] = defaultdict(list)
        reciprocal: list[float] = []
        latencies: list[float] = []
        detail = []
        for query, target, group in probes:
            target_id = point_id(target)
            hit_rank = 0
            for _ in range(args.repeats):
                started = time.perf_counter()
                rows = memory.search(query=query, top_k=args.top_k, filters=filters,
                                     threshold=args.threshold, rerank=False)
                latencies.append((time.perf_counter() - started) * 1000)
                rows = rows.get("results", rows) if isinstance(rows, dict) else rows
                ids = [str(row.get("id")) for row in (rows or [])]
                if target_id in ids:
                    hit_rank = ids.index(target_id) + 1
            per_group[group].append(1 if hit_rank else 0)
            reciprocal.append(1.0 / hit_rank if hit_rank else 0.0)
            detail.append((query, group, hit_rank))
        results[label] = {
            "groups": {group: sum(v) / len(v) for group, v in per_group.items()},
            "recall": sum(sum(v) for v in per_group.values()) / sum(len(v) for v in per_group.values()),
            "mrr": statistics.fmean(reciprocal),
            "latency": latencies,
            "detail": detail,
        }

    print(f"\n② 三种配置的召回与耗时（top_k={args.top_k}，每条查询重复 {args.repeats} 次）")
    groups = [group for _, _, group in probes]
    groups = list(dict.fromkeys(groups))  # 按数据集出现顺序，保持稳定
    header = f"{'配置':<20}" + "".join(f"{g:>12}" for g in groups) + f"{'总Recall':>10}{'MRR':>8}{'中位ms':>9}"
    print(header)
    print("-" * len(header))
    for label in results:
        item = results[label]
        line = f"{label:<20}" + "".join(f"{item['groups'].get(g, 0):>12.2f}" for g in groups)
        line += f"{item['recall']:>10.2f}{item['mrr']:>8.2f}{statistics.median(item['latency']):>9.1f}"
        print(line)

    print("\n③ 逐查询命中排名（0 = 未命中）")
    print(f"{'查询':<28}{'分组':<12}" + "".join(f"{label[:12]:>14}" for label in results))
    print("-" * (28 + 12 + 14 * len(results)))
    for index, (query, group, _) in enumerate(results[list(results)[0]]["detail"]):
        label = query if len(query) <= 26 else query[:25] + "…"
        line = f"{label:<28}{group:<12}"
        for name in results:
            rank = results[name]["detail"][index][2]
            line += f"{(str(rank) if rank else '未命中'):>14}"
        print(line)

    if args.json_out:
        payload = {
            "config": vars(args),
            "corpus_size": len(corpus),
            "bm25_hits": bm25_hits,
            "results": {k: {"recall": v["recall"], "mrr": v["mrr"],
                            "groups": v["groups"], "latency_ms": v["latency"]}
                        for k, v in results.items()},
        }
        Path(args.json_out).expanduser().write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
        print(f"\n原始数据：{args.json_out}")
    ltm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
