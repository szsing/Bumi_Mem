#!/usr/bin/env python3
"""把公开数据集 LoCoMo（snap-research/locomo 的 locomo10.json）转成召回回归数据集。

输出格式与 tools/latency/datasets/recall_demo.json 一致：
    {"memories": [...], "probes": [{"query","expected","group"}, ...]}

- memories：每条对话话语，带会话时间与说话人前缀（`[<session_time>] <speaker>: <text>`），
  与记忆系统里带时间戳的 L1 消息形态接近；
- probes：每个带 evidence 的 QA 一条，query=问题，expected=evidence 指向的话语原文
  （多条 evidence 时是列表，检索命中任意一条即算召回），group=LoCoMo 的 category。

用法：
    tools/latency/locomo_to_recall_dataset.py \
      --input ~/mem0-runtime/datasets/locomo10.json \
      --out ~/mem0-runtime/datasets/recall_locomo.json --max-probes 300

    # 只取 1 段对话、80 条查询，并翻成中文（带翻译缓存，可复用/复现）
    MEM0_PYTHON=$HOME/.venvs/bumi-cuda/bin/python \
    tools/latency/locomo_to_recall_dataset.py --samples 1 --max-probes 80 \
      --translate --out ~/mem0-runtime/datasets/recall_locomo_zh.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reexec_with_mem0_python() -> None:
    """--translate 需要 openai 依赖；MEM0_PYTHON 指向项目 venv 时先切过去。"""
    python = os.environ.get("MEM0_PYTHON")
    marker = os.environ.get("BUMI_LOCOMO_PYTHON")
    if python and os.access(python, os.X_OK) and marker != python:
        env = dict(os.environ)
        env["BUMI_LOCOMO_PYTHON"] = python
        os.execve(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def load_api_key(runtime: Path) -> tuple[str, str]:
    values: dict[str, str] = {}
    secrets = runtime / "secrets" / "deepseek.env"
    if secrets.is_file():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")
    return (values.get("DEEPSEEK_API_KEY", ""),
            values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))


def extract_json_array(text: str) -> list:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        raise ValueError("输出里找不到 JSON 数组")
    return json.loads(text[start:end + 1])


def translate_batch(client, model: str, items: list[str], glossary: str) -> list[str]:
    prompt = (
        "把下面 JSON 数组里的英文逐条翻译成自然、口语化的中文，保持信息完整。\n"
        "要求：1) 只输出与输入等长的 JSON 字符串数组，不要解释、不要代码块；"
        "2) 人名/地名按术语表翻译；3) 日期、时间、数字保持原样；"
        "4) 不要增加原文没有的信息。\n"
        f"术语表：{glossary or '（无）'}\n"
        f"输入：{json.dumps(items, ensure_ascii=False)}"
    )
    for attempt in (1, 2):
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2)
        try:
            translated = extract_json_array(response.choices[0].message.content or "")
            if len(translated) != len(items) or not all(str(x).strip() for x in translated):
                raise ValueError(f"返回 {len(translated)} 条，期望 {len(items)} 条")
            return [str(x).strip() for x in translated]
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise
            prompt += f"\n上一次输出不合法（{exc}），请重试并只输出 JSON 数组。"
    raise RuntimeError("unreachable")


def translate_many(client, model: str, items: list[str], cache: dict[str, str],
                   glossary: str, batch_size: int, sleep_seconds: float) -> None:
    pending = [item for item in items if item not in cache]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        for src, dst in zip(batch, translate_batch(client, model, batch, glossary)):
            cache[src] = dst
        print(f"  已翻译 {min(start + batch_size, len(pending))}/{len(pending)} 条")
        if sleep_seconds:
            time.sleep(sleep_seconds)


def iter_turns(conversation: dict) -> list[tuple[str, str, str]]:
    """返回 (dia_id, memory_text, session_date) 列表，按会话顺序。"""
    turns: list[tuple[str, str, str]] = []
    session_index = 1
    while True:
        key = f"session_{session_index}"
        if key not in conversation:
            break
        date = str(conversation.get(f"{key}_date_time", "")).strip()
        for turn in conversation[key]:
            text = str(turn.get("text", "")).strip()
            dia_id = str(turn.get("dia_id", "")).strip()
            speaker = str(turn.get("speaker", "")).strip()
            if not text or not dia_id:
                continue
            prefix = f"[{date}] " if date else ""
            turns.append((dia_id, f"{prefix}{speaker}: {text}", date))
        session_index += 1
    return turns


def main() -> int:
    reexec_with_mem0_python()
    workspace = Path(os.environ.get("MEM0_WORKSPACE_ROOT", REPO.parent.parent)).expanduser()
    runtime = Path(os.environ.get("MEM0_RUNTIME_ROOT", workspace / "mem0-runtime")).expanduser()

    parser = argparse.ArgumentParser(description="LoCoMo → 召回回归数据集")
    parser.add_argument("--input", default=str(runtime / "datasets" / "locomo10.json"))
    parser.add_argument("--out", default=str(runtime / "datasets" / "recall_locomo.json"))
    parser.add_argument("--max-probes", type=int, default=300,
                        help="最多导出多少条 QA（按 category 轮询取，保证各类别都有）")
    parser.add_argument("--categories", default="1,2,3,4,5")
    parser.add_argument("--max-memories", type=int, default=0, help="0 = 全部话语")
    parser.add_argument("--samples", type=int, default=0,
                        help="只取前 N 段对话（0 = 全部 10 段）；先用它把规模压下来")
    parser.add_argument("--translate", action="store_true", help="把语料与查询翻成中文")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--batch-size", type=int, default=20, help="每次翻译请求的条数")
    parser.add_argument("--sleep", type=float, default=0.0, help="翻译批次之间的间隔秒数")
    parser.add_argument("--cache", default=None,
                        help="翻译缓存（默认 <runtime>/datasets/locomo_zh_cache.json，可复用/复现）")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).expanduser().read_text(encoding="utf-8"))
    if args.samples > 0:
        payload = payload[:args.samples]
    wanted = {int(item) for item in args.categories.split(",") if item.strip()}

    memories: list[str] = []
    memory_index: dict[tuple[str, str], str] = {}   # (sample_id, dia_id) -> memory text
    by_category: dict[str, list[dict]] = defaultdict(list)
    speakers: set[str] = set()
    for sample_position, sample in enumerate(payload):
        conversation = sample.get("conversation", {})
        sample_id = str(sample.get("sample_id") or sample_position)
        for name in (conversation.get("speaker_a"), conversation.get("speaker_b")):
            if name:
                speakers.add(str(name))
        for dia_id, text, _ in iter_turns(conversation):
            key = (sample_id, dia_id)
            if key in memory_index:
                continue
            # dia_id 只在单个会话内唯一，加 sample_id 前缀保证跨会话不冲突、可复现。
            memory_index[key] = f"[{sample_id}] {text}"
            memories.append(memory_index[key])
        for qa in sample.get("qa", []):
            category = int(qa.get("category", 0))
            if category not in wanted:
                continue
            targets = [memory_index.get((sample_id, str(dia))) for dia in qa.get("evidence", [])]
            targets = [item for item in targets if item]
            if not targets:
                continue
            by_category[f"cat{category}"].append({"query": str(qa["question"]), "expected": targets})

    if args.max_memories > 0:
        keep = set(memories[:args.max_memories])
        memories = memories[:args.max_memories]
        for group, probes in list(by_category.items()):
            by_category[group] = [p for p in probes if all(t in keep for t in p["expected"])]

    # 按 category 轮询取样，保证类别均衡且结果可复现
    probes: list[dict] = []
    groups = sorted(by_category)
    cursor = 0
    while len(probes) < args.max_probes and any(by_category[group] for group in groups):
        group = groups[cursor % len(groups)]
        cursor += 1
        if by_category[group]:
            item = by_category[group].pop(0)
            item["group"] = group
            probes.append(item)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.translate:
        cache_path = Path(args.cache).expanduser() if args.cache else runtime / "datasets" / "locomo_zh_cache.json"
        cache: dict[str, str] = (json.loads(cache_path.read_text(encoding="utf-8"))
                                 if cache_path.is_file() else {})
        api_key, base_url = load_api_key(runtime)
        if not api_key:
            print("找不到 DEEPSEEK_API_KEY，无法翻译", file=sys.stderr)
            return 2
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"翻译：{len(memories)} 条记忆 + {len(probes)} 条查询（缓存 {cache_path}）")
        # 先统一人名译法，再把它作为术语表塞进每个翻译请求，避免跨批次漂移
        translate_many(client, args.model, sorted(speakers), cache, "",
                       args.batch_size, args.sleep)
        glossary = "、".join(f"{name}→{cache.get(name, name)}" for name in sorted(speakers))
        translate_many(client, args.model, memories, cache, glossary,
                       args.batch_size, args.sleep)
        translate_many(client, args.model, [probe["query"] for probe in probes], cache,
                       glossary, args.batch_size, args.sleep)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        memories = [cache[item] for item in memories]
        for probe in probes:
            probe["query"] = cache[probe["query"]]
            probe["expected"] = [cache[item] for item in probe["expected"]]

    out.write_text(json.dumps({"memories": memories, "probes": probes},
                              ensure_ascii=False), encoding="utf-8")

    counts: dict[str, int] = defaultdict(int)
    multi = 0
    for probe in probes:
        counts[probe["group"]] += 1
        multi += 1 if len(probe["expected"]) > 1 else 0
    print(f"已生成：{out}")
    print(f"  memories: {len(memories)}（去重后全部话语）")
    print(f"  probes  : {len(probes)}  按类别 {dict(sorted(counts.items()))}")
    print(f"  其中多证据（任一命中即算召回）: {multi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
