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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


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
    workspace = Path(os.environ.get("MEM0_WORKSPACE_ROOT", REPO.parent.parent)).expanduser()
    runtime = Path(os.environ.get("MEM0_RUNTIME_ROOT", workspace / "mem0-runtime")).expanduser()

    parser = argparse.ArgumentParser(description="LoCoMo → 召回回归数据集")
    parser.add_argument("--input", default=str(runtime / "datasets" / "locomo10.json"))
    parser.add_argument("--out", default=str(runtime / "datasets" / "recall_locomo.json"))
    parser.add_argument("--max-probes", type=int, default=300,
                        help="最多导出多少条 QA（按 category 轮询取，保证各类别都有）")
    parser.add_argument("--categories", default="1,2,3,4,5")
    parser.add_argument("--max-memories", type=int, default=0, help="0 = 全部话语")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).expanduser().read_text(encoding="utf-8"))
    wanted = {int(item) for item in args.categories.split(",") if item.strip()}

    memories: list[str] = []
    memory_index: dict[tuple[str, str], str] = {}   # (sample_id, dia_id) -> memory text
    by_category: dict[str, list[dict]] = defaultdict(list)
    for sample_position, sample in enumerate(payload):
        conversation = sample.get("conversation", {})
        sample_id = str(sample.get("sample_id") or sample_position)
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
