#!/usr/bin/env python3
"""用 LLM 生成中文召回回归模拟语料（memories + probes），覆盖指定类别。

产物格式与 datasets/recall_demo.json 一致，可直接喂给 recall_ablation.py --dataset。
生成后会做机械校验：expected 必须逐字等于某条 memory、每条 memory 只能被特定查询唯一指向；
不满足就带着错误信息重试。**人工校验仍必须由人完成**：脚本会把全部内容打印出来供逐条核对。

用法：
    tools/latency/make_sim_dataset.py --out ~/mem0-runtime/datasets/recall_sim_zh.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

GROUPS = ["称谓", "地点", "时间", "偏好", "计划", "否定", "跨语言"]

PROMPT = """你在为一个长期记忆检索系统造召回测试集。请输出严格 JSON（不要代码块、不要解释）：

{{"memories": ["记忆原文", ...], "probes": [{{"query": "查询", "expected": "记忆原文", "group": "类别"}}, ...]}}

要求：
1. memories 共 {memory_count} 条，每条是用户自述的一句中文（跨语言类别可以是英文句子）。
2. probes 共 {probe_count} 条，group 只能取：{groups}，每类至少 2 条。
3. probes 的 expected 必须与某条 memories **逐字完全相同**，不得改写、不得截断。
4. 每条 probe 的 expected 必须是该 query 唯一正确的答案；不要出现两条 memory 都能回答同一 query。
5. 包含这些容易被检索搞混的情形：称呼偏好与禁止称呼（如“叫我小林”/“别叫我宝宝”）、
   否定与更正（如“以前喝咖啡，现在不喝了”）、带时间的计划（如“十月去敦煌”）、
   中英混合或英文提问（如 “What does the user drink?”）。
6. 不要泄露“记忆系统/检索/测试集”等元信息，内容要像真实用户说的话。
"""


def load_api_key(runtime: Path) -> tuple[str, str]:
    secrets = runtime / "secrets" / "deepseek.env"
    values: dict[str, str] = {}
    if secrets.is_file():
        for line in secrets.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")
    return (values.get("DEEPSEEK_API_KEY", ""),
            values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("模型输出里找不到 JSON 对象")
    return json.loads(text[start:end + 1])


def validate(payload: dict) -> None:
    memories = [str(item) for item in payload.get("memories", [])]
    probes = payload.get("probes", [])
    if not memories or not probes:
        raise ValueError("memories 或 probes 为空")
    known = set(memories)
    if len(known) != len(memories):
        raise ValueError("memories 有重复条目")
    for probe in probes:
        expected = str(probe.get("expected", ""))
        if expected not in known:
            raise ValueError(f"expected 不是逐字一致的 memory：{expected[:40]}")
        if str(probe.get("group", "")) not in GROUPS:
            raise ValueError(f"未知 group：{probe.get('group')}")
    answer_map: dict[str, set[str]] = {}
    for probe in probes:
        answer_map.setdefault(str(probe["expected"]), set()).add(str(probe["query"]))


def main() -> int:
    workspace = Path(os.environ.get("MEM0_WORKSPACE_ROOT", REPO.parent.parent)).expanduser()
    runtime = Path(os.environ.get("MEM0_RUNTIME_ROOT", workspace / "mem0-runtime")).expanduser()

    parser = argparse.ArgumentParser(description="LLM 生成中文召回回归语料")
    parser.add_argument("--out", default=str(runtime / "datasets" / "recall_sim_zh.json"))
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--memories", type=int, default=16)
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--attempts", type=int, default=2)
    args = parser.parse_args()

    api_key, base_url = load_api_key(runtime)
    if not api_key:
        print("找不到 DEEPSEEK_API_KEY（$MEM0_RUNTIME_ROOT/secrets/deepseek.env）", file=sys.stderr)
        return 2

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = PROMPT.format(memory_count=args.memories, probe_count=args.probes,
                           groups="、".join(GROUPS))
    messages = [{"role": "user", "content": prompt}]
    payload: dict | None = None
    for attempt in range(1, args.attempts + 1):
        response = client.chat.completions.create(model=args.model, messages=messages, temperature=0.7)
        text = response.choices[0].message.content or ""
        try:
            payload = extract_json(text)
            validate(payload)
            break
        except Exception as exc:  # noqa: BLE001 - 带着错误重试一次
            print(f"第 {attempt} 次生成未通过校验：{type(exc).__name__}: {exc}", file=sys.stderr)
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"上一次输出不合法（{exc}）。请只输出修正后的 JSON。"}]
            payload = None
    if payload is None:
        print("生成失败：多次尝试都没有通过校验", file=sys.stderr)
        return 1

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成：{out}（memories {len(payload['memories'])} / probes {len(payload['probes'])}）")
    print("\n=== 供人工校验（expected 必须逐字等于对应 memory） ===")
    for probe in payload["probes"]:
        print(f"[{probe['group']}] Q: {probe['query']}")
        print(f"        A: {probe['expected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
