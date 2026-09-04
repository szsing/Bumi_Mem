#!/usr/bin/env python3
"""Data-driven long-form evaluation for Mem0 v2.0.20."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from runtime_paths import DATA_ROOT, MODEL_ROOT, configure_environment


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = DEFAULT_ROOT / "tests" / "mem0_long_case.json"


def unwrap_results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        results = value.get("results", [])
        return results if isinstance(results, list) else []
    return value if isinstance(value, list) else []


def clean_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError("run-id 不能为空")
    return cleaned[:80]


def normalize_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.lower(), flags=re.UNICODE)


class Recorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8")

    def emit(self, text: str = "") -> None:
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class LongMem0Test:
    def __init__(
        self,
        root: Path,
        run_id: str,
        case: dict[str, Any],
        model: str,
        base_url: str,
        memory_model: str,
        memory_base_url: str,
        memory_api_key: str,
        memory_max_tokens: int,
        data_subdir: str,
        collection_name: str,
        search_threshold: float,
        search_explain: bool,
        top_k: int,
        device: str,
        max_answer_tokens: int,
    ) -> None:
        from mem0 import Memory
        from openai import OpenAI

        self.root = root
        self.run_id = run_id
        self.case = case
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.memory_model = memory_model
        self.memory_base_url = memory_base_url.rstrip("/")
        self.search_threshold = search_threshold
        self.search_explain = search_explain
        self.top_k = top_k
        self.device = device
        self.max_answer_tokens = max_answer_tokens
        self.user_id = f"mem0_long_{run_id}"
        self.run_dir = DATA_ROOT / data_subdir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.recorder = Recorder(self.run_dir / "transcript.txt")
        self.recent_messages: list[dict[str, str]] = []
        self.turn_records: list[dict[str, Any]] = []
        self.evaluation_records: list[dict[str, Any]] = []
        self.errors: list[dict[str, str]] = []
        self.event_counts: Counter[str] = Counter()

        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stderr),
                logging.FileHandler(self.run_dir / "runtime.log", encoding="utf-8"),
            ],
            force=True,
        )

        self.recorder.emit("=" * 76)
        self.recorder.emit(self.case.get("name", "Mem0 v2.0.20 长程测试"))
        self.recorder.emit(f"run-id: {run_id}")
        self.recorder.emit(f"user-id: {self.user_id}")
        self.recorder.emit(f"模型: {model} @ {self.base_url}")
        self.recorder.emit(f"记忆模型: {memory_model} @ {self.memory_base_url}")
        self.recorder.emit(f"检索: top_k={top_k}, threshold={search_threshold}")
        self.recorder.emit(f"嵌入设备: {device}")
        self.recorder.emit(f"独立数据目录: {self.run_dir}")
        self.recorder.emit("=" * 76)

        started = time.perf_counter()
        self.memory = Memory.from_config(
            {
                "version": "v1.1",
                "history_db_path": str(self.run_dir / "history.db"),
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": memory_model,
                        "api_key": memory_api_key,
                        "openai_base_url": self.memory_base_url,
                        "temperature": 0.0,
                        "max_tokens": memory_max_tokens,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": str(MODEL_ROOT / "Qwen3-Embedding-0.6B"),
                        "embedding_dims": 1024,
                        "model_kwargs": {
                            "device": device,
                            "local_files_only": True,
                        },
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": collection_name,
                        "embedding_model_dims": 1024,
                        "path": str(self.run_dir / "qdrant"),
                        "on_disk": True,
                    },
                },
            }
        )
        self.client = OpenAI(api_key="local", base_url=f"{self.base_url}/v1")
        self.init_seconds = time.perf_counter() - started
        self.recorder.emit(f"初始化完成: {self.init_seconds:.3f}s")

    def search(self, query: str) -> list[dict[str, Any]]:
        return unwrap_results(
            self.memory.search(
                query=query,
                top_k=self.top_k,
                filters={"user_id": self.user_id},
                threshold=self.search_threshold,
                explain=self.search_explain,
            )
        )

    def all_memories(self) -> list[dict[str, Any]]:
        return unwrap_results(
            self.memory.get_all(filters={"user_id": self.user_id}, top_k=500)
        )

    def print_memories(self, title: str, memories: list[dict[str, Any]]) -> None:
        self.recorder.emit(title)
        if not memories:
            self.recorder.emit("  （没有结果）")
            return
        for index, item in enumerate(memories, 1):
            text = item.get("memory", item.get("data", ""))
            score = item.get("score")
            score_text = f" score={score:.4f}" if isinstance(score, (int, float)) else ""
            self.recorder.emit(f"  {index}. [{item.get('id', '-')}] {text}{score_text}")

    def system_prompt(self, recalled: list[dict[str, Any]]) -> str:
        if recalled:
            memory_block = "\n".join(
                f"- {item.get('memory', item.get('data', ''))}" for item in recalled
            )
        else:
            memory_block = "（没有检索到相关长期记忆）"
        return (
            "你是一个自然、可靠的中文助手。下面是关于当前用户的长期记忆。"
            "仅在与当前问题相关时使用。遇到冲突时优先采用明确标记为更新、最新或现在的信息。"
            "不要把不同人物的属性混在一起，也不要向用户提及向量库或记忆系统。\n\n"
            f"长期记忆：\n{memory_block}"
        )

    def generate_answer(
        self,
        user_text: str,
        recalled: list[dict[str, Any]],
        include_recent: bool,
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt(recalled)}
        ]
        if include_recent:
            messages.extend(self.recent_messages[-12:])
        messages.append({"role": "user", "content": user_text})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=self.max_answer_tokens,
        )
        return response.choices[0].message.content or ""

    def ingest_turn(self, turn: dict[str, Any], index: int, total: int) -> None:
        turn_id = turn["id"]
        user_text = turn["text"]
        self.recorder.emit()
        self.recorder.emit(f"[写入 {index}/{total}] {turn_id}")
        self.recorder.emit(f"用户> {user_text}")
        record: dict[str, Any] = {"id": turn_id, "text": user_text}

        started = time.perf_counter()
        try:
            recalled = self.search(user_text)
            record["retrieval_seconds"] = time.perf_counter() - started
        except Exception as exc:
            recalled = []
            record["retrieval_seconds"] = time.perf_counter() - started
            record["retrieval_error"] = str(exc)
            self.errors.append({"stage": "retrieval", "turn": turn_id, "error": str(exc)})
        self.print_memories("[检索]", recalled)

        started = time.perf_counter()
        try:
            answer = self.generate_answer(user_text, recalled, include_recent=True)
            record["answer_seconds"] = time.perf_counter() - started
        except Exception as exc:
            answer = ""
            record["answer_seconds"] = time.perf_counter() - started
            record["answer_error"] = str(exc)
            self.errors.append({"stage": "answer", "turn": turn_id, "error": str(exc)})
        self.recorder.emit(f"助手> {answer or '（生成失败）'}")

        events: list[dict[str, Any]] = []
        started = time.perf_counter()
        if answer:
            try:
                result = self.memory.add(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": answer},
                    ],
                    user_id=self.user_id,
                )
                events = unwrap_results(result)
            except Exception as exc:
                record["write_error"] = str(exc)
                self.errors.append({"stage": "write", "turn": turn_id, "error": str(exc)})
        record["write_seconds"] = time.perf_counter() - started
        record["events"] = events
        for event in events:
            self.event_counts[str(event.get("event", "UNKNOWN"))] += 1
        if events:
            self.recorder.emit("[写入事件]")
            for event in events:
                text = event.get("memory", event.get("data", ""))
                self.recorder.emit(f"  - {event.get('event', 'UNKNOWN')}: {text}")
        else:
            self.recorder.emit("[写入事件] 无事件或写入失败")

        if answer:
            self.recent_messages.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": answer},
                ]
            )
        record["total_seconds"] = (
            record["retrieval_seconds"]
            + record["answer_seconds"]
            + record["write_seconds"]
        )
        self.recorder.emit(
            "[耗时] "
            f"检索={record['retrieval_seconds']:.3f}s "
            f"回答={record['answer_seconds']:.3f}s "
            f"写入={record['write_seconds']:.3f}s "
            f"总计={record['total_seconds']:.3f}s"
        )
        self.turn_records.append(record)

    @staticmethod
    def check_expected_groups(answer: str, groups: list[list[str]]) -> list[list[str]]:
        lowered = answer.lower()
        return [group for group in groups if not any(term.lower() in lowered for term in group)]

    def evaluate_turn(self, item: dict[str, Any], index: int, total: int) -> None:
        eval_id = item["id"]
        question = item["question"]
        self.recorder.emit()
        self.recorder.emit(f"[评测 {index}/{total}] {eval_id}")
        self.recorder.emit(f"问题> {question}")
        record: dict[str, Any] = {"id": eval_id, "question": question}

        started = time.perf_counter()
        try:
            recalled = self.search(question)
            record["retrieval_seconds"] = time.perf_counter() - started
        except Exception as exc:
            recalled = []
            record["retrieval_seconds"] = time.perf_counter() - started
            record["retrieval_error"] = str(exc)
            self.errors.append({"stage": "eval_retrieval", "turn": eval_id, "error": str(exc)})
        self.print_memories("[检索]", recalled)

        started = time.perf_counter()
        try:
            answer = self.generate_answer(question, recalled, include_recent=False)
            record["answer_seconds"] = time.perf_counter() - started
        except Exception as exc:
            answer = ""
            record["answer_seconds"] = time.perf_counter() - started
            record["answer_error"] = str(exc)
            self.errors.append({"stage": "eval_answer", "turn": eval_id, "error": str(exc)})
        self.recorder.emit(f"回答> {answer or '（生成失败）'}")

        expected_groups = item.get("expected_groups", [])
        missing = self.check_expected_groups(answer, expected_groups)
        passed = bool(answer) and not missing
        record.update(
            {
                "answer": answer,
                "recalled": recalled,
                "expected_groups": expected_groups,
                "missing_groups": missing,
                "passed": passed,
            }
        )
        self.recorder.emit(f"[结果] {'PASS' if passed else 'FAIL'}")
        if missing:
            self.recorder.emit(f"[缺失关键词组] {missing}")
        self.evaluation_records.append(record)

    def memory_audit(self) -> dict[str, Any]:
        memories = self.all_memories()
        texts = [str(item.get("memory", item.get("data", ""))) for item in memories]
        normalized = [normalize_text(text) for text in texts]
        counts = Counter(value for value in normalized if value)
        exact_duplicates = [
            {"normalized": key, "count": count, "examples": [text for text in texts if normalize_text(text) == key]}
            for key, count in counts.items()
            if count > 1
        ]

        corpus = "\n".join(texts).lower()
        expectations = self.case.get("memory_expectations", {})
        missing_groups = [
            group
            for group in expectations.get("must_contain_groups", [])
            if not any(term.lower() in corpus for term in group)
        ]
        forbidden_hits = [
            term
            for term in expectations.get("must_not_contain", [])
            if term.lower() in corpus
        ]
        return {
            "count": len(memories),
            "memories": memories,
            "exact_duplicates": exact_duplicates,
            "missing_required_groups": missing_groups,
            "forbidden_hits": forbidden_hits,
            "passed": not missing_groups and not forbidden_hits,
        }

    def timing_summary(self) -> dict[str, Any]:
        def summarize(key: str) -> dict[str, float]:
            values = [float(item[key]) for item in self.turn_records if key in item]
            if not values:
                return {"mean": 0.0, "median": 0.0, "max": 0.0}
            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "max": max(values),
            }

        return {
            "initialization_seconds": self.init_seconds,
            "retrieval": summarize("retrieval_seconds"),
            "answer": summarize("answer_seconds"),
            "write": summarize("write_seconds"),
            "total": summarize("total_seconds"),
        }

    def run(self, ingest_turns: list[dict[str, Any]], eval_turns: list[dict[str, Any]]) -> bool:
        overall_started = time.perf_counter()
        self.recorder.emit()
        self.recorder.emit(f"阶段一：写入 {len(ingest_turns)} 轮高信息量对话")
        for index, turn in enumerate(ingest_turns, 1):
            self.ingest_turn(turn, index, len(ingest_turns))

        self.recorder.emit()
        self.recorder.emit("阶段二：模拟新会话，清空所有短期聊天上下文")
        self.recent_messages.clear()
        self.recorder.emit("短期上下文已清空；评测问题不会写回 Mem0，避免答案自我污染。")

        self.recorder.emit()
        self.recorder.emit(f"阶段三：执行 {len(eval_turns)} 个独立召回问题")
        for index, item in enumerate(eval_turns, 1):
            self.evaluate_turn(item, index, len(eval_turns))

        audit = self.memory_audit()
        passed_evals = sum(1 for item in self.evaluation_records if item["passed"])
        total_evals = len(self.evaluation_records)
        score = passed_evals / total_evals if total_evals else 0.0
        elapsed = time.perf_counter() - overall_started
        report = {
            "case_name": self.case.get("name"),
            "run_id": self.run_id,
            "user_id": self.user_id,
            "model": self.model,
            "base_url": self.base_url,
            "memory_model": self.memory_model,
            "memory_base_url": self.memory_base_url,
            "search_threshold": self.search_threshold,
            "device": self.device,
            "top_k": self.top_k,
            "started_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": elapsed,
            "evaluation": {
                "passed": passed_evals,
                "total": total_evals,
                "score": score,
                "records": self.evaluation_records,
            },
            "memory_audit": audit,
            "event_counts": dict(self.event_counts),
            "timing": self.timing_summary(),
            "ingest_records": self.turn_records,
            "errors": self.errors,
        }
        report_path = self.run_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        self.recorder.emit()
        self.recorder.emit("=" * 76)
        self.recorder.emit("最终汇总")
        self.recorder.emit(f"召回答题：{passed_evals}/{total_evals} = {score:.1%}")
        self.recorder.emit(f"最终记忆数：{audit['count']}")
        self.recorder.emit(f"完全重复组数：{len(audit['exact_duplicates'])}")
        self.recorder.emit(f"缺失的必要记忆组：{audit['missing_required_groups'] or '无'}")
        self.recorder.emit(f"不应保存但被命中的内容：{audit['forbidden_hits'] or '无'}")
        self.recorder.emit(f"记忆事件统计：{dict(self.event_counts)}")
        self.recorder.emit(f"运行错误数：{len(self.errors)}")
        self.recorder.emit(f"总耗时：{elapsed:.1f}s")
        self.recorder.emit(f"完整报告：{report_path}")
        self.recorder.emit(f"逐轮记录：{self.recorder.path}")
        self.recorder.emit("=" * 76)
        self.print_memories("[最终长期记忆]", audit["memories"])
        return score == 1.0 and audit["passed"] and not self.errors

    def close(self) -> None:
        self.recorder.close()


def load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value.get("ingest_turns"), list) or not isinstance(
        value.get("evaluation_turns"), list
    ):
        raise ValueError("测试 JSON 必须包含 ingest_turns 和 evaluation_turns 数组")
    return value


def check_memory_api(base_url: str, api_key: str, model: str) -> None:
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        return
    if not api_key or api_key == "local":
        raise RuntimeError("远程记忆模型需要有效的 API key。")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.models.list()
    model_ids = {item.id for item in response.data}
    if model not in model_ids:
        available = ", ".join(sorted(model_ids)) or "（接口未返回模型）"
        raise RuntimeError(f"接口可访问，但没有找到模型 {model}。可用模型：{available}")
    print(f"记忆模型 API 检查通过：{model} @ {base_url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mem0 v2.0.20 多实体长程记忆一键测试")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model", default="qwen2.5-7b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--memory-model", default=None)
    parser.add_argument("--memory-base-url", default=None)
    parser.add_argument("--memory-api-key-env", default=None)
    parser.add_argument("--memory-max-tokens", type=int, default=2048)
    parser.add_argument("--memory-profile", type=Path, default=None)
    parser.add_argument("--data-subdir", default="new_long_tests")
    parser.add_argument("--collection-name", default="mem0_long_test")
    parser.add_argument("--search-threshold", type=float, default=0.1)
    parser.add_argument("--search-explain", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-answer-tokens", type=int, default=240)
    parser.add_argument("--quick", action="store_true", help="只运行标记为 quick 的小测试")
    parser.add_argument("--dry-run", action="store_true", help="只检查并打印用例，不加载模型")
    parser.add_argument("--strict", action="store_true", help="存在测试失败时返回非零退出码")
    parser.add_argument("--skip-memory-api-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    case_path = args.case.expanduser().resolve()
    configure_environment(root)
    case = load_case(case_path)
    ingest_turns = case["ingest_turns"]
    eval_turns = case["evaluation_turns"]
    if args.quick:
        ingest_turns = [item for item in ingest_turns if item.get("quick")]
        eval_turns = [item for item in eval_turns if item.get("quick")]
        if case.get("quick_memory_expectations"):
            case = dict(case)
            case["memory_expectations"] = case["quick_memory_expectations"]

    print(f"测试用例：{case.get('name')}")
    print(f"写入轮数：{len(ingest_turns)}")
    print(f"评测问题：{len(eval_turns)}")
    if args.dry_run:
        print("\n[写入场景]")
        for index, item in enumerate(ingest_turns, 1):
            print(f"{index:02d}. {item['id']}: {item['text']}")
        print("\n[评测问题]")
        for index, item in enumerate(eval_turns, 1):
            print(f"{index:02d}. {item['id']}: {item['question']}")
        return

    run_id = clean_run_id(args.run_id)
    memory_profile: dict[str, Any] = {}
    if args.memory_profile:
        profile_path = args.memory_profile.expanduser().resolve()
        with profile_path.open("r", encoding="utf-8") as handle:
            memory_profile = json.load(handle)
    memory_model = args.memory_model or memory_profile.get("model") or args.model
    memory_base_url = (
        args.memory_base_url
        or memory_profile.get("base_url")
        or f"{args.base_url.rstrip('/')}/v1"
    ).rstrip("/")
    memory_api_key_env = args.memory_api_key_env or memory_profile.get("api_key_env")
    memory_max_tokens = int(memory_profile.get("max_tokens", args.memory_max_tokens))
    memory_api_key = os.environ.get("OPENAI_API_KEY", "local")
    if memory_api_key_env:
        memory_api_key = os.environ.get(memory_api_key_env, "")
        if not memory_api_key:
            print(f"环境变量 {memory_api_key_env} 未设置。", file=sys.stderr)
            raise SystemExit(1)
    if not args.skip_memory_api_check:
        try:
            check_memory_api(memory_base_url, memory_api_key, memory_model)
        except Exception as exc:
            print(f"记忆模型 API 检查失败：{exc}", file=sys.stderr)
            raise SystemExit(1)
    app: LongMem0Test | None = None
    try:
        app = LongMem0Test(
            root=root,
            run_id=run_id,
            case=case,
            model=args.model,
            base_url=args.base_url,
            memory_model=memory_model,
            memory_base_url=memory_base_url,
            memory_api_key=memory_api_key,
            memory_max_tokens=memory_max_tokens,
            data_subdir=args.data_subdir,
            collection_name=args.collection_name,
            search_threshold=args.search_threshold,
            search_explain=args.search_explain,
            top_k=args.top_k,
            device=args.device,
            max_answer_tokens=args.max_answer_tokens,
        )
        passed = app.run(ingest_turns, eval_turns)
    except FileExistsError:
        print(
            f"运行目录已经存在：{DATA_ROOT / args.data_subdir / run_id}\n"
            "请更换 --run-id，避免把两次测试混在一起。",
            file=sys.stderr,
        )
        raise SystemExit(1)
    finally:
        if app is not None:
            app.close()
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
