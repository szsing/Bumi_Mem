#!/usr/bin/env python3
"""LongMemEval Pair-6 评估脚本：在 6 个精选问题上评估 Mem0 记忆系统的性能。"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from importlib.metadata import version as package_version

SELECTED_IDS = {
    "knowledge-update": "852ce960",
    "multi-session": "ba358f49",
    "single-session-assistant": "71a3fd6b",
    "single-session-preference": "d6233ab6",
    "single-session-user": "8550ddae",
    "temporal-reasoning": "d01c6aa8",
}
CUTOFFS = [10, 20, 50, 200]


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def parse_date(value: str) -> datetime:
    import re
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", value).strip()
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)


def human_date(value: str) -> str:
    try:
        return parse_date(value).strftime("%A, %B %d, %Y")
    except Exception:
        return value


def pair_turns(session: list[dict]) -> list[list[dict]]:
    cleaned = [{"role": row["role"], "content": row["content"]} for row in session]
    return [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)]


def load_selected(dataset_path: Path) -> list[dict]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    wanted = set(SELECTED_IDS.values())
    selected = [row for row in data if row["question_id"] in wanted]
    if len(selected) != 6:
        raise RuntimeError(f"Expected 6 selected questions, found {len(selected)}")
    return sorted(selected, key=lambda row: row["question_type"])


def describe(selected: list[dict]) -> list[dict]:
    rows = []
    for item in selected:
        pairs = sum(len(pair_turns(session)) for session in item["haystack_sessions"])
        rows.append({
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "sessions": len(item["haystack_sessions"]),
            "messages": sum(len(session) for session in item["haystack_sessions"]),
            "pairs": pairs,
            "question": item["question"],
        })
    return rows


def build_memory(root: Path, run_dir: Path, model: str, key: str, base: str):
    os.environ["MEM0_TELEMETRY"] = "False"
    from mem0 import Memory
    return Memory.from_config({
        "version": "v1.1",
        "history_db_path": str(run_dir / "history.db"),
        "llm": {"provider": "openai", "config": {
            "model": model, "api_key": key, "openai_base_url": base,
            "temperature": 0.0, "max_tokens": 4096,
        }},
        "embedder": {"provider": "huggingface", "config": {
            "model": str(Path(os.getenv("MEM0_MODEL_ROOT", str(root.parent / "models"))) / "Qwen3-Embedding-0.6B"),
            "embedding_dims": 1024,
            "model_kwargs": {"device": "cpu", "local_files_only": True},
        }},
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "longmemeval_pair6",
            "embedding_model_dims": 1024,
            "path": str(run_dir / "qdrant"),
            "on_disk": True,
        }},
    })


def api_text(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    params = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content or ""


def clean_answer(value: str) -> str:
    value = re.sub(r"[<\[]mem_thinking[>\]].*?[<\[]/mem_thinking[>\]]", "", value, flags=re.DOTALL).strip()
    if "ANSWER:" in value:
        value = value.rsplit("ANSWER:", 1)[-1].strip()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Mem0 2.0.19 LongMemEval-S pair-level six-question pilot")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    root = args.root.resolve()
    selected = load_selected(args.dataset.resolve())
    manifest = describe(selected)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("total_pairs=", sum(row["pairs"] for row in manifest))
    if args.dry_run:
        return

    key = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    if not key or not base:
        raise RuntimeError("OPENAI_API_KEY and OPENAI_BASE_URL are required")

    results_root = Path(os.getenv("MEM0_RESULTS_ROOT", str(root.parent / "mem0-runtime" / "results"))).resolve()
    run_dir = results_root / "longmemeval_pair6" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {"completed_pairs": [], "completed_questions": []}
    completed_pairs = set(checkpoint.get("completed_pairs", []))
    completed_questions = set(checkpoint.get("completed_questions", []))

    benchmark_root = root / "src" / "memory-benchmarks"
    sys.path.insert(0, str(benchmark_root))
    from benchmarks.longmemeval.prompts import get_answer_generation_prompt, get_judge_prompt

    memory = build_memory(root, run_dir, model, key, base)
    client = OpenAI(api_key=key, base_url=base)
    pair_budget = args.max_pairs
    newly_processed = 0
    ingestion_rows = []
    ingestion_log = run_dir / "ingestion.jsonl"
    results = []

    for question in selected:
        qid = question["question_id"]
        user_id = f"longmemeval_{qid}_{args.run_id}"
        order = sorted(
            zip(question["haystack_session_ids"], question["haystack_dates"], question["haystack_sessions"]),
            key=lambda row: parse_date(row[1]),
        )
        question_failed = 0
        for session_index, (session_id, date_text, session) in enumerate(order):
            observation = parse_date(date_text).strftime("%Y-%m-%d")
            for pair_index, messages in enumerate(pair_turns(session)):
                pair_key = f"{qid}:s{session_index}:p{pair_index}"
                if pair_key in completed_pairs:
                    continue
                if pair_budget is not None and newly_processed >= pair_budget:
                    atomic_json(checkpoint_path, {"completed_pairs": sorted(completed_pairs), "completed_questions": sorted(completed_questions)})
                    print(f"Stopped at max-pairs={pair_budget}; resume with the same run-id")
                    return
                if any(not msg.get("content", "").strip() for msg in messages):
                    completed_pairs.add(pair_key)
                    continue
                prompt = (
                    f"This conversation occurred on {observation}. Resolve relative dates against this observation date. "
                    "Preserve exact entity, role, speaker attribution, updated state, and relevant assistant-provided information."
                )
                metadata = {
                    "question_id": qid,
                    "session_id": session_id,
                    "pair_index": pair_index,
                    "observation_date": observation,
                }
                started = time.perf_counter(); ok = False; error = None; event_count = 0; attempts = 0
                for attempt in range(1, args.retries + 1):
                    attempts = attempt
                    try:
                        response = memory.add(messages, user_id=user_id, metadata=metadata, prompt=prompt)
                        event_count = len((response or {}).get("results", [])); ok = True; error = None; break
                    except Exception as exc:
                        error = str(exc)
                        if attempt < args.retries:
                            time.sleep(2 * attempt)
                elapsed = time.perf_counter() - started
                ingest_row = {"pair_key": pair_key, "seconds": elapsed, "ok": ok, "attempts": attempts, "events": event_count, "error": error}
                ingestion_rows.append(ingest_row)
                with ingestion_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(ingest_row, ensure_ascii=False) + "\n")
                if ok:
                    completed_pairs.add(pair_key)
                else:
                    question_failed += 1
                newly_processed += 1
                atomic_json(checkpoint_path, {"completed_pairs": sorted(completed_pairs), "completed_questions": sorted(completed_questions)})
                print(f"[{newly_processed}] {pair_key} {elapsed:.2f}s attempts={attempts} events={event_count} ok={ok}", flush=True)
                if not ok:
                    print("Stopping after exhausted retries; rerun with the same run-id to resume", flush=True)
                    return

        completed_questions.add(qid)
        atomic_json(checkpoint_path, {"completed_pairs": sorted(completed_pairs), "completed_questions": sorted(completed_questions)})
        if args.ingest_only:
            continue

        started = time.perf_counter()
        search_response = memory.search(
            question["question"],
            top_k=args.top_k,
            filters={"user_id": user_id},
            threshold=args.threshold,
        )
        search_seconds = time.perf_counter() - started
        memories = search_response.get("results", []) if isinstance(search_response, dict) else search_response
        formatted = []
        for row in memories:
            metadata = row.get("metadata") or {}
            formatted.append({
                "memory": row.get("memory", ""),
                "score": row.get("score", 0),
                "id": row.get("id", ""),
                "created_at": metadata.get("observation_date") or row.get("created_at", ""),
                "session_id": metadata.get("session_id"),
            })

        cutoff_results = {}
        for cutoff in CUTOFFS:
            sliced = sorted(formatted[:cutoff], key=lambda row: row.get("created_at", ""))
            prompt = get_answer_generation_prompt(
                question=question["question"],
                search_results=sliced,
                question_date=human_date(question.get("question_date", "")),
                user_profile=None,
            )
            started = time.perf_counter()
            generated = clean_answer(api_text(client, model, prompt, 1200))
            answer_seconds = time.perf_counter() - started
            judge_prompt = get_judge_prompt(
                question_type=question["question_type"],
                question_id=qid,
                question=question["question"],
                answer=str(question["answer"]),
                response=generated,
                question_date=human_date(question.get("question_date", "")),
            )
            started = time.perf_counter()
            judge_raw = api_text(client, model, judge_prompt, 512)
            judge_seconds = time.perf_counter() - started
            correct = judge_raw.strip().lower().startswith("yes")
            retrieved_sessions = {row.get("session_id") for row in formatted[:cutoff] if row.get("session_id")}
            gold_sessions = set(question.get("answer_session_ids", []))
            recall_all = bool(gold_sessions) and gold_sessions.issubset(retrieved_sessions)
            cutoff_results[f"top_{cutoff}"] = {
                "score": 1 if correct else 0,
                "judgment": "PASS" if correct else "FAIL",
                "generated_answer": generated,
                "judge_raw": judge_raw,
                "memories_evaluated": len(sliced),
                "answer_seconds": answer_seconds,
                "judge_seconds": judge_seconds,
                "session_recall_all": recall_all,
            }
        result = {
            "question_id": qid,
            "question_type": question["question_type"],
            "question": question["question"],
            "ground_truth_answer": question["answer"],
            "question_date": question.get("question_date"),
            "is_abstention": qid.endswith("_abs"),
            "pairs_failed": question_failed,
            "search_seconds": search_seconds,
            "total_memories": len(formatted),
            "search_results": formatted,
            "cutoff_results": cutoff_results,
        }
        results.append(result)
        atomic_json(run_dir / f"result_{qid}.json", result)

    existing_results = []
    for question in selected:
        path = run_dir / f"result_{question['question_id']}.json"
        if path.exists():
            existing_results.append(json.loads(path.read_text(encoding="utf-8")))
    metrics = {}
    for cutoff in CUTOFFS:
        label = f"top_{cutoff}"
        by_type = defaultdict(list)
        scores = []
        recalls = []
        for row in existing_results:
            value = row["cutoff_results"][label]
            scores.append(value["score"])
            recalls.append(value["session_recall_all"])
            by_type[row["question_type"]].append(value["score"])
        type_accuracy = {key: sum(values) / len(values) for key, values in by_type.items()}
        metrics[label] = {
            "overall_accuracy": sum(scores) / len(scores) if scores else 0,
            "task_average_accuracy": statistics.mean(type_accuracy.values()) if type_accuracy else 0,
            "session_recall_all": sum(recalls) / len(recalls) if recalls else 0,
            "by_type": type_accuracy,
        }
    all_ingestion_rows = []
    if ingestion_log.exists():
        all_ingestion_rows = [json.loads(line) for line in ingestion_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {
        "run_id": args.run_id,
        "mem0_version": package_version("mem0ai"),
        "memory_model": model,
        "answerer_model": model,
        "judge_model": model,
        "judge_protocol": "LongMemEval prompt with DeepSeek custom judge",
        "ingest_granularity": "pair",
        "manifest": manifest,
        "ingestion": all_ingestion_rows,
        "metrics": metrics,
        "results": existing_results,
    }
    atomic_json(run_dir / "report.json", report)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("REPORT", run_dir / "report.json")


if __name__ == "__main__":
    main()