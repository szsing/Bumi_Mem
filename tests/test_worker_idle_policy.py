"""后台整理的前台让路策略：有前台活动时延后，空闲或超时后必须照常执行。"""
from __future__ import annotations

import time
from pathlib import Path

from bumi_mem.consolidator import ConsolidatorConfig, SessionConsolidator
from bumi_mem.long_term import FakeLongTermMemory
from bumi_mem.providers import FakeTextProvider
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import FixedTokenCounter
from bumi_mem.worker import MemoryWorker


def build_worker(tmp_path: Path, **worker_kwargs):
    store = SQLiteSessionStore(tmp_path / "db.sqlite3")
    session = store.create_session("u", "a")
    for index, role in enumerate(["user", "assistant", "user", "assistant"]):
        store.append_message(session.session_id, role=role, content=f"m{index}", token_count=1000)
    consolidator = SessionConsolidator(
        store=store,
        long_term=FakeLongTermMemory(),
        summary_provider=FakeTextProvider("local", ["摘要"]),
        memory_provider_name="local",
        memory_token_counter=FixedTokenCounter(),
        config=ConsolidatorConfig(local_chunk_tokens=15),
    )
    receipt = consolidator.submit(session.session_id, reason="manual_new")
    worker = MemoryWorker(consolidator, **worker_kwargs)
    return store, worker, receipt


def test_foreground_activity_defers_background_job(tmp_path: Path) -> None:
    store, worker, receipt = build_worker(tmp_path, idle_seconds=60, max_defer_seconds=600)
    worker.note_foreground()
    assert worker.run_once() is False
    assert store.get_job(receipt.job_id).status == "queued"
    worker.stop()


def test_background_job_runs_after_idle_window(tmp_path: Path) -> None:
    store, worker, receipt = build_worker(tmp_path, idle_seconds=0.05, max_defer_seconds=600)
    worker.note_foreground()
    assert worker.run_once() is False
    worker._last_foreground = time.monotonic() - 1.0  # 模拟已空闲
    assert worker.run_once() is True
    assert store.get_job(receipt.job_id).status == "succeeded"
    worker.stop()


def test_max_defer_keeps_long_conversations_from_starving_jobs(tmp_path: Path) -> None:
    store, worker, receipt = build_worker(tmp_path, idle_seconds=3600, max_defer_seconds=0.01)
    worker.note_foreground()
    time.sleep(0.05)  # 任务年龄超过 max_defer，即使一直有前台活动也要执行
    assert worker.run_once() is True
    assert store.get_job(receipt.job_id).status == "succeeded"
    worker.stop()


def test_default_policy_is_unchanged(tmp_path: Path) -> None:
    store, worker, receipt = build_worker(tmp_path)  # 不传参数 = 旧行为
    worker.note_foreground()
    assert worker.run_once() is True
    assert store.get_job(receipt.job_id).status == "succeeded"
    worker.stop()
