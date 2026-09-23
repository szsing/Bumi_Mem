"""In-process daemon worker: durable queue, finite retries, renewable ownership."""
from __future__ import annotations

import logging
import os
import threading
import time
from uuid import uuid4

from bumi_mem.consolidator import SessionConsolidator

logger = logging.getLogger(__name__)


class MemoryWorker:
    def __init__(self, consolidator: SessionConsolidator, *, poll_seconds: float = 0.1,
                 lease_seconds: float = 30.0, idle_seconds: float | None = None,
                 max_defer_seconds: float | None = None) -> None:
        """idle_seconds>0 时，后台整理只在最近 idle_seconds 秒内没有前台活动时才跑。

        前台检索和后台整理共用 SerialMemoryAccess 的同一个 SDK 线程，实测后台整理会让
        约 5% 的前台检索多等 0.6–1s（docs/latency/REPORT.md §8.3）。默认 0 = 保持旧行为；
        可用 BUMI_CONSOLIDATION_IDLE_SECONDS / BUMI_CONSOLIDATION_MAX_DEFER_SECONDS 配置。
        max_defer_seconds 防止长时间连续对话把整理永久饿死。
        """
        if idle_seconds is None:
            idle_seconds = float(os.getenv("BUMI_CONSOLIDATION_IDLE_SECONDS", "0"))
        if max_defer_seconds is None:
            max_defer_seconds = float(os.getenv("BUMI_CONSOLIDATION_MAX_DEFER_SECONDS", "60"))
        if poll_seconds <= 0 or lease_seconds <= 0:
            raise ValueError('worker intervals must be positive')
        if idle_seconds < 0 or max_defer_seconds <= 0:
            raise ValueError('worker idle/defer windows must be non-negative and finite')
        self.consolidator = consolidator
        self.store = consolidator.store
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.idle_seconds = idle_seconds
        self.max_defer_seconds = max_defer_seconds
        self._last_foreground = 0.0
        self.owner = uuid4().hex
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError('worker has been stopped')
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='bumi-memory-worker', daemon=True)
                self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def note_foreground(self) -> None:
        """前台读写发生后调用；后台整理据此让路。"""
        self._last_foreground = time.monotonic()

    def _should_start_now(self) -> bool:
        if self.idle_seconds <= 0:
            return True
        if self._last_foreground and (time.monotonic() - self._last_foreground) < self.idle_seconds:
            age = self.store.oldest_due_job_age_seconds()
            if age is not None and age < self.max_defer_seconds:
                return False
        return True

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    did_work = self.run_once()
                except Exception:
                    logger.exception('memory worker iteration failed; durable jobs retained')
                    did_work = False
                if not did_work:
                    self._wake.wait(self.poll_seconds)
                    self._wake.clear()
        finally:
            self.store.release_lease(self.owner)

    def run_once(self) -> bool:
        """Also usable in deterministic tests with no sleeps or model services."""
        if self._stop.is_set():
            return False
        if not self._should_start_now():
            return False
        job = self.store.claim_job(self.owner, lease_seconds=self.lease_seconds)
        if job is None:
            return False
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(max(0.01, self.lease_seconds / 3)):
                try:
                    if not self.store.renew_lease(self.owner, lease_seconds=self.lease_seconds):
                        return
                except Exception:
                    logger.exception('memory lease renewal failed')

        pulse = threading.Thread(target=heartbeat, name='bumi-memory-lease', daemon=True)
        pulse.start()
        try:
            self.consolidator.execute(job, self.owner)
        finally:
            heartbeat_stop.set()
            pulse.join(timeout=1.0)
        return True

    def stop(self, *, timeout: float = 0.2) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        # Do not revoke a live call's lease. On hard process exit it expires;
        # a mid-write checkpoint will then require explicit replay confirmation.
        if thread is None or not thread.is_alive():
            self.store.release_lease(self.owner)
