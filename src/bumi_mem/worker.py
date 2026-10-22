"""后台记忆整理工作线程：持久化队列、有限重试、可续租所有权。"""
from __future__ import annotations

import logging
import os
import threading
import time
from uuid import uuid4

from bumi_mem.consolidator import SessionConsolidator

logger = logging.getLogger(__name__)


class MemoryWorker:
    """后台记忆整理工作线程：轮询持久化作业队列，串行执行整理任务。

    特性：
    - 持久化队列：作业存储在 SQLite 中，进程重启后不丢失
    - 有限重试：max_attempts 控制最大重试次数，指数退避
    - 可续租所有权：作业被工作线程认领后定期续租，防止僵尸作业
    - 空闲让路：idle_seconds > 0 时，前台活跃期间推迟后台整理
    - 最大延迟：max_defer_seconds 防止长时间连续对话永久饿死整理
    """
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
        self._last_idle_scan = 0.0
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
        """判断是否应立即开始后台整理。

        空闲让路逻辑：
        - idle_seconds <= 0：始终立即执行（旧行为）
        - 前台空闲超过 idle_seconds：立即执行
        - 前台仍在活跃但作业已等待超过 max_defer_seconds：强制执行（防饿死）
        """
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
        if self.consolidator.config.backend == 'graphiti' and time.monotonic()-self._last_idle_scan >= 1:
            self.store.submit_idle_snapshots(settings=self.consolidator._settings())
            self._last_idle_scan = time.monotonic()
        if not self._should_start_now():
            return False
        # 没有待处理作业时不碰长期记忆：否则关闭预热（BUMI_MEMORY_WARMUP=off）后，
        # 队列空也会被 ensure_available() 立刻探测一次，违背"启动时不碰长期记忆"的初衷。
        if not self.store.has_pending_jobs():
            return False
        # 长期记忆不可用时不认领真实作业当探针：claim_job 一认领就会消耗 attempts，
        # 宕机期间反复这样做会把排队的整理/删除作业永久耗光重试次数。
        # consolidator.long_term 在独立构造（不经 BumiMemory 包装）时可能是裸的
        # LongTermMemory 实现，没有熔断能力——那种情况下沿用旧行为，不做门控。
        ensure_available = getattr(self.consolidator.long_term, 'ensure_available', None)
        if ensure_available is not None and not ensure_available():
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
        # 不撤销正在执行的调用的租约。硬进程退出时租约自然过期；
        # 中间写入检查点需要显式重放确认。
        if thread is None or not thread.is_alive():
            self.store.release_lease(self.owner)