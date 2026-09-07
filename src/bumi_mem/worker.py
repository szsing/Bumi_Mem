"""In-process daemon worker: durable queue, finite retries, renewable ownership."""
from __future__ import annotations

import logging
import threading
from uuid import uuid4

from bumi_mem.consolidator import SessionConsolidator

logger = logging.getLogger(__name__)


class MemoryWorker:
    def __init__(self, consolidator: SessionConsolidator, *, poll_seconds: float = 0.1,
                 lease_seconds: float = 30.0) -> None:
        if poll_seconds <= 0 or lease_seconds <= 0:
            raise ValueError('worker intervals must be positive')
        self.consolidator = consolidator
        self.store = consolidator.store
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
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
