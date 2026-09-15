"""Single-thread ownership of the memory SDK, with bounded foreground reads.

Mem0/local-Qdrant/history/embedding instances must not be called concurrently.
All SDK work (including lazy construction) runs on this daemon thread. Foreground
reads time out rather than wait behind consolidation; expired queued reads are
cancelled, not accumulated. SQLite job durability lives in session_store.py.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import CancelledError, Future, TimeoutError
from typing import Any, Callable

from bumi_mem.long_term import LongTermMemory
from bumi_mem.schemas import MemoryReadUnavailable, RetryableMemoryError


class SerialMemoryAccess:
    def __init__(self, factory: Callable[[], LongTermMemory], *, read_timeout: float = 0.2) -> None:
        if read_timeout <= 0:
            raise ValueError('read_timeout must be positive')
        self._factory = factory
        self.read_timeout = read_timeout
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._thread: threading.Thread | None = None

    def _start(self) -> None:
        with self._lock:
            if self._closed:
                raise MemoryReadUnavailable('memory executor is closed')
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='bumi-memory-sdk', daemon=True)
                self._thread.start()

    def _run(self) -> None:
        backend = None
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                future, method, kwargs = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    if backend is None:
                        try:
                            backend = self._factory()
                        except Exception as exc:
                            raise RetryableMemoryError(f"Memory initialization failed: {exc}") from exc
                    future.set_result(getattr(backend, method)(**kwargs))
                except BaseException as exc:
                    future.set_exception(exc)
        finally:
            if backend is not None:
                close = getattr(backend, 'close', None)
                if close is not None:
                    close()

    def _call(
        self,
        method: str,
        kwargs: dict[str, Any],
        *,
        read: bool = False,
        timeout: float | None = None,
    ) -> Any:
        try:
            self._start()
        except MemoryReadUnavailable as exc:
            if not read:
                raise RetryableMemoryError(str(exc)) from exc
            raise
        future: Future = Future()
        with self._lock:
            if self._closed:
                raise (MemoryReadUnavailable if read else RetryableMemoryError)('memory executor is closed')
            self._queue.put((future, method, kwargs))
        wait_timeout = timeout if timeout is not None else (self.read_timeout if read else None)
        try:
            return future.result(timeout=wait_timeout)
        except CancelledError as exc:
            raise RetryableMemoryError('queued memory call cancelled before execution') from exc
        except TimeoutError as exc:
            future.cancel()
            raise MemoryReadUnavailable('memory busy or read deadline exceeded') from exc

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._call('search', kwargs, read=True)

    def warmup(self, *, timeout: float, query: str) -> None:
        if timeout <= 0:
            raise ValueError('warmup timeout must be positive')
        self._call('warmup', {"query": query}, read=True, timeout=timeout)

    def get_all(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._call('get_all', kwargs, read=True)

    def add_session(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._call('add_session', kwargs)

    def remember(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._call('remember', kwargs)

    def get_memory(self, **kwargs: Any) -> dict[str, Any] | None:
        return self._call('get_memory', kwargs, read=True)

    def delete_all(self, **kwargs: Any) -> None:
        self._call('delete_all', kwargs)

    def close(self) -> None:
        """No unbounded join: an in-flight SDK call cannot be safely killed."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[0].cancel()
            self._queue.put(None)
