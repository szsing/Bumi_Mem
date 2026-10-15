"""Synchronous orchestration bridge to one persistent Graphiti event loop."""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import TimeoutError

from bumi_mem.memory_access import SerialMemoryAccess
from bumi_mem.schemas import MemoryReadUnavailable


class AsyncMemoryAccess(SerialMemoryAccess):
    """Preserve the facade, not the legacy single-file SDK execution queue.

    SDK clients are created/used/closed on their owning loop. Foreground reads
    can proceed while ingestion awaits network I/O. Writes serialize in backend.
    """
    backend_name = 'graphiti'

    def __init__(self, factory, *, read_timeout=30.0):
        self._factory = factory
        self.read_timeout = read_timeout
        self._closed = False
        self._lock = threading.Lock()
        self._thread = None
        self._ready = threading.Event()
        self._backend = None
        self._loop = None

    def _start(self):
        with self._lock:
            if self._closed:
                raise MemoryReadUnavailable('memory executor is closed')
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='bumi-graphiti-loop', daemon=True)
                self._thread.start()
        self._ready.wait()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._init_lock = asyncio.Lock()
        self._ready.set()
        self._maintenance_task = self._loop.create_task(self._maintenance())
        try:
            self._loop.run_forever()
        finally:
            tasks = asyncio.all_tasks(self._loop)
            for task in tasks:
                task.cancel()
            self._loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    async def _get_backend(self):
        async with self._init_lock:
            if self._backend is None:
                self._backend = self._factory()
        return self._backend

    async def _maintenance(self):
        while True:
            await asyncio.sleep(5)
            if self._backend is not None:
                try:
                    await self._backend.build_due_communities()
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception('Graphiti maintenance failed; journal retained')

    async def _invoke(self, method, kwargs):
        backend = await self._get_backend()
        return await getattr(backend, method)(**kwargs)

    def _call(self, method, kwargs, *, read=False, timeout=None):
        self._start()
        with self._lock:
            if self._closed:
                raise MemoryReadUnavailable('memory executor is closed')
            future = asyncio.run_coroutine_threadsafe(self._invoke(method, kwargs), self._loop)
        try:
            return future.result(timeout=timeout if timeout is not None else self.read_timeout if read else None)
        except TimeoutError as exc:
            if read:
                future.cancel()
            raise MemoryReadUnavailable('Graphiti read deadline exceeded') from exc

    def debug(self, **kwargs):
        return self._call('debug', kwargs, read=True)

    async def _shutdown(self):
        self._maintenance_task.cancel()
        await asyncio.gather(self._maintenance_task, return_exceptions=True)
        active = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        if self._backend is not None:
            await self._backend.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if loop is not None:
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1)
