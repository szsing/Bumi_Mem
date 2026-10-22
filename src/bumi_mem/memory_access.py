"""单线程 SDK 所有权：带界前台读取超时，并维护长期记忆可用性熔断。

Mem0/local-Qdrant/history/embedding 实例不得并发调用。
所有 SDK 工作（包括惰性构造）在此守护线程上运行。前台读取超时而非
等待整理；过期的排队读取被取消而非累积。SQLite 作业持久化在 session_store.py。

熔断三态（unchecked/ready/degraded）：读路径（BumiMemory.search_long_term）和
写路径（MemoryWorker/SessionConsolidator 的 remember/add_session/delete_all）
都经过这一层，所以熔断状态放在这里，两条路径自动共享同一套"冷却期 + 半开探测"保护，
不需要在上层各写一份。
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import traceback
from concurrent.futures import CancelledError, Future, TimeoutError
from typing import Any, Callable

from bumi_mem.long_term import LongTermMemory
from bumi_mem.schemas import MemoryReadUnavailable, RetryableMemoryError, ProviderConfigurationError, MemoryScopeError

logger = logging.getLogger(__name__)

try:
    import httpx
    _TRANSPORT_ERRORS: tuple = (httpx.TransportError,)
except ImportError:  # httpx 是 qdrant-client 的传递依赖，防御性处理
    _TRANSPORT_ERRORS = ()

try:
    import torch
    # AcceleratorError（torch 2.x 新增，覆盖 OOM 之外更多 CUDA/加速器故障）和
    # OutOfMemoryError 都用 getattr 兜底：老版本 PyTorch 可能没有前者。
    _CUDA_ERRORS: tuple = tuple(
        cls for cls in (getattr(torch, 'AcceleratorError', None), getattr(torch.cuda, 'OutOfMemoryError', None))
        if cls is not None)
except ImportError:
    _CUDA_ERRORS = ()

# 只挑网络相关的 OSError 子类（连接失败/超时），不能用裸 OSError——它也覆盖
# PermissionError/FileNotFoundError 这类配置/环境问题，那些该 fail-fast，不该被当成"等恢复"。
_KNOWN_AVAILABILITY = (ConnectionError, TimeoutError, *_TRANSPORT_ERRORS)

# RuntimeError 不能加入这个白名单：BumiMemoryError（含 RetryableMemoryError/MemoryReadUnavailable）
# 本身就继承自 RuntimeError，加了会把真正的可用性错误也挡成 fail-fast。
# torch 的 CUDA 异常同理是 RuntimeError 子类，必须单独列出。
_KNOWN_PERMANENT = (ProviderConfigurationError, MemoryScopeError, ValueError, TypeError,
                    AttributeError, AssertionError, NotImplementedError,
                    PermissionError, FileNotFoundError, IsADirectoryError, NotADirectoryError,
                    *_CUDA_ERRORS)


def _iter_causes(exc: BaseException):
    """沿异常链展开：优先用 qdrant ResponseHandlingException 的 `.source` 属性
    （它不是标准 `raise ... from e`，原始异常不在 __cause__ 里），否则走标准
    __cause__/__context__ 链。"""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        source = getattr(current, 'source', None)
        current = source if isinstance(source, BaseException) else (current.__cause__ or current.__context__)


def _cause_chain_names(exc: BaseException) -> str:
    """异常链上每一层的类型名+抛出位置（文件名:行号），用于日志定位根因（网络/CUDA/SDK）。

    不含 str(exc)：异常消息可能带连接串/凭据片段，位置信息（代码在哪一行抛的）足够
    诊断根因，且不属于运行时数据，不会泄露敏感信息。
    """
    parts = []
    for cause in _iter_causes(exc):
        location = ''
        if cause.__traceback__ is not None:
            frames = traceback.extract_tb(cause.__traceback__)
            if frames:
                frame = frames[-1]
                location = f'@{os.path.basename(frame.filename)}:{frame.lineno}'
        parts.append(f'{type(cause).__name__}{location}')
    return ' <- '.join(parts)


def _is_fail_fast(exc: BaseException, *, lenient_unknown: bool) -> bool:
    """判断一次失败是否为永久性错误（配置/程序错误），不应被当作"长期记忆暂时不可用"处理。

    lenient_unknown=True（warmup/可用性探测路径）时，无法识别的异常类型默认当作可用性问题
    （这类调用本来就是为了探测后端是否活着）；其它调用路径对无法识别的异常类型保守处理，
    不翻转全局熔断状态，但这一次调用该怎么失败仍然怎么失败。
    """
    for cause in _iter_causes(exc):
        if isinstance(cause, MemoryReadUnavailable):
            return False
        if isinstance(cause, _KNOWN_AVAILABILITY):
            return False
        if isinstance(cause, _KNOWN_PERMANENT):
            return True
    return not lenient_unknown


class SerialMemoryAccess:
    """单线程 SDK 所有权管理器：在守护线程上串行执行所有 Mem0 SDK 调用。

    Mem0/local-Qdrant/history/embedding 实例不得并发调用，因此所有 SDK 工作
    （包括惰性构造）都在此守护线程上运行。前台读取带超时，不等待后台整理；
    过期的排队读取被取消而非累积。

    副作用:
        启动守护线程；在守护线程上惰性构造 LongTermMemory 实例。
    """
    def __init__(self, factory: Callable[[], LongTermMemory], *, read_timeout: float = 0.2,
                 degrade_cooloff_seconds: float = 60.0, degrade_probe_timeout: float = 5.0,
                 degrade_probe_query: str = "记忆系统预热") -> None:
        """初始化串行访问管理器。

        参数:
            factory:                  延迟构造 LongTermMemory 实例的工厂函数
            read_timeout:             读取操作超时秒数（默认 0.2s）
            degrade_cooloff_seconds:  进入 degraded 后，多久才允许下一次探测（默认60s）
            degrade_probe_timeout:    周期性重新探测用的超时（默认5s，不复用 warmup 的120s，
                                      否则宕机期间每次探测都要等满2分钟，拖住整条串行队列）
            degrade_probe_query:      探测用的查询文本

        副作用:
            首次 _call 时启动守护线程。
        """
        if read_timeout <= 0:
            raise ValueError('read_timeout must be positive')
        self._factory = factory
        self.read_timeout = read_timeout
        self.degrade_cooloff_seconds = degrade_cooloff_seconds
        self.degrade_probe_timeout = degrade_probe_timeout
        self.degrade_probe_query = degrade_probe_query
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._status = 'unchecked'
        self._degraded_since: float | None = None
        self._degradation_lock = threading.Lock()
        # 调用结果按"派发顺序"（seq，和守护线程实际执行顺序一致）生效，不是按"调用方线程
        # 被唤醒去处理结果"的顺序——否则一个更早派发、更晚被调度的成功，可能在一个更晚派发、
        # 先被调度的失败之后，把刚写入的 degraded 状态覆盖回去。
        self._outcome_lock = threading.Lock()
        self._next_seq = 0
        self._applied_seq = -1

    @property
    def status(self) -> str:
        """长期记忆可用性三态：'unchecked'（从未探测过）/'ready'/'degraded'。"""
        return self._status

    @property
    def degraded(self) -> bool:
        return self._status == 'degraded'

    @property
    def degraded_since(self) -> float | None:
        return self._degraded_since

    def is_in_cooloff(self) -> bool:
        """是否仍处于 degraded 后的冷却期内（冷却期内不应发起任何探测）。"""
        return self._status == 'degraded' and self._degraded_since is not None and (
            time.monotonic() - self._degraded_since < self.degrade_cooloff_seconds)

    def _record_outcome(self, seq: int, status: str, degraded_since: float | None) -> None:
        """按派发顺序（seq）应用一次真正反映可用性的调用结果（成功→ready，可用性类失败→degraded）。

        seq 更小（更旧）的调用不能覆盖已经生效的更新结果。只应在调用确实提供了可用性信息时
        调用这个方法——fail-fast 的永久错误不提供这类信息，调用方不应为它分配/占用 seq 位
        （见 _call 里的说明），否则会错误地让一个真正更新的可用性结果被当成"过期"丢弃。
        """
        with self._outcome_lock:
            if seq <= self._applied_seq:
                return
            self._applied_seq = seq
            self._status = status
            self._degraded_since = degraded_since

    def ensure_available(self) -> bool:
        """供后台写入路径（MemoryWorker）在认领真实作业前调用。

        不拿真实作业当探针：真正不确定时发起一次专门的 warmup 探测（走下面 _call
        的半开锁，和读路径共享，不会重复探测）。已识别的永久性错误（配置/程序错误）
        不吞成 False——那样会把真正需要人介入的问题永远当成"等恢复"。
        """
        if self._status == 'ready':
            return True
        if self._status == 'degraded' and self.is_in_cooloff():
            return False
        try:
            self.warmup(timeout=self.degrade_probe_timeout, query=self.degrade_probe_query)
        except Exception as exc:
            if _is_fail_fast(exc, lenient_unknown=True):
                raise
            logger.warning('long-term memory recovery probe failed: %s', _cause_chain_names(exc))
            return False
        return True

    def _start(self) -> None:
        """确保守护线程已启动。若已关闭则抛 MemoryReadUnavailable。"""
        with self._lock:
            if self._closed:
                raise MemoryReadUnavailable('memory executor is closed')
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='bumi-memory-sdk', daemon=True)
                self._thread.start()

    def _run(self) -> None:
        """守护线程主循环：从队列取任务并串行执行。

        首次执行时惰性构造后端实例。收到 None 哨兵时退出并关闭后端。
        """
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
        """向守护线程提交一个 SDK 调用并等待结果。

        参数:
            method:  后端方法名（如 'search'、'add_session'、'warmup'）
            kwargs:  方法关键字参数
            read:    是否为读取操作（读取操作带超时，写入操作无超时）
            timeout: 自定义超时秒数（仅读取操作有效）

        返回:
            后端方法的返回值。

        副作用:
            - 将任务放入队列，由守护线程执行
            - 读取超时时取消 future 并抛 MemoryReadUnavailable
            - 队列关闭时抛 RetryableMemoryError（写入）或 MemoryReadUnavailable（读取）
            - 非 'ready' 状态下短路跳过真实调用，或作为半开探测更新熔断三态
        """
        try:
            self._start()
        except MemoryReadUnavailable as exc:
            if not read:
                raise RetryableMemoryError(str(exc)) from exc
            raise
        probing = False
        if self._status != 'ready':
            if self._status == 'degraded' and self.is_in_cooloff():
                raise (MemoryReadUnavailable if read else RetryableMemoryError)(
                    'long-term memory degraded; skipping call during cool-off')
            # unchecked 或冷却期已过的 degraded：只放一个请求去真正探测，
            # 抢不到锁的并发请求直接短路，不会一起打到后端。
            if not self._degradation_lock.acquire(blocking=False):
                raise (MemoryReadUnavailable if read else RetryableMemoryError)(
                    'long-term memory availability probe already in progress')
            probing = True
        seq = None
        try:
            future: Future = Future()
            with self._lock:
                if self._closed:
                    raise (MemoryReadUnavailable if read else RetryableMemoryError)('memory executor is closed')
                # seq 在和 queue.put 同一把锁下分配，和入队顺序一一对应——
                # 入队顺序就是守护线程真正执行的顺序（单线程、FIFO）。
                seq = self._next_seq
                self._next_seq += 1
                self._queue.put((future, method, kwargs))
            wait_timeout = timeout if timeout is not None else (self.read_timeout if read else None)
            try:
                result = future.result(timeout=wait_timeout)
            except CancelledError as exc:
                raise RetryableMemoryError('queued memory call cancelled before execution') from exc
            except TimeoutError as exc:
                future.cancel()
                raise MemoryReadUnavailable('memory busy or read deadline exceeded') from exc
        except Exception as exc:
            if seq is not None and not self._closed:
                if not _is_fail_fast(exc, lenient_unknown=(method == 'warmup')):
                    self._record_outcome(seq, 'degraded', time.monotonic())
                # 永久错误（fail-fast）不提供任何可用性信息，故意不调用 _record_outcome：
                # 它不该占用 seq 位——否则一个更早派发、真正反映可用性的失败（更小的 seq），
                # 在它之后才被调用方应用时，会被这次"路过"的无关错误错误地判定成"更旧、该丢弃"。
            raise
        else:
            self._record_outcome(seq, 'ready', None)
            return result
        finally:
            if probing:
                self._degradation_lock.release()

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        """搜索长期记忆（读取操作，带默认超时）。"""
        return self._call('search', kwargs, read=True)

    def search_with_timeout(self, *, timeout: float, **kwargs: Any) -> list[dict[str, Any]]:
        """搜索长期记忆（读取操作，带自定义超时）。

        参数:
            timeout: 超时秒数，必须为正数
        """
        if timeout <= 0:
            raise ValueError('search timeout must be positive')
        return self._call('search', kwargs, read=True, timeout=timeout)

    def warmup(self, *, timeout: float, query: str) -> None:
        """预热后端（读取操作，带自定义超时）。

        参数:
            timeout: 超时秒数，必须为正数
            query:   预热查询文本
        """
        if timeout <= 0:
            raise ValueError('warmup timeout must be positive')
        self._call('warmup', {"query": query}, read=True, timeout=timeout)

    def get_all(self, **kwargs: Any) -> list[dict[str, Any]]:
        """获取全部长期记忆（读取操作，带默认超时）。"""
        return self._call('get_all', kwargs, read=True)

    def add_session(self, **kwargs: Any) -> list[dict[str, Any]]:
        """添加会话记忆（写入操作，无超时等待完成）。"""
        return self._call('add_session', kwargs)

    def remember(self, **kwargs: Any) -> list[dict[str, Any]]:
        """显式写入长期记忆（写入操作，无超时等待完成）。"""
        return self._call('remember', kwargs)

    def get_memory(self, **kwargs: Any) -> dict[str, Any] | None:
        """按 ID 获取单条记忆（读取操作，带默认超时）。"""
        return self._call('get_memory', kwargs, read=True)

    def delete_all(self, **kwargs: Any) -> None:
        """删除全部长期记忆（写入操作，无超时等待完成）。"""
        self._call('delete_all', kwargs)

    def close(self) -> None:
        """关闭串行访问管理器：取消所有排队任务，发送退出哨兵。

        不做无界 join：正在执行的 SDK 调用无法安全终止。
        """
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