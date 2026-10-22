"""Mem0 后端 L3 长期记忆适配器。"""

from __future__ import annotations

import os
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator, Protocol, Sequence
from uuid import uuid4

from bumi_mem.identity import require_pair
from bumi_mem.providers import ProviderConfig
from bumi_mem.provenance import extraction_provenance
from bumi_mem.schemas import (MemoryWriteUncertainError, RetryableMemoryError, MemoryScopeError,
                              ProviderConfigurationError)

# 接收 {"id", "memory", "user_id", "agent_id", "metadata"} 行视图，
# 返回仍允许作为抽取上下文的行（例如未撤回、未被 source block）。
ExistingMemoryFilter = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]

# Mem0 2.0.20 默认只请求 10 条已有记忆。逻辑删除或 generation 不匹配的记录
# 可能占满这个前缀，因此逐步扩大候选范围，同时设置延迟与候选数量硬上限。
_EXISTING_SEARCH_MAX_TOP_K = 100
_ENTITY_PURGE_LIMIT = 10_000


def unwrap_results(value: Any) -> list[dict[str, Any]]:
    """从 Mem0 返回值中提取结果列表，兼容 dict 和 list 两种返回格式。"""
    if isinstance(value, dict):
        results = value.get("results", [])
        return results if isinstance(results, list) else []
    return value if isinstance(value, list) else []


def resolve_embed_device(requested: str, *, cuda_available: bool | None = None) -> str:
    """解析嵌入模型设备偏好，返回实际可用的设备名。

    参数:
        requested:       用户请求的设备，"auto"/"cpu"/"cuda"
        cuda_available:  可选的 CUDA 可用性覆写；None 时自动检测 PyTorch

    返回:
        "cpu" 或 "cuda"

    副作用:
        当 cuda_available 为 None 时会 import torch 检测 CUDA。

    异常:
        ValueError:                 requested 不在允许值中
        ProviderConfigurationError: 显式请求 cuda 但环境不支持（永久性配置问题，
                                    不是"长期记忆暂时不可用"，调用方应 fail-fast）
    """
    requested = requested.strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("embed device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"
    if cuda_available is None:
        try:
            import torch
        except ImportError as exc:
            if requested == "cuda":
                raise ProviderConfigurationError("CUDA embedding requested but PyTorch is not installed") from exc
            return "cpu"
        cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise ProviderConfigurationError(
            "CUDA embedding requested but this Python environment has no CUDA-enabled PyTorch"
        )
    return "cuda" if cuda_available else "cpu"


def warmup_mem0_backend(memory: Any, *, query: str) -> None:
    """预热 Mem0 后端：执行一次完整搜索路径，不使用真实用户命名空间。

    用途：启动时验证嵌入模型和向量库是否可用，避免首次真实请求时才发现故障。

    参数:
        memory: Mem0 Memory 实例
        query:   预热查询文本，不能为空

    副作用:
        调用 memory.search()，会触发嵌入编码和向量检索。
    """
    query = query.strip()
    if not query:
        raise ValueError("warmup query must not be empty")
    memory.search(
        query=query,
        top_k=1,
        filters={"run_id": f"__bumi_internal_warmup__{uuid4().hex}"},
        threshold=0.0,
        rerank=False,
    )


class LongTermMemory(Protocol):
    """L3 长期记忆协议：定义搜索、添加、删除、获取等接口。

    实现可以是 Mem0 适配器或其他向量存储后端。
    """
    def warmup(self, *, query: str) -> None: ...

    def search(
        self, *, user_id: str, agent_id: str, query: str, top_k: int, threshold: float
    ) -> list[dict[str, Any]]: ...

    def add_session(
        self, *, user_id: str, agent_id: str, messages: Sequence[dict[str, str]],
        metadata: dict[str, Any], prompt: str,
        replay_authorization: dict[str, Any] | None = None,
        existing_memory_filter: ExistingMemoryFilter | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_all(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]: ...

    def delete_all(self, *, user_id: str, agent_id: str) -> None: ...

    def remember(self, *, user_id: str, agent_id: str, contents: Sequence[str], metadata: dict[str, Any]) -> list[dict[str, Any]]: ...

    def get_memory(self, *, user_id: str, agent_id: str, memory_id: str) -> dict[str, Any] | None: ...


@dataclass(slots=True, frozen=True)
class Mem0AdapterConfig:
    """Mem0 适配器配置：向量库路径、嵌入模型和维度等。

    属性:
        data_dir:           数据根目录（Qdrant 本地存储、history.db 均在此下）
        embed_model:        嵌入模型本地路径
        embed_device:       嵌入设备偏好，"auto"/"cpu"/"cuda"
        collection_name:    Qdrant 集合名
        embedding_dims:     嵌入向量维度
        memory_max_tokens:  Mem0 LLM 单次最大 token 数
    """
    data_dir: Path
    embed_model: Path
    embed_device: str = "auto"
    collection_name: str = "bumi_mem_l3"
    embedding_dims: int = 1024
    memory_max_tokens: int = 4096


def _vector_store_config(config: Mem0AdapterConfig) -> dict[str, Any]:
    """Qdrant 后端配置：默认本地 path client；设置 BUMI_QDRANT_URL 时改用 Server。

    本地 path client 是单进程精确扫描，只适合开发和小数据；生产/验收用
    BUMI_QDRANT_URL=http://127.0.0.1:6333 走服务端 HNSW 与 payload 索引。
    """
    store: dict[str, Any] = {
        "collection_name": config.collection_name,
        "embedding_model_dims": config.embedding_dims,
        "on_disk": True,
    }
    url = os.environ.get("BUMI_QDRANT_URL", "").strip()
    if url:
        store["url"] = url
    else:
        store["path"] = str(config.data_dir / "qdrant")
    return store


def _apply_entity_boost_policy(memory, *, mode: str | None = None) -> str:
    """实体加成开关：BUMI_ENTITY_BOOST=on|off。

    本地实测（docs/latency/REPORT.md §3.8 与 retrieval-ablation）：实体加成在中文语料上
    对 Recall/MRR 没有贡献，却要在检索里多付一次批量编码（GPU 下 P90 26–30ms）。
    off 只跳过检索热路径上的加成计算，实体库仍照常写入，随时可切回 on。
    """
    mode = (mode or os.environ.get("BUMI_ENTITY_BOOST", "on")).strip().lower()
    if mode not in {"on", "off"}:
        raise ValueError("BUMI_ENTITY_BOOST must be 'on' or 'off'")
    if mode == "off":
        if not hasattr(memory, "_compute_entity_boosts"):
            # 这是 mem0 的内部方法；升级 SDK 后若改名/移除，必须显式失败，
            # 否则会出现“以为关了、实际还在算”的静默时延回退。
            # 用 ProviderConfigurationError（不是裸 RuntimeError）：这是永久性的 SDK 版本
            # 不兼容问题，等多久都不会自己好，不能被长期记忆降级机制当成"暂时不可用"吞掉。
            raise ProviderConfigurationError(
                "BUMI_ENTITY_BOOST=off 需要 mem0 的 Memory._compute_entity_boosts；"
                "当前 mem0 版本不匹配，请复核 src/bumi_mem/long_term.py 的适配")
        memory._compute_entity_boosts = lambda query_entities, filters: {}

        async def _skip_async(query_entities, filters):
            return {}

        if hasattr(memory, "_compute_entity_boosts_async"):
            memory._compute_entity_boosts_async = _skip_async
    return mode


def _list_rows(listed: Any) -> list[Any]:
    """统一 vector_store.list() 的返回格式：某些提供者返回 [rows]，某些直接返回 rows。"""
    if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], (list, tuple)):
        return list(listed[0])
    return list(listed or [])


def _row_owned_by(row: Any, user_id: str, agent_id: str) -> bool:
    """判断向量行是否属于指定的 user_id/agent_id 作用域。"""
    payload = getattr(row, "payload", None) or {}
    if not isinstance(payload, dict):
        return False
    return (payload.get("user_id"), payload.get("agent_id")) == (user_id, agent_id)


@dataclass
class _ActiveScope:
    """当前活跃的长期记忆作用域：跟踪一次 add_session 调用中的检索和过滤状态。

    属性:
        user_id:          用户 ID
        agent_id:         Agent ID
        run_id:           当前 run 标识（隔离 SDK 的 last-k 缓存）
        generation:       作用域代数（用于过滤旧代记忆）
        existing_filter:  外部已有记忆过滤器
        hook_hits:        各 hook 被命中次数
        existing:         最终返回的已有记忆列表
        screened_out:     被过滤掉的记录数
        requested_top_k:  请求的 top_k
        fetched_top_k:    实际拉取的 top_k
        filtered_count:   因代数/撤回被过滤的记录数
        returned_count:   最终返回的记录数
        fetch_attempts:   拉取尝试次数
    """
    user_id: str
    agent_id: str
    run_id: str
    generation: int
    existing_filter: ExistingMemoryFilter | None
    hook_hits: Counter = field(default_factory=Counter)
    existing: list[dict[str, Any]] = field(default_factory=list)
    screened_out: int = 0
    requested_top_k: int | None = None
    fetched_top_k: int | None = None
    filtered_count: int = 0
    returned_count: int = 0
    fetch_attempts: int = 0

    def run_filters(self) -> dict[str, str]:
        """返回包含 run_id 的 SDK 过滤器（用于 Mem0 的 last-k 缓存隔离）。"""
        return {"user_id": self.user_id, "agent_id": self.agent_id, "run_id": self.run_id}

    def long_term_filters(self) -> dict[str, str]:
        """返回跨 run 的长期记忆过滤器（user_id + agent_id，不含 run_id）。"""
        return {"user_id": self.user_id, "agent_id": self.agent_id}

    def diagnostics(self) -> dict[str, Any]:
        """返回当前作用域的诊断信息，供日志和调试使用。"""
        hooks = {name: self.hook_hits.get(name, 0)
                 for name in ("vector_search", "entity_list", "entity_search_batch")}
        return {"user_id": self.user_id, "agent_id": self.agent_id, "run_id": self.run_id,
                "existing_scope": self.long_term_filters(), "existing_memories": list(self.existing),
                "screened_out": self.screened_out, "requested_top_k": self.requested_top_k,
                "fetched_top_k": self.fetched_top_k, "filtered_count": self.filtered_count,
                "returned_count": self.returned_count, "fetch_attempts": self.fetch_attempts,
                "hook_hits": hooks}


class _LongTermScope:
    """将说话人的长期作用域与 Mem0 的 run/session 作用域分离。

    Mem0 2.0.20 add(infer=True) 用 search_filters = user_id+agent_id+run_id 读取已有记忆
    （同时供哈希去重）和实体。在 add_session 执行期间，这些读取需要扩展到
    user_id+agent_id（跨 run）。Mem0 的 last-k 消息使用独立的 session_scope 字符串，
    因此保持当前 run 不变。
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory
        self.active: _ActiveScope | None = None
        self._activation_guard = Lock()
        # 拦截向量库的 search 调用，将 run 作用域扩展为长期作用域
        self._wrap(memory.vector_store, {"search": ("vector_search", self._search_existing)})

    def _wrap(self, store: Any, methods: dict[str, tuple[str, Callable[..., Any]]]) -> None:
        """对存储对象的方法进行猴子补丁，注入作用域扩展逻辑。已包装过则跳过。"""
        if getattr(store, "_bumi_long_term_scope", None) is self:
            return
        for name, (hook_name, handler) in methods.items():
            setattr(store, name, self._widening(hook_name, getattr(store, name), handler))
        store._bumi_long_term_scope = self

    def _widening(self, hook_name: str, original: Callable[..., Any], handler: Callable[..., Any]):
        """返回一个包装函数：当活跃作用域匹配时，将 run 过滤器扩展为长期过滤器并调用 handler。"""
        def call(*args, **kwargs):
            active = self.active
            if active is None or kwargs.get("filters") != active.run_filters():
                return original(*args, **kwargs)
            kwargs["filters"] = active.long_term_filters()
            active.hook_hits[hook_name] += 1
            return handler(active, original, args, kwargs)
        return call

    @staticmethod
    def _owned(row: Any, active: _ActiveScope) -> bool:
        """判断行是否属于活跃作用域的 user_id/agent_id。"""
        return _row_owned_by(row, active.user_id, active.agent_id)

    def _filter_existing(self, rows: Any, active: _ActiveScope) -> list[Any]:
        """过滤已有记忆行：按代数和外部过滤器筛选，确保作用域安全。

        步骤:
        1. 验证所有行都属于当前说话人作用域
        2. 按 generation 过滤：只保留与当前代数匹配的记录
        3. 如果有外部 existing_filter，进一步过滤（例如排除已撤回的记录）
        """
        rows = list(rows or [])
        if not all(self._owned(r, active) for r in rows):
            raise MemoryScopeError("existing-memory retrieval crossed the speaker scope")
        kept = []
        for row in rows:
            generation = (row.payload or {}).get("memory_generation")
            if generation is None:
                allowed_generation = active.generation == 0
            else:
                allowed_generation = (isinstance(generation, int)
                                      and not isinstance(generation, bool)
                                      and generation == active.generation)
            if allowed_generation:
                kept.append(row)
        if active.existing_filter is not None and kept:
            views = [{"id": str(r.id), "memory": r.payload.get("data", ""), "user_id": active.user_id,
                      "agent_id": active.agent_id,
                      "metadata": {"source_message_ids": r.payload.get("source_message_ids", [])}} for r in kept]
            allowed = {str(v["id"]) for v in active.existing_filter(views)}
            kept = [r for r in kept if str(r.id) in allowed]
        return kept

    def _search_existing(self, active: _ActiveScope, original: Callable[..., Any],
                         args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
        """扩展后的已有记忆搜索：逐步扩大 top_k 直到获得足够的有效记录。

        Mem0 默认只请求 10 条已有记忆，逻辑删除或代数不匹配的记录可能占满前缀，
        因此需要逐步扩大候选范围（倍增 top_k），同时设置硬上限
        _EXISTING_SEARCH_MAX_TOP_K 防止无限拉取。
        """
        requested = kwargs.get("top_k")
        if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
            raise MemoryScopeError("Mem0 existing-memory search must pass a positive keyword top_k")
        fetch_top_k = requested
        fetch_cap = max(requested, _EXISTING_SEARCH_MAX_TOP_K)
        previous_ids: tuple[str, ...] | None = None
        while True:
            fetch_kwargs = dict(kwargs)
            fetch_kwargs["top_k"] = fetch_top_k
            rows = list(original(*args, **fetch_kwargs) or [])
            kept = self._filter_existing(rows, active)
            active.fetch_attempts += 1
            row_ids = tuple(str(getattr(row, "id", "")) for row in rows)
            exhausted = len(rows) < fetch_top_k or row_ids == previous_ids
            if len(kept) >= requested or exhausted or fetch_top_k >= fetch_cap:
                returned = kept[:requested]
                active.requested_top_k = requested
                active.fetched_top_k = fetch_top_k
                active.filtered_count = len(rows) - len(kept)
                active.screened_out = active.filtered_count
                active.returned_count = len(returned)
                active.existing = [
                    {"id": str(r.id), "memory": r.payload.get("data", ""),
                     "user_id": r.payload.get("user_id"), "run_id": r.payload.get("run_id")}
                    for r in returned
                ]
                return returned
            previous_ids = row_ids
            fetch_top_k = min(fetch_cap, fetch_top_k * 2)

    def _screen_entity_rows(self, active: _ActiveScope, original: Callable[..., Any],
                            args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
        """过滤实体行，只保留属于当前说话人作用域的记录。"""
        return [r for r in _list_rows(original(*args, **kwargs)) if self._owned(r, active)]

    def _screen_entity_batches(self, active: _ActiveScope, original: Callable[..., Any],
                               args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[list[Any]]:
        """过滤实体批次，每批只保留属于当前说话人作用域的记录。"""
        batches = original(*args, **kwargs)
        return [[r for r in (batch or []) if self._owned(r, active)] for batch in (batches or [])]

    @contextmanager
    def serialized_operation(self, operation: str) -> Iterator[None]:
        """串行化操作上下文管理器：用非阻塞锁保证同一时刻只有一个写操作执行。

        副作用:
            获取 self._activation_guard 锁；并发或重入调用会抛 MemoryScopeError。
        """
        if not self._activation_guard.acquire(blocking=False):
            raise MemoryScopeError(
                f"Mem0LongTermMemory requires serialized access; concurrent or reentrant {operation} is unsupported"
            )
        try:
            yield
        finally:
            self._activation_guard.release()

    @contextmanager
    def session(self, *, user_id: str, agent_id: str, run_id: str, generation: int,
                existing_filter: ExistingMemoryFilter | None) -> Iterator[_ActiveScope]:
        """进入长期记忆会话上下文：激活作用域并包装实体存储，退出时清理。

        副作用:
            获取串行化锁；设置 self.active；对 entity_store 进行猴子补丁。
        """
        with self.serialized_operation("add_session"):
            active = _ActiveScope(user_id, agent_id, run_id, generation, existing_filter)
            try:
                if self.active is not None:
                    raise MemoryScopeError(
                        "Mem0LongTermMemory requires serialized access; long-term scope is already active"
                    )
                self.active = active
                # Mem0 惰性创建实体存储；在 add() 访问前先包装它
                self._wrap(self.memory.entity_store, {
                    "list": ("entity_list", self._screen_entity_rows),
                    "search_batch": ("entity_search_batch", self._screen_entity_batches),
                })
                yield active
            finally:
                if self.active is active:
                    self.active = None


class Mem0LongTermMemory:
    """Mem0 长期记忆适配器：基于 Mem0 SDK 实现搜索、添加、删除等 L3 操作。

    核心职责:
    - 将 Mem0 SDK 的 search/add/delete 封装为 LongTermMemory 协议
    - 通过 _LongTermScope 管理 run/session 作用域分离
    - 对写入结果进行回读验证，区分可重试错误和写入不确定错误
    - 实体加成策略可按环境变量关闭以减少延迟

    副作用:
        构造时即创建 Mem0 Memory 实例（含向量库连接和嵌入模型加载）。
    """
    def __init__(self, llm: ProviderConfig, config: Mem0AdapterConfig) -> None:
        """初始化 Mem0 后端。

        参数:
            llm:    LLM 提供者配置（用于 Mem0 的记忆抽取和推理）
            config: Mem0 适配器配置（向量库路径、嵌入模型等）

        副作用:
            - 调用 Memory.from_config() 创建 Mem0 实例（加载嵌入模型和向量库）
            - 应用实体加成策略
            - 绑定 thinking_mode 到 LLM 请求选项
        """
        from bumi_mem.providers import ensure_explicit_mem0_route
        ensure_explicit_mem0_route()
        from mem0 import Memory

        self.provider_name = llm.provider
        self.llm_config = llm
        self.config = config
        embed_device = resolve_embed_device(config.embed_device)
        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory = Memory.from_config(
            {
                "version": "v1.1",
                "history_db_path": str(config.data_dir / "history.db"),
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": llm.model,
                        "api_key": llm.api_key,
                        "openai_base_url": llm.base_url.rstrip("/"),
                        "temperature": 0.0,
                        "max_tokens": config.memory_max_tokens,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": str(config.embed_model),
                        "embedding_dims": config.embedding_dims,
                        "model_kwargs": {
                            "device": embed_device,
                            "local_files_only": True,
                        },
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": _vector_store_config(config),
                },
            }
        )
        self.entity_boost = _apply_entity_boost_policy(self.memory)
        # 测试替身没有向量存储，保持 SDK 自身的作用域
        self._scope = _LongTermScope(self.memory) if hasattr(self.memory, "vector_store") else None
        self.last_scope_diagnostics: dict[str, Any] | None = None

        if llm.thinking_mode is not None:
            from bumi_mem.providers import bind_provider_request_options
            self.memory.llm.generate_response = bind_provider_request_options(self.memory.llm.generate_response, llm)

    def warmup(self, *, query: str) -> None:
        """预热后端：执行一次搜索以验证嵌入模型和向量库可用。"""
        warmup_mem0_backend(self.memory, query=query)

    @staticmethod
    def _filters(user_id: str, agent_id: str) -> dict[str, str]:
        """构建 user_id + agent_id 过滤器，同时验证身份对有效性。"""
        require_pair(user_id, agent_id)
        return {"user_id": user_id, "agent_id": agent_id}

    def search(
        self, *, user_id: str, agent_id: str, query: str, top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
        """搜索长期记忆。

        参数:
            user_id:   用户 ID
            agent_id:  Agent ID
            query:     查询文本
            top_k:     返回最大条数
            threshold: 相似度阈值

        返回:
            匹配的记忆列表，每条包含 id、memory、score 等字段。

        副作用:
            调用 Mem0 search，触发嵌入编码和向量检索。
        """
        rows = unwrap_results(
            self.memory.search(
                query=query,
                top_k=top_k,
                filters=self._filters(user_id, agent_id),
                threshold=threshold,
                rerank=False,
            )
        )
        self._verify_scope(rows, user_id, agent_id)
        return rows

    @staticmethod
    def _verify_scope(rows, user_id, agent_id):
        """验证返回行都属于请求的 user_id/agent_id，防止作用域泄漏。"""
        require_pair(user_id, agent_id)
        if any(row.get('user_id') != user_id or row.get('agent_id') != agent_id for row in rows):
            raise MemoryScopeError('backend result does not match the requested user/agent')

    @staticmethod
    def _check_metadata(metadata):
        """检查 metadata 不包含保留字段，防止覆写记忆所有权或身份。"""
        if {'user_id', 'agent_id', 'id', 'memory_id', 'run_id'} & set(metadata):
            raise MemoryScopeError('metadata cannot override memory ownership or identity')

    def add_session(
        self, *, user_id: str, agent_id: str, messages: Sequence[dict[str, str]],
        metadata: dict[str, Any], prompt: str,
        replay_authorization: dict[str, Any] | None = None,
        existing_memory_filter: ExistingMemoryFilter | None = None,
    ) -> list[dict[str, Any]]:
        """将会话消息添加到长期记忆，由 Mem0 LLM 自动抽取和去重。

        参数:
            user_id:               用户 ID
            agent_id:              Agent ID
            messages:              消息列表，每条含 role 和 content
            metadata:              元数据（含 memory_generation、consolidation_id 等）
            prompt:                Mem0 抽取提示词
            existing_memory_filter: 已有记忆过滤器（排除已撤回/被 source block 的记录）

        返回:
            Mem0 事件列表（ADD/UPDATE/DELETE 事件）。

        副作用:
            - 调用 Mem0 add()，触发 LLM 抽取、嵌入编码和向量写入
            - 对写入结果进行回读验证
            - 失败时抛 RetryableMemoryError（可重试）或 MemoryWriteUncertainError（不可盲目重放）

        关键逻辑:
            run_id 隔离每个证据块在 SDK 的 last-k 缓存中，防止撤回的来源在新块下复活。
            长期记忆跨 run 读取（通过 _LongTermScope 作用域扩展）。
        """
        # 失败调用不得暴露之前成功调用的诊断信息
        self.last_scope_diagnostics = None
        require_pair(user_id, agent_id)
        self._check_metadata(metadata)
        # 在当前锁定的 SDK 版本中，LLMError 在向量持久化之前抛出，
        # 其他异常可能发生在副作用之后，不能盲目重放。
        from mem0.exceptions import LLMError

        generation = metadata.get("memory_generation", 0)
        if (not isinstance(generation, int) or isinstance(generation, bool) or generation < 0):
            raise ValueError("memory_generation must be a non-negative integer")
        # run_id 隔离每个证据块和删除代数在 SDK 的 last-k 缓存中，
        # 防止撤回的来源在新块下复活。长期记忆跨 run 读取。
        run_id = f"bumi-generation-{generation}-chunk-{metadata.get('consolidation_id') or uuid4().hex}"
        scope = (self._scope.session(user_id=user_id, agent_id=agent_id, run_id=run_id, generation=generation,
                                     existing_filter=existing_memory_filter)
                 if self._scope is not None else nullcontext())
        active = None
        fact_sources: dict[str, list[str]] = {}
        try:
            with scope as active:
                boundary = (extraction_provenance(self.memory, user_id=user_id, agent_id=agent_id,
                            run_id=run_id, metadata=metadata) if self._scope is not None else nullcontext({}))
                with boundary as fact_sources:
                    result = self.memory.add(
                        [{"role": m["role"], "content": m["content"]} for m in messages], user_id=user_id,
                        agent_id=agent_id, run_id=run_id, metadata=dict(metadata), prompt=prompt, infer=True,
                    )
            if active is not None and active.hook_hits["vector_search"] < 1:
                # add() 可能已经产生副作用，因此这是写入不确定而非可重试的兼容性错误
                raise MemoryWriteUncertainError(
                    "Mem0 add() skipped the existing-memory vector search hook; "
                    "the pinned SDK adapter is incompatible and the write outcome is uncertain"
                )
            if not ((isinstance(result, dict) and isinstance(result.get("results"), list))
                    or isinstance(result, list)):
                raise MemoryWriteUncertainError("Unexpected Mem0 add response shape")
            events = unwrap_results(result)
            # v2.0.20 可能吞没单条向量插入错误。在声明分块完成前验证返回的 ADD 记录
            # （这不是精确一次保证，而是尽力验证）。
            try:
                for event in events:
                    if event.get("event") == "ADD":
                        saved = self.memory.get(event["id"])
                        if (not saved or saved.get("memory") != event.get("memory")
                            or saved.get("user_id") != user_id or saved.get("agent_id") != agent_id
                            or saved.get("metadata", {}).get("consolidation_id") != metadata.get("consolidation_id")):
                            raise RuntimeError("Mem0 ADD read-back mismatch")
                        expected_sources = fact_sources.get(event.get('memory'), [])
                        if expected_sources and saved.get('metadata', {}).get('fact_source_message_ids') != expected_sources:
                            raise RuntimeError('Mem0 fact provenance read-back mismatch')
            except Exception as exc:
                raise MemoryWriteUncertainError(f"Memory write verification failed: {exc}") from exc
        except LLMError as exc:
            raise RetryableMemoryError(str(exc)) from exc
        except MemoryWriteUncertainError:
            raise
        except Exception as exc:
            raise MemoryWriteUncertainError(str(exc)) from exc
        if active is not None:
            self.last_scope_diagnostics = active.diagnostics()
        return events

    def get_all(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        """获取指定用户/agent 的全部长期记忆。

        参数:
            user_id:  用户 ID
            agent_id: Agent ID

        返回:
            记忆列表。最多拉取 500 条（Mem0 get_all 的 top_k 上限）。

        副作用:
            调用 Mem0 get_all，触发向量库扫描。
        """
        rows = unwrap_results(self.memory.get_all(filters=self._filters(user_id, agent_id), top_k=500))
        self._verify_scope(rows, user_id, agent_id)
        return rows

    def get_memory(self, *, user_id: str, agent_id: str, memory_id: str) -> dict[str, Any] | None:
        """按 ID 获取单条长期记忆，验证归属后返回。

        参数:
            user_id:   用户 ID
            agent_id:  Agent ID
            memory_id: 记忆 ID

        返回:
            记忆字典，不存在或不属于当前作用域时返回 None。
        """
        require_pair(user_id, agent_id)
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError('memory_id is required')
        row = self.memory.get(memory_id)
        if not row or row.get('id') != memory_id or row.get('user_id') != user_id or row.get('agent_id') != agent_id:
            return None
        return row

    def remember(self, *, user_id: str, agent_id: str, contents: Sequence[str], metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """显式写入长期记忆：不经过 LLM 抽取，直接将内容写入后端。

        参数:
            user_id:  用户 ID
            agent_id: Agent ID
            contents: 要写入的记忆文本列表
            metadata: 元数据

        返回:
            已保存的记忆列表（含 id、memory 等字段）。

        副作用:
            - 调用 Mem0 add(infer=False)，触发嵌入编码和向量写入
            - 对每条写入结果进行回读验证
            - 失败时抛 MemoryWriteUncertainError，不可自动重放

        注意:
            验证失败可能发生在部分写入之后，因此不可盲目重放。
        """
        require_pair(user_id, agent_id)
        self._check_metadata(metadata)
        if isinstance(contents, (str, bytes)) or not contents or any(not isinstance(x, str) or not x.strip() for x in contents):
            raise ValueError('contents must contain nonempty memory strings')
        texts = [x.strip() for x in contents]
        try:
            result = self.memory.add([{'role':'user','content':x} for x in texts], user_id=user_id, agent_id=agent_id,
                                     metadata=dict(metadata), infer=False)
            events = unwrap_results(result)
            if len(events) != len(texts):
                raise RuntimeError('not every explicit memory has a write result')
            if len({event.get('id') for event in events}) != len(texts):
                raise RuntimeError('duplicate memory IDs in write result')
            saved = []
            for event in events:
                row = self.get_memory(user_id=user_id, agent_id=agent_id, memory_id=event['id'])
                if not row or row.get('metadata', {}).get('write_batch_id') != metadata.get('write_batch_id'):
                    raise RuntimeError('explicit memory write verification failed')
                saved.append(row)
            if Counter(x['memory'] for x in saved) != Counter(texts):
                raise RuntimeError('saved content differs from requested content')
            return saved
        except Exception as exc:
            raise MemoryWriteUncertainError('explicit write outcome is uncertain; do not blindly replay') from exc

    def delete_all(self, *, user_id: str, agent_id: str) -> None:
        """删除指定用户/agent 的全部长期记忆和实体。

        参数:
            user_id:  用户 ID
            agent_id: Agent ID

        副作用:
            - 调用 Mem0 delete_all，触发向量库和实体库的物理删除
            - 验证删除后无残留记忆和实体
            - 失败时不释放持久读抑制（delete_pending 状态保持）

        关键逻辑:
            - 使用串行化锁防止并发删除
            - 对 Mem0 内部的实体清理进行作用域校验
            - 删除后额外清理跨 run 的实体链接（Mem0 的按 memory 清理看不到跨 run 实体）
            - 实体清理有 _ENTITY_PURGE_LIMIT 行数上限，超出则失败
        """
        require_pair(user_id, agent_id)
        guard = (self._scope.serialized_operation("delete_all")
                 if self._scope is not None else nullcontext())
        with guard:
            self._delete_all_serialized(user_id=user_id, agent_id=agent_id)

    def _delete_all_serialized(self, *, user_id: str, agent_id: str) -> None:
        """串行化删除实现：在已持有锁的情况下执行删除和验证。"""
        entities = self.memory.entity_store if self._scope is not None else None
        if entities is None:
            self.memory.delete_all(user_id=user_id, agent_id=agent_id)
        else:
            # Mem0 先按 memory/run 清理实体，再执行本适配器的说话人范围清除。
            # 需要对内部 list 结果也做作用域守卫：Mem0 信任提供者过滤器，
            # 可能在显式清除验证归属之前就更新/删除了外来的行。
            original_list = entities.list
            cleanup_scope_error: MemoryScopeError | None = None

            def checked_list(*args, **kwargs):
                nonlocal cleanup_scope_error
                listed = original_list(*args, **kwargs)
                if not all(_row_owned_by(row, user_id, agent_id) for row in _list_rows(listed)):
                    cleanup_scope_error = MemoryScopeError(
                        "entity cleanup crossed the requested speaker scope"
                    )
                    raise cleanup_scope_error
                return listed

            entities.list = checked_list
            try:
                try:
                    self.memory.delete_all(user_id=user_id, agent_id=agent_id)
                except Exception:
                    if cleanup_scope_error is not None:
                        raise cleanup_scope_error
                    raise
            finally:
                entities.list = original_list
            # Mem0 2.0.20 吞没实体清理异常。在其删除循环返回后，抛出记录的作用域违规。
            if cleanup_scope_error is not None:
                raise cleanup_scope_error
        # 失败/部分删除不得释放持久读抑制
        remaining = self.memory.get_all(filters=self._filters(user_id, agent_id), top_k=1, show_expired=True)
        if not isinstance(remaining, (dict, list)) or (isinstance(remaining, dict) and not isinstance(remaining.get("results"), list)):
            raise RuntimeError("Unexpected Mem0 delete verification response")
        if unwrap_results(remaining):
            raise RuntimeError("Mem0 delete verification found remaining memories")
        if self._scope is not None:
            # 跨 run 的实体链接对 Mem0 的按 memory 清理不可见
            # （它按 memory 的 run_id 列出实体）：需要清除说话人作用域。
            # 这是一次有界批量操作。超出上限或残留任何记录都会导致删除失败，
            # 因此持久 delete_pending 抑制保持活跃。
            entity_filters = self._filters(user_id, agent_id)
            rows = _list_rows(entities.list(filters=entity_filters, top_k=_ENTITY_PURGE_LIMIT + 1))
            if len(rows) > _ENTITY_PURGE_LIMIT:
                raise RuntimeError(
                    f"Mem0 entity purge exceeds the bounded {_ENTITY_PURGE_LIMIT}-row limit"
                )
            for row in rows:
                if not _row_owned_by(row, user_id, agent_id):
                    raise MemoryScopeError("entity purge crossed the requested speaker scope")
            for row in rows:
                entities.delete(vector_id=row.id)
            leftovers = _list_rows(entities.list(filters=entity_filters, top_k=1))
            if leftovers:
                if not _row_owned_by(leftovers[0], user_id, agent_id):
                    raise MemoryScopeError("entity purge verification crossed the requested speaker scope")
                raise RuntimeError(
                    f"Mem0 delete verification found entity records beyond the {_ENTITY_PURGE_LIMIT}-row purge limit"
                )

    def close(self) -> None:
        """关闭 Mem0 后端：释放向量库客户端连接。

        在 SDK 执行线程上调用，此时所有 SDK 调用已完成。
        """
        try:
            self.memory.close()
        finally:
            client = getattr(getattr(self.memory, "vector_store", None), "client", None)
            if client is not None:
                client.close()


class FakeLongTermMemory:
    """内存中的测试替身：实现 LongTermMemory 协议，所有调用可观测。"""

    def __init__(self) -> None:
        """初始化空记忆存储和调用记录。"""
        self.warmup_calls: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self.memories: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.search_results: list[dict[str, Any]] = []
        self.fail_add_count = 0

    def warmup(self, *, query: str) -> None:
        """记录预热调用（测试替身无实际操作）。"""
        self.warmup_calls.append(query)

    @staticmethod
    def _rows_for_scope(rows, user_id, agent_id):
        """为行补全 user_id/agent_id（测试替身不持久化这些字段）。"""
        result = []
        for item in rows:
            row = dict(item)
            row.setdefault('user_id', user_id)
            row.setdefault('agent_id', agent_id)
            result.append(row)
        return result

    def search(
        self, *, user_id: str, agent_id: str, query: str, top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
        """搜索记忆（测试替身：记录调用并返回预设结果或存储的记忆）。"""
        self.search_calls.append(
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "query": query,
                "top_k": top_k,
                "threshold": threshold,
            }
        )
        if self.search_results:
            return self._rows_for_scope(self.search_results[:top_k], user_id, agent_id)
        return self._rows_for_scope(self.memories.get((user_id, agent_id), [])[:top_k], user_id, agent_id)

    def add_session(
        self, *, user_id: str, agent_id: str, messages: Sequence[dict[str, str]],
        metadata: dict[str, Any], prompt: str,
        replay_authorization: dict[str, Any] | None = None,
        existing_memory_filter: ExistingMemoryFilter | None = None,
    ) -> list[dict[str, Any]]:
        """添加会话记忆（测试替身：记录调用，可配置失败次数）。"""
        call = {
            "user_id": user_id,
            "agent_id": agent_id,
            "messages": [dict(item) for item in messages],
            "metadata": dict(metadata),
            "prompt": prompt,
            "replay_authorization": replay_authorization,
            "existing_memory_filter": existing_memory_filter,
        }
        self.add_calls.append(call)
        if self.fail_add_count > 0:
            self.fail_add_count -= 1
            raise RetryableMemoryError("fake Mem0 failure")
        # 测试替身模拟抽取器的文本视图，而非来源头
        user_texts = [item["content"].split("\n", 1)[-1] if item["content"].startswith("[source ") else item["content"]
                      for item in messages if item.get("role") == "user"]
        text = " | ".join(user_texts) or "session memory"
        event = {"id": f"fake-{len(self.add_calls)}", "event": "ADD", "memory": text, "metadata": dict(metadata)}
        self.memories.setdefault((user_id, agent_id), []).append(event)
        return [event]

    def get_all(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        """获取全部记忆（测试替身：返回内存存储）。"""
        return self._rows_for_scope(self.memories.get((user_id, agent_id), []), user_id, agent_id)

    def delete_all(self, *, user_id: str, agent_id: str) -> None:
        """删除全部记忆（测试替身：清空内存存储）。"""
        self.memories.pop((user_id, agent_id), None)


    def remember(self, *, user_id: str, agent_id: str, contents: Sequence[str], metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """显式写入记忆（测试替身：直接存入内存）。"""
        from uuid import uuid4
        require_pair(user_id, agent_id)
        rows = [{'id': str(uuid4()), 'memory': x.strip(), 'user_id': user_id, 'agent_id': agent_id, 'metadata': dict(metadata)} for x in contents]
        self.memories.setdefault((user_id, agent_id), []).extend(rows)
        return [dict(x) for x in rows]

    def get_memory(self, *, user_id: str, agent_id: str, memory_id: str) -> dict[str, Any] | None:
        """按 ID 获取单条记忆（测试替身：从内存存储查找）。"""
        require_pair(user_id, agent_id)
        return next((x for x in self._rows_for_scope(self.memories.get((user_id, agent_id), []), user_id, agent_id) if x['id'] == memory_id), None)
