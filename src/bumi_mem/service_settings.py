# 本次改动（2026-10-21）：Graphiti HTTP 服务默认开启注册隔离，解析环境开关并传入引擎和图后端。
"""环境变量驱动的服务配置：不硬编码服务器文件系统路径。"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from bumi_mem.identity import require_id


def read_env_switch(name: str, default: str = 'on') -> bool:
    """统一解析服务与 CLI 共用的环境开关，拒绝拼写错误。"""
    value = os.getenv(name, default).strip().lower()
    if value not in ('on', 'off', 'true', 'false'):
        raise ValueError(f"{name} must be 'on'/'off' or 'true'/'false'")
    return value in ('on', 'true')


@dataclass(frozen=True)
class ServiceSettings:
    """服务配置：所有参数从环境变量读取，不可变。

    核心属性:
        data_dir:           数据目录
        embed_model_path:   嵌入模型路径
        api_key:            服务 API 密钥
        memory_api_key:     记忆 LLM API 密钥
        agent_id:           默认 Agent ID
        memory_provider:    记忆提供者（deepseek/local/openai-compatible）
        memory_model:       记忆 LLM 模型名
        memory_base_url:    记忆 LLM API 基础 URL
        embed_device:       嵌入设备（auto/cpu/cuda）
    """
    data_dir: Path
    embed_model_path: Path
    api_key: str = field(repr=False)
    memory_api_key: str = field(repr=False)
    neo4j_uri: str = 'bolt://127.0.0.1:7687'
    neo4j_user: str = 'neo4j'
    neo4j_password: str = field(default='', repr=False)
    reranker_model_path: Path = Path('.')  # Deprecated; no longer loaded.
    extraction_turns: int = 10
    extraction_tokens: int = 8000
    community_every: int = 5
    graph_max_coroutines: int = 3
    agent_id: str = 'bumi_test_agent'
    memory_provider: str = 'deepseek'
    memory_model: str = 'deepseek-flash'
    memory_base_url: str = 'https://api.deepseek.com/v1'
    embed_device: str = 'auto'
    collection_name: str = 'bumi_mem_l3'
    read_timeout: float = 30.0
    explicit_recall_timeout: float = 30.0
    warmup_timeout: float = 120.0
    memory_warmup_enabled: bool = True
    long_term_degrade_enabled: bool = True
    long_term_degrade_cooloff_seconds: float = 60.0
    long_term_degrade_probe_timeout: float = 5.0
    max_messages: int = 20
    max_l1_tokens: int = 8000
    carryover_messages: int = 20
    top_k: int = 5
    search_threshold: float = 0.1
    summary_max_tokens: int = 512
    memory_max_tokens: int = 4096
    local_chunk_tokens: int = 3500
    max_attempts: int = 3
    retry_base_seconds: float = 5.0
    prefetch_ttl_seconds: float = 3.0
    prefetch_max_growth_chars: int = 12
    reranker_fp16: bool = True
    registered_isolation: bool = True

    def __post_init__(self):
        if not math.isfinite(self.explicit_recall_timeout) or self.explicit_recall_timeout <= 0:
            raise ValueError('explicit recall timeout must be positive and finite')
        object.__setattr__(self, 'data_dir', Path(self.data_dir).expanduser().resolve())
        object.__setattr__(self, 'embed_model_path', Path(self.embed_model_path).expanduser().resolve())
        require_id(self.agent_id, 'agent_id')
        if len(self.api_key) < 32 or self.api_key != self.api_key.strip():
            raise ValueError('BUMI_SERVICE_API_KEY must contain at least 32 characters')
        if not self.memory_api_key or not self.memory_model.strip():
            raise ValueError('memory API key and model must be configured')
        if self.memory_provider not in ('deepseek', 'local', 'openai-compatible'):
            raise ValueError('unsupported memory provider')
        url = urlparse(self.memory_base_url)
        if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password:
            raise ValueError('memory base URL must be http(s), without embedded credentials')
        if self.embed_device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('invalid embedding device')
        for name in ('read_timeout', 'warmup_timeout', 'long_term_degrade_cooloff_seconds',
                     'long_term_degrade_probe_timeout'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('max_messages', 'max_l1_tokens', 'top_k', 'summary_max_tokens',
                     'memory_max_tokens', 'local_chunk_tokens', 'max_attempts'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if min(self.extraction_turns, self.extraction_tokens, self.community_every, self.graph_max_coroutines) <= 0:
            raise ValueError('Graphiti batch/community limits must be positive')
        if self.carryover_messages < 0 or not 0 <= self.search_threshold <= 1:
            raise ValueError('invalid carryover or search threshold')
        if not math.isfinite(self.retry_base_seconds) or self.retry_base_seconds < 0:
            raise ValueError('retry delay must be finite and nonnegative')

    @property
    def db_path(self):
        return self.data_dir / 'bumi_memory.sqlite3'

    @classmethod
    def from_env(cls):
        def required(name):
            value = os.environ.get(name, '')
            if not value:
                raise ValueError(f'{name} must be configured')
            return value

        return cls(
            data_dir=Path(required('BUMI_DATA_DIR')),
            embed_model_path=Path(required('BUMI_EMBED_MODEL_PATH')),
            api_key=required('BUMI_SERVICE_API_KEY'),
            memory_api_key=required('BUMI_MEMORY_API_KEY'),
            neo4j_uri=os.getenv('BUMI_NEO4J_URI', 'bolt://127.0.0.1:7687'),
            neo4j_user=os.getenv('BUMI_NEO4J_USER', 'neo4j'),
            neo4j_password=required('BUMI_NEO4J_PASSWORD'),
            reranker_model_path=Path(os.getenv('BUMI_RERANKER_MODEL_PATH', '.')),
            reranker_fp16=read_env_switch('BUMI_RERANKER_FP16'),
            registered_isolation=read_env_switch('BUMI_REGISTERED_ISOLATION'),
            extraction_turns=int(os.getenv('BUMI_EXTRACTION_TURNS', '10')),
            extraction_tokens=int(os.getenv('BUMI_EXTRACTION_TOKENS', '8000')),
            community_every=int(os.getenv('BUMI_COMMUNITY_EVERY', '5')),
            graph_max_coroutines=int(os.getenv('BUMI_GRAPH_MAX_COROUTINES', '3')),
            agent_id=os.getenv('BUMI_AGENT_ID', 'bumi_test_agent'),
            memory_provider=os.getenv('BUMI_MEMORY_PROVIDER', 'deepseek'),
            memory_model=os.getenv('BUMI_MEMORY_MODEL', 'deepseek-flash'),
            memory_base_url=os.getenv('BUMI_MEMORY_BASE_URL', 'https://api.deepseek.com/v1'),
            embed_device=os.getenv('BUMI_EMBED_DEVICE', 'auto'),
            collection_name=os.getenv('BUMI_COLLECTION_NAME', 'bumi_mem_l3'),
            read_timeout=float(os.getenv('BUMI_MEMORY_READ_TIMEOUT', '30')),
            explicit_recall_timeout=float(os.getenv('BUMI_EXPLICIT_RECALL_TIMEOUT', '30')),
            warmup_timeout=float(os.getenv('BUMI_MEMORY_WARMUP_TIMEOUT', '120')),
            memory_warmup_enabled=read_env_switch('BUMI_MEMORY_WARMUP'),
            long_term_degrade_enabled=read_env_switch('BUMI_LONG_TERM_DEGRADE_ENABLED'),
            long_term_degrade_cooloff_seconds=float(os.getenv('BUMI_LONG_TERM_DEGRADE_COOLOFF_SECONDS', '60')),
            long_term_degrade_probe_timeout=float(
                os.getenv('BUMI_LONG_TERM_DEGRADE_PROBE_TIMEOUT_SECONDS', '5')),
            max_messages=int(os.getenv('BUMI_MAX_MESSAGES', '20')),
            max_l1_tokens=int(os.getenv('BUMI_MAX_L1_TOKENS', '8000')),
            carryover_messages=int(os.getenv('BUMI_CARRYOVER_MESSAGES', '20')),
            top_k=int(os.getenv('BUMI_TOP_K', '5')),
            search_threshold=float(os.getenv('BUMI_SEARCH_THRESHOLD', '0.1')),
            summary_max_tokens=int(os.getenv('BUMI_SUMMARY_MAX_TOKENS', '512')),
            memory_max_tokens=int(os.getenv('BUMI_MEMORY_MAX_TOKENS', '4096')),
            local_chunk_tokens=int(os.getenv('BUMI_LOCAL_CHUNK_TOKENS', '3500')),
            max_attempts=int(os.getenv('BUMI_MEMORY_MAX_ATTEMPTS', '3')),
            retry_base_seconds=float(os.getenv('BUMI_MEMORY_RETRY_BASE_SECONDS', '5')),
            prefetch_ttl_seconds=float(os.getenv('BUMI_PREFETCH_TTL_SECONDS', '3')),
            prefetch_max_growth_chars=int(os.getenv('BUMI_PREFETCH_MAX_GROWTH_CHARS', '12')),
        )


def memory_provider_config(settings: ServiceSettings):
    from bumi_mem.providers import ProviderConfig
    return ProviderConfig(settings.memory_provider, settings.memory_model,
                          settings.memory_base_url, settings.memory_api_key,
                          settings.summary_max_tokens, 0.0,
                          thinking_mode='disabled' if settings.memory_provider == 'deepseek' else None)


def build_service_engine(settings: ServiceSettings):
    """Graphiti-only runtime. Never imports or constructs Mem0."""
    from bumi_mem.consolidator import ConsolidatorConfig, SessionConsolidator
    from bumi_mem.engine import BumiMemory, BumiMemoryConfig
    from bumi_mem.async_memory_access import AsyncMemoryAccess
    from bumi_mem.graphiti_memory import GraphitiConfig, GraphitiLongTermMemory
    from bumi_mem.providers import OpenAICompatibleProvider
    from bumi_mem.session_store import SQLiteSessionStore
    from bumi_mem.token_counter import ApproxTokenCounter

    for path in (settings.embed_model_path,):
        if not Path(path).is_dir():
            raise ValueError(f'Expected existing local model directory: {path}')
    if not settings.neo4j_password:
        raise ValueError('BUMI_NEO4J_PASSWORD must be configured')
    os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')
    provider = memory_provider_config(settings)
    counter = ApproxTokenCounter()
    store = SQLiteSessionStore(settings.db_path)
    with store._connect() as conn:
        if conn.execute("SELECT 1 FROM memory_jobs WHERE COALESCE(json_extract(payload_json,'$.settings.backend'),'legacy') != 'graphiti' AND kind='consolidate' LIMIT 1").fetchone():
            raise ValueError('This is a legacy data directory; select a NEW BUMI_DATA_DIR for the Graphiti experiment')
    graph_config = GraphitiConfig(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password,
        settings.data_dir / 'graphiti', settings.embed_model_path, settings.reranker_model_path,
        device=settings.embed_device, community_every=settings.community_every,
        max_coroutines=settings.graph_max_coroutines, reranker_fp16=settings.reranker_fp16, registered_isolation=settings.registered_isolation)
    access = AsyncMemoryAccess(lambda: GraphitiLongTermMemory(provider, graph_config), read_timeout=settings.read_timeout,
        degrade_cooloff_seconds=settings.long_term_degrade_cooloff_seconds,
        degrade_probe_timeout=settings.long_term_degrade_probe_timeout)
    consolidator = SessionConsolidator(store=store, long_term=access,
        summary_provider=OpenAICompatibleProvider(provider), memory_provider_name=settings.memory_provider,
        memory_token_counter=counter, config=ConsolidatorConfig(backend='graphiti',
            extraction_turns=settings.extraction_turns, extraction_tokens=settings.extraction_tokens,
            carryover_messages=settings.carryover_messages, l1_max_messages=settings.max_messages,
            local_chunk_tokens=settings.extraction_tokens, max_attempts=settings.max_attempts))
    return BumiMemory(store=store, long_term=access, chat_provider=None, consolidator=consolidator,
        chat_token_counter=counter, auto_start=False, config=BumiMemoryConfig(
            registered_isolation=settings.registered_isolation, semantic_state_enabled=False, max_messages=settings.max_messages,
            max_l1_tokens=settings.max_l1_tokens, memory_read_timeout=settings.read_timeout,
            explicit_recall_timeout=settings.explicit_recall_timeout,
            memory_warmup_timeout=settings.warmup_timeout,
            memory_warmup_enabled=settings.memory_warmup_enabled,
            long_term_degrade_enabled=settings.long_term_degrade_enabled,
            long_term_degrade_cooloff_seconds=settings.long_term_degrade_cooloff_seconds,
            long_term_degrade_probe_timeout=settings.long_term_degrade_probe_timeout,
            prefetch_ttl_seconds=settings.prefetch_ttl_seconds,
            prefetch_max_growth_chars=settings.prefetch_max_growth_chars))
