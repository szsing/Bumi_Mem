"""流式前缀预取：把检索提前到 ASR 稳定前缀上发起，终稿只做拼装。

契约见 docs/latency/REPORT.md 附录 B。实现刻意保持最小：
- 预取**不写 L1、不建轮次、不发布引用**，只在内存里缓存候选，随时可丢弃；
- prepare 携带 prefetch_id 时才尝试复用，条件是同一 agent/user/client_turn_id、
  终稿（归一化后）以前缀开头、增量不超过阈值、未过期、generation 未变；
- 任一条件不满足就静默重算，并在统计里记为 stale/expired/miss。
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


def normalize_prefix(text: str) -> str:
    """折叠空白后比较，避免 ASR 分片带来的空格差异。"""
    return ''.join(str(text).split())


@dataclass
class PrefetchEntry:
    prefetch_id: str
    agent_id: str
    user_id: str
    client_turn_id: str
    query_prefix: str
    generation: int
    recall_status: str
    rows: list[dict]
    created_at: float
    expires_at: float

    def age_ms(self, now: float) -> float:
        return (now - self.created_at) * 1000


@dataclass
class PrefetchStats:
    created: int = 0
    hit: int = 0
    stale: int = 0
    expired: int = 0
    miss: int = 0
    evicted: int = 0
    rejected_guest: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        reuse = self.hit + self.stale + self.expired + self.miss
        return {
            'created': self.created,
            'hit': self.hit,
            'stale': self.stale,
            'expired': self.expired,
            'miss': self.miss,
            'evicted': self.evicted,
            'rejected_guest': self.rejected_guest,
            'reuse_attempts': reuse,
            'hit_rate': (self.hit / reuse) if reuse else None,
            'errors': dict(self.errors),
        }


class RetrievalPrefetchCache:
    """有界、带 TTL 的预取缓存；take() 是消费式的，同一轮不会被复用两次。"""

    def __init__(self, *, ttl_seconds: float = 3.0, max_entries: int = 256,
                 max_growth_chars: int = 12, clock=time.monotonic) -> None:
        if ttl_seconds <= 0 or max_entries <= 0 or max_growth_chars < 0:
            raise ValueError('prefetch limits must be positive')
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.max_growth_chars = max_growth_chars
        self._clock = clock
        self._entries: dict[str, PrefetchEntry] = {}
        self._lock = threading.Lock()
        self.stats = PrefetchStats()

    def put(self, *, agent_id: str, user_id: str, client_turn_id: str, query_prefix: str,
            generation: int, recall_status: str, rows: list[dict]) -> PrefetchEntry:
        now = self._clock()
        entry = PrefetchEntry(
            prefetch_id=f"pf_{uuid.uuid4().hex}", agent_id=agent_id, user_id=user_id,
            client_turn_id=client_turn_id, query_prefix=query_prefix, generation=generation,
            recall_status=recall_status, rows=list(rows or []), created_at=now,
            expires_at=now + self.ttl_seconds)
        with self._lock:
            self._evict_expired_locked(now)
            while len(self._entries) >= self.max_entries:
                oldest = min(self._entries.values(), key=lambda item: item.created_at)
                self._entries.pop(oldest.prefetch_id, None)
                self.stats.evicted += 1
            self._entries[entry.prefetch_id] = entry
            self.stats.created += 1
        return entry

    def take(self, prefetch_id: str, *, agent_id: str, user_id: str, client_turn_id: str,
             query: str, generation: int) -> tuple[str, list[dict] | None, float | None]:
        """返回 (outcome, rows, age_ms)，outcome ∈ hit/stale/expired/miss。"""
        now = self._clock()
        with self._lock:
            entry = self._entries.pop(prefetch_id, None)
            if entry is None:
                self.stats.miss += 1
                return 'miss', None, None
            age_ms = entry.age_ms(now)
            if now > entry.expires_at:
                self.stats.expired += 1
                return 'expired', None, age_ms
            prefix = normalize_prefix(entry.query_prefix)
            final = normalize_prefix(query)
            growth = len(final) - len(prefix)
            if (entry.agent_id != agent_id or entry.user_id != user_id
                    or entry.client_turn_id != client_turn_id or entry.generation != generation
                    or not prefix or not final.startswith(prefix)
                    or growth < 0 or growth > self.max_growth_chars
                    or entry.recall_status != 'ok'):
                self.stats.stale += 1
                return 'stale', None, age_ms
            self.stats.hit += 1
            return 'hit', entry.rows, age_ms

    def invalidate_scope(self, *, user_id: str, agent_id: str) -> int:
        """删除/更正后立即丢弃该作用域的全部预取条目。

        写入或遗忘操作导致检索结果失效时调用，generation 校验只是兜底。
        返回被移除的条目数。
        """
        with self._lock:
            stale = [key for key, item in self._entries.items()
                     if item.user_id == user_id and item.agent_id == agent_id]
            for key in stale:
                self._entries.pop(key, None)
        return len(stale)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def pending(self) -> int:
        with self._lock:
            self._evict_expired_locked(self._clock())
            return len(self._entries)

    def record_error(self, kind: str) -> None:
        with self._lock:
            self.stats.errors[kind] = self.stats.errors.get(kind, 0) + 1

    def _evict_expired_locked(self, now: float) -> None:
        expired = [key for key, item in self._entries.items() if now > item.expires_at]
        for key in expired:
            self._entries.pop(key, None)
