"""Mem0-backed L3 adapter."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence
from uuid import uuid4

from bumi_mem.identity import require_pair
from bumi_mem.providers import ProviderConfig
from bumi_mem.schemas import MemoryWriteUncertainError, RetryableMemoryError, MemoryScopeError


def unwrap_results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        results = value.get("results", [])
        return results if isinstance(results, list) else []
    return value if isinstance(value, list) else []


def resolve_embed_device(requested: str, *, cuda_available: bool | None = None) -> str:
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
                raise RuntimeError("CUDA embedding requested but PyTorch is not installed") from exc
            return "cpu"
        cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA embedding requested but this Python environment has no CUDA-enabled PyTorch"
        )
    return "cuda" if cuda_available else "cpu"


def warmup_mem0_backend(memory: Any, *, query: str) -> None:
    """Exercise the full Mem0 search path without using a real user namespace."""
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
    def warmup(self, *, query: str) -> None: ...

    def search(
        self, *, user_id: str, agent_id: str, query: str, top_k: int, threshold: float
    ) -> list[dict[str, Any]]: ...

    def add_session(
        self,
        *,
        user_id: str,
        agent_id: str,
        messages: Sequence[dict[str, str]],
        metadata: dict[str, Any],
        prompt: str,
    ) -> list[dict[str, Any]]: ...

    def get_all(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]: ...

    def delete_all(self, *, user_id: str, agent_id: str) -> None: ...

    def remember(self, *, user_id: str, agent_id: str, contents: Sequence[str], metadata: dict[str, Any]) -> list[dict[str, Any]]: ...

    def get_memory(self, *, user_id: str, agent_id: str, memory_id: str) -> dict[str, Any] | None: ...


@dataclass(slots=True, frozen=True)
class Mem0AdapterConfig:
    data_dir: Path
    embed_model: Path
    embed_device: str = "auto"
    collection_name: str = "bumi_mem_l3"
    embedding_dims: int = 1024
    memory_max_tokens: int = 4096


class Mem0LongTermMemory:
    def __init__(self, llm: ProviderConfig, config: Mem0AdapterConfig) -> None:
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
                    "config": {
                        "collection_name": config.collection_name,
                        "embedding_model_dims": config.embedding_dims,
                        "path": str(config.data_dir / "qdrant"),
                        "on_disk": True,
                    },
                },
            }
        )

        if llm.thinking_mode is not None:
            from bumi_mem.providers import bind_provider_request_options
            self.memory.llm.generate_response = bind_provider_request_options(self.memory.llm.generate_response, llm)

    def warmup(self, *, query: str) -> None:
        warmup_mem0_backend(self.memory, query=query)

    @staticmethod
    def _filters(user_id: str, agent_id: str) -> dict[str, str]:
        require_pair(user_id, agent_id)
        return {"user_id": user_id, "agent_id": agent_id}

    def search(
        self, *, user_id: str, agent_id: str, query: str, top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
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
        require_pair(user_id, agent_id)
        if any(row.get('user_id') != user_id or row.get('agent_id') != agent_id for row in rows):
            raise MemoryScopeError('backend result does not match the requested user/agent')

    @staticmethod
    def _check_metadata(metadata):
        if {'user_id', 'agent_id', 'id', 'memory_id', 'run_id'} & set(metadata):
            raise MemoryScopeError('metadata cannot override memory ownership or identity')

    def add_session(
        self,
        *,
        user_id: str,
        agent_id: str,
        messages: Sequence[dict[str, str]],
        metadata: dict[str, Any],
        prompt: str,
    ) -> list[dict[str, Any]]:
        require_pair(user_id, agent_id)
        self._check_metadata(metadata)
        # In the pinned SDK, LLMError is raised before vector persistence.
        # Other failures can happen after side effects and must not be replayed blindly.
        from mem0.exceptions import LLMError

        try:
            result = self.memory.add(
                [{"role": m["role"], "content": m["content"]} for m in messages], user_id=user_id, agent_id=agent_id,
                # Isolate each evidence chunk as well as deletion generation.
                # Otherwise a targeted deletion could be resurrected by the SDK's
                # previous-input cache under a NEW chunk's unrelated source IDs.
                run_id=(f"bumi-generation-{int(metadata.get('memory_generation', 0))}"
                        f"-chunk-{metadata.get('consolidation_id') or uuid4().hex}"),
                metadata=dict(metadata), prompt=prompt, infer=True,
            )
        except LLMError as exc:
            raise RetryableMemoryError(str(exc)) from exc
        except Exception as exc:
            raise MemoryWriteUncertainError(str(exc)) from exc
        if not ((isinstance(result, dict) and isinstance(result.get("results"), list))
                or isinstance(result, list)):
            raise MemoryWriteUncertainError("Unexpected Mem0 add response shape")
        events = unwrap_results(result)
        # v2.0.20 can swallow individual vector insert errors. Verify returned
        # ADD records before declaring the chunk complete (not an exactly-once guarantee).
        try:
            for event in events:
                if event.get("event") == "ADD":
                    saved = self.memory.get(event["id"])
                    if (not saved or saved.get("memory") != event.get("memory")
                        or saved.get("user_id") != user_id or saved.get("agent_id") != agent_id
                        or saved.get("metadata", {}).get("consolidation_id") != metadata.get("consolidation_id")):
                        raise RuntimeError("Mem0 ADD read-back mismatch")
        except Exception as exc:
            raise MemoryWriteUncertainError(f"Memory write verification failed: {exc}") from exc
        return events

    def get_all(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        rows = unwrap_results(self.memory.get_all(filters=self._filters(user_id, agent_id), top_k=500))
        self._verify_scope(rows, user_id, agent_id)
        return rows

    def get_memory(self, *, user_id: str, agent_id: str, memory_id: str) -> dict[str, Any] | None:
        require_pair(user_id, agent_id)
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError('memory_id is required')
        row = self.memory.get(memory_id)
        if not row or row.get('id') != memory_id or row.get('user_id') != user_id or row.get('agent_id') != agent_id:
            return None
        return row

    def remember(self, *, user_id: str, agent_id: str, contents: Sequence[str], metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Explicit facts, no LLM extraction. Same existing persistent backend.

        No automatic replay: a failed verification may follow a partial write.
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
        require_pair(user_id, agent_id)
        self.memory.delete_all(user_id=user_id, agent_id=agent_id)
        # A failed/partial delete must not release the durable read suppression.
        remaining = self.memory.get_all(filters=self._filters(user_id, agent_id), top_k=1, show_expired=True)
        if not isinstance(remaining, (dict, list)) or (isinstance(remaining, dict) and not isinstance(remaining.get("results"), list)):
            raise RuntimeError("Unexpected Mem0 delete verification response")
        if unwrap_results(remaining):
            raise RuntimeError("Mem0 delete verification found remaining memories")

    def close(self) -> None:
        # Invoked on the same executor thread, after all SDK calls have finished.
        try:
            self.memory.close()
        finally:
            client = getattr(getattr(self.memory, "vector_store", None), "client", None)
            if client is not None:
                client.close()


class FakeLongTermMemory:
    """In-memory test double with observable calls."""

    def __init__(self) -> None:
        self.warmup_calls: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self.memories: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.search_results: list[dict[str, Any]] = []
        self.fail_add_count = 0

    def warmup(self, *, query: str) -> None:
        self.warmup_calls.append(query)

    @staticmethod
    def _rows_for_scope(rows, user_id, agent_id):
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
        self,
        *,
        user_id: str,
        agent_id: str,
        messages: Sequence[dict[str, str]],
        metadata: dict[str, Any],
        prompt: str,
    ) -> list[dict[str, Any]]:
        call = {
            "user_id": user_id,
            "agent_id": agent_id,
            "messages": [dict(item) for item in messages],
            "metadata": dict(metadata),
            "prompt": prompt,
        }
        self.add_calls.append(call)
        if self.fail_add_count > 0:
            self.fail_add_count -= 1
            raise RetryableMemoryError("fake Mem0 failure")
        # The fake mirrors the extractor's text view, not its provenance header.
        user_texts = [item["content"].split("\n", 1)[-1] if item["content"].startswith("[source ") else item["content"]
                      for item in messages if item.get("role") == "user"]
        text = " | ".join(user_texts) or "session memory"
        event = {"id": f"fake-{len(self.add_calls)}", "event": "ADD", "memory": text, "metadata": dict(metadata)}
        self.memories.setdefault((user_id, agent_id), []).append(event)
        return [event]

    def get_all(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        return self._rows_for_scope(self.memories.get((user_id, agent_id), []), user_id, agent_id)

    def delete_all(self, *, user_id: str, agent_id: str) -> None:
        self.memories.pop((user_id, agent_id), None)


    def remember(self, *, user_id: str, agent_id: str, contents: Sequence[str], metadata: dict[str, Any]) -> list[dict[str, Any]]:
        from uuid import uuid4
        require_pair(user_id, agent_id)
        rows = [{'id': str(uuid4()), 'memory': x.strip(), 'user_id': user_id, 'agent_id': agent_id, 'metadata': dict(metadata)} for x in contents]
        self.memories.setdefault((user_id, agent_id), []).extend(rows)
        return [dict(x) for x in rows]

    def get_memory(self, *, user_id: str, agent_id: str, memory_id: str) -> dict[str, Any] | None:
        require_pair(user_id, agent_id)
        return next((x for x in self._rows_for_scope(self.memories.get((user_id, agent_id), []), user_id, agent_id) if x['id'] == memory_id), None)
