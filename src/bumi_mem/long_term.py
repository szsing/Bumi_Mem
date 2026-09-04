"""Mem0-backed L3 adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from bumi_mem.providers import ProviderConfig


def unwrap_results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        results = value.get("results", [])
        return results if isinstance(results, list) else []
    return value if isinstance(value, list) else []


class LongTermMemory(Protocol):
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


@dataclass(slots=True, frozen=True)
class Mem0AdapterConfig:
    data_dir: Path
    embed_model: Path
    embed_device: str = "cpu"
    collection_name: str = "bumi_mem_l3"
    embedding_dims: int = 1024
    memory_max_tokens: int = 4096


class Mem0LongTermMemory:
    def __init__(self, llm: ProviderConfig, config: Mem0AdapterConfig) -> None:
        from mem0 import Memory

        self.provider_name = llm.provider
        self.llm_config = llm
        self.config = config
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
                            "device": config.embed_device,
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

    @staticmethod
    def _filters(user_id: str, agent_id: str) -> dict[str, str]:
        return {"user_id": user_id, "agent_id": agent_id}

    def search(
        self, *, user_id: str, agent_id: str, query: str, top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
        return unwrap_results(
            self.memory.search(
                query=query,
                top_k=top_k,
                filters=self._filters(user_id, agent_id),
                threshold=threshold,
                rerank=False,
            )
        )

    def add_session(
        self,
        *,
        user_id: str,
        agent_id: str,
        messages: Sequence[dict[str, str]],
        metadata: dict[str, Any],
        prompt: str,
    ) -> list[dict[str, Any]]:
        result = self.memory.add(
            list(messages),
            user_id=user_id,
            agent_id=agent_id,
            metadata=dict(metadata),
            prompt=prompt,
            infer=True,
        )
        return unwrap_results(result)

    def get_all(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        return unwrap_results(
            self.memory.get_all(
                filters=self._filters(user_id, agent_id),
                top_k=500,
            )
        )

    def delete_all(self, *, user_id: str, agent_id: str) -> None:
        self.memory.delete_all(user_id=user_id, agent_id=agent_id)


class FakeLongTermMemory:
    """In-memory test double with observable calls."""

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self.memories: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.search_results: list[dict[str, Any]] = []
        self.fail_add_count = 0

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
            return [dict(item) for item in self.search_results[:top_k]]
        return [dict(item) for item in self.memories.get((user_id, agent_id), [])[:top_k]]

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
            raise RuntimeError("fake Mem0 failure")
        user_texts = [item["content"] for item in messages if item.get("role") == "user"]
        text = " | ".join(user_texts) or "session memory"
        event = {"id": f"fake-{len(self.add_calls)}", "event": "ADD", "memory": text}
        self.memories.setdefault((user_id, agent_id), []).append(event)
        return [event]

    def get_all(self, *, user_id: str, agent_id: str) -> list[dict[str, Any]]:
        return [dict(item) for item in self.memories.get((user_id, agent_id), [])]

    def delete_all(self, *, user_id: str, agent_id: str) -> None:
        self.memories.pop((user_id, agent_id), None)
