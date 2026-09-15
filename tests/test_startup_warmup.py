from __future__ import annotations

import pytest

from bumi_mem.long_term import resolve_embed_device, warmup_mem0_backend


class RecordingMemory:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": []}


def test_mem0_warmup_uses_an_internal_non_user_scope():
    memory = RecordingMemory()
    warmup_mem0_backend(memory, query="boot warmup")
    call = memory.calls[0]
    assert call["query"] == "boot warmup"
    assert call["top_k"] == 1
    assert call["threshold"] == 0.0
    assert set(call["filters"]) == {"run_id"}
    assert call["filters"]["run_id"].startswith("__bumi_internal_warmup__")
    assert "user_id" not in call["filters"]
    assert "agent_id" not in call["filters"]


def test_server_startup_warms_before_health_is_ready(monkeypatch):
    from apps import mem0_locomo_server as server

    memory = RecordingMemory()
    monkeypatch.setattr(server, "_memory", memory)
    monkeypatch.setattr(server, "_warmup_elapsed_ms", None)
    monkeypatch.setattr(server, "_settings", lambda: {"embed_device": "cuda"})
    monkeypatch.setenv("BUMI_MEMORY_WARMUP", "true")
    elapsed_ms = server._startup_warmup()
    assert elapsed_ms >= 0
    assert server._warmup_elapsed_ms == elapsed_ms
    assert len(memory.calls) == 1
    assert server._startup_warmup() == elapsed_ms
    assert len(memory.calls) == 1


def test_embed_device_auto_uses_cuda_when_available():
    assert resolve_embed_device("auto", cuda_available=True) == "cuda"
    assert resolve_embed_device("auto", cuda_available=False) == "cpu"


def test_explicit_cuda_never_silently_falls_back():
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        resolve_embed_device("cuda", cuda_available=False)
