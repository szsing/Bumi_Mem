from __future__ import annotations

from bumi_mem.cli import _provider_config


def test_deepseek_chat_and_memory_configs_are_independent(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("BUMI_CHAT_MODEL", "chat-model")
    monkeypatch.setenv("BUMI_MEMORY_MODEL", "memory-model")
    chat = _provider_config(
        "deepseek", purpose="chat", model_override=None, base_url_override=None, max_tokens=100
    )
    memory = _provider_config(
        "deepseek", purpose="memory", model_override=None, base_url_override=None, max_tokens=200
    )
    assert chat.model == "chat-model"
    assert memory.model == "memory-model"
    assert chat.max_tokens == 100
    assert memory.max_tokens == 200


def test_local_provider_does_not_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _provider_config(
        "local", purpose="chat", model_override=None, base_url_override=None, max_tokens=100
    )
    assert config.api_key == "local"
    assert config.base_url == "http://127.0.0.1:8000/v1"
