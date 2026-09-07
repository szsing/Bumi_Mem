"""OpenAI-compatible chat providers used for conversation and summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from bumi_mem.schemas import ProviderConfigurationError


@dataclass(slots=True, frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    max_tokens: int = 512
    temperature: float = 0.2


class TextProvider(Protocol):
    name: str
    config: ProviderConfig

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ProviderConfigurationError(
                f"{config.provider} provider is missing an API key"
            )
        from openai import OpenAI

        self.config = config
        self.name = config.provider
        self.client = OpenAI(
            api_key=config.api_key, base_url=config.base_url.rstrip("/"),
            timeout=60.0, max_retries=0,
        )

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature if temperature is None else temperature,
        )
        return response.choices[0].message.content or ""


class FakeTextProvider:
    """Deterministic provider for unit tests."""

    def __init__(self, name: str = "fake", responses: list[str] | None = None) -> None:
        self.name = name
        self.config = ProviderConfig(name, "fake-model", "http://fake", "fake")
        self.responses = list(responses or [])
        self.calls: list[list[dict[str, str]]] = []
        self.error: Exception | None = None

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        del max_tokens, temperature
        self.calls.append([dict(item) for item in messages])
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return "测试回答"
