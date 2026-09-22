"""OpenAI-compatible chat providers used for conversation and summarization."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
import os
from typing import Protocol, Sequence

from bumi_mem.schemas import ProviderConfigurationError


@dataclass(slots=True, frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    max_tokens: int = 512
    temperature: float = 0.2
    thinking_mode: str | None = None


def ensure_explicit_mem0_route():
    # Pinned Mem0's OpenAI adapter otherwise overrides the configured host/key.
    if os.getenv('OPENROUTER_API_KEY'):
        raise ProviderConfigurationError('Unset OPENROUTER_API_KEY for this service; it would override the explicit Mem0 LLM route')


def provider_request_options(config: ProviderConfig) -> dict:
    """Vendor-specific options are explicit; never leak them to other providers."""
    if config.thinking_mode is None:
        return {}
    if config.provider != 'deepseek' or config.thinking_mode not in {'enabled', 'disabled'}:
        raise ProviderConfigurationError('thinking_mode is only supported for the DeepSeek provider')
    return {'extra_body': {'thinking': {'type': config.thinking_mode}}}


def bind_provider_request_options(generate_response, config: ProviderConfig):
    """Bind request options to one pinned Mem0 LLM instance, not global SDK state.

    Its public generate_response(**kwargs) forwards extra_body to the OpenAI
    compatible client. Merge other extras but preserve the selected mode.
    """
    options = provider_request_options(config)
    if not options:
        return generate_response

    @wraps(generate_response)
    def generate(*args, **kwargs):
        kwargs['extra_body'] = {**(kwargs.get('extra_body') or {}), **options['extra_body']}
        return generate_response(*args, **kwargs)
    return generate


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
            **provider_request_options(self.config),
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
