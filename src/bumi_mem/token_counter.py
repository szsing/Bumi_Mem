"""Provider-aware token counting."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Protocol, Sequence


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: Sequence[dict[str, str]]) -> int:
        return sum(self.count_text(item.get("content", "")) + 4 for item in messages)


class ApproxTokenCounter:
    """Conservative counter for OpenAI-compatible APIs without a local tokenizer."""

    _cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    _latin_word = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        cjk_count = len(self._cjk.findall(text))
        remaining = self._cjk.sub(" ", text)
        units = self._latin_word.findall(remaining)
        latin_count = 0
        for unit in units:
            latin_count += max(1, math.ceil(len(unit) / 4)) if unit.isalnum() else 1
        return max(1, cjk_count + latin_count)

    def count_messages(self, messages: Sequence[dict[str, str]]) -> int:
        return sum(self.count_text(item.get("content", "")) + 4 for item in messages)


class HuggingFaceTokenCounter:
    """Exact-enough local counter loaded once from a local Hugging Face model."""

    def __init__(self, model_path: str | Path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_messages(self, messages: Sequence[dict[str, str]]) -> int:
        try:
            rendered = self.tokenizer.apply_chat_template(
                list(messages), tokenize=True, add_generation_prompt=False
            )
            return len(rendered)
        except Exception:
            return sum(self.count_text(item.get("content", "")) + 4 for item in messages)


class FixedTokenCounter:
    """Small deterministic counter used by tests."""

    def __init__(self, per_message: int = 1) -> None:
        self.per_message = per_message

    def count_text(self, text: str) -> int:
        return self.per_message if text else 0

    def count_messages(self, messages: Sequence[dict[str, str]]) -> int:
        return self.per_message * len(messages)
