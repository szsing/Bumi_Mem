"""Prompt assembly for L3 memories plus the bounded L1 window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from bumi_mem.prompts import chat_system_prompt
from bumi_mem.schemas import MessageRecord


@dataclass(slots=True)
class ContextBuilder:
    max_messages: int = 20
    max_l1_tokens: int = 8000

    def select_l1(self, messages: Sequence[MessageRecord]) -> list[MessageRecord]:
        """Keep the newest real messages plus carry-over when the token budget allows."""
        selected: list[MessageRecord] = []
        real_count = 0
        token_total = 0
        for message in reversed(messages):
            if not message.context_only and real_count >= self.max_messages:
                continue
            projected = token_total + message.token_count
            if selected and projected > self.max_l1_tokens:
                break
            selected.append(message)
            token_total = projected
            if not message.context_only:
                real_count += 1
        selected.reverse()
        return selected

    def build(
        self,
        *,
        recalled: Sequence[dict[str, Any]],
        l1_messages: Sequence[MessageRecord],
    ) -> list[dict[str, str]]:
        selected = self.select_l1(l1_messages)
        return [
            {"role": "system", "content": chat_system_prompt(recalled)},
            *(message.to_chat_message() for message in selected),
        ]
