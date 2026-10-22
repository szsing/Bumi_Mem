"""确定性日期溯源：基于消息时间戳归一化相对日期，不查询工作线程时钟。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re


@dataclass(frozen=True, slots=True)
class NormalizedRelativeDates:
    text: str
    original_text: str
    observed_at: str


# 较长的英文表达不得被部分重写为 yesterday/tomorrow
_RELATIVE_DAY = re.compile(
    r"\b(?:the\s+)?day\s+(?:before\s+yesterday|after\s+tomorrow)\b|"
    r"\b(?:yesterday|today|tomorrow)\b|昨天|今天|明天", re.IGNORECASE
)
_OFFSETS = {"yesterday": -1, "today": 0, "tomorrow": 1,
            "昨天": -1, "今天": 0, "明天": 1}


def normalize_relative_dates(text: str, observed_at: str) -> NormalizedRelativeDates:
    """Resolve simple day words against the source's *local* calendar day.

    Missing/naive/invalid anchors and vague terms (recently/最近) remain untouched.
    This is lexical normalization, not event extraction or speaker attribution:
    questions, quotations and modality remain in place and require LLM filtering.
    The original wording and exact timestamp are always retained alongside text.
    """
    try:
        anchor = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        if anchor.utcoffset() is None:
            raise ValueError("an offset is required")
    except (ValueError, TypeError, AttributeError):
        return NormalizedRelativeDates(text, text, observed_at)

    def replace(match: re.Match[str]) -> str:
        offset = _OFFSETS.get(match.group().lower())
        if offset is None:
            return match.group()
        try:
            return (anchor.date() + timedelta(days=offset)).isoformat()
        except OverflowError:
            return match.group()

    return NormalizedRelativeDates(_RELATIVE_DAY.sub(replace, text), text, observed_at)