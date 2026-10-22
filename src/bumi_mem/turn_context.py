"""结构化单轮响应：直接从持久化记忆行构建，无需模型调用和旧提示词解析。

内部评分、作业、消息 ID 和数据库键故意不暴露给 Agent API。
"""
from __future__ import annotations

from typing import Any, Sequence

from bumi_mem.schemas import MessageRecord
from bumi_mem.reference_selection import select_reference_items, log_reference_selection
from bumi_mem.turn_prompt import _core_description, _event_description, _STATUS_TEXT
from bumi_mem.memory_view import memory_times, reference_item as _reference_item

MAX_REFERENCE_CHARS = 24000
MAX_HISTORY_ROWS = 16


def _core_content(row: dict[str, Any], *, historical: bool = False) -> str:
    content = _core_description(row).rstrip('。')
    qualifier = str(row.get('qualifier') or '').strip()
    status = str(row.get('status') or '').strip()
    details = []
    if qualifier:
        details.append('时间限定：' + qualifier)
    # confirmed is the default useful state; avoid a redundant label. Other
    # states materially change meaning and therefore remain in the content.
    if status and status != 'confirmed':
        details.append(_STATUS_TEXT.get(status, status))
    if details:
        content += '（' + '；'.join(details) + '）'
    return ('历史：' if historical else '') + content + '。'


def long_term_items(*, refs: Sequence[dict[str, Any]], core_rows: Sequence[dict[str, Any]],
                    history_rows: Sequence[dict[str, Any]], max_reference_chars: int,
                    current_recalled: Sequence[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    references = select_reference_items(
        refs, current_recalled=current_recalled,
        max_chars=max_reference_chars, project=_reference_item)
    log_reference_selection(references)
    items = references.items
    for row in core_rows:
        items.append({'user_id': str(row.get('user_id') or ''),
                      'content': _core_content(row), 'time': memory_times(row)})
    for row in history_rows[:MAX_HISTORY_ROWS]:
        items.append({'user_id': str(row.get('user_id') or ''),
                      'content': _core_content(row, historical=True), 'time': memory_times(row)})
    return items


def short_term_items(selected: Sequence[MessageRecord]) -> list[dict[str, str]]:
    items = []
    for message in selected:
        if message.role == 'user':
            speaker = message.user_id
            content = message.content
        elif message.role == 'assistant':
            speaker = message.agent_id
            content = message.content
        else:
            speaker = 'system'
            content = _event_description(message.content)
        items.append({'role': message.role, 'speaker_id': speaker,
                      'content': content, 'time': message.created_at})
    return items


def render_turn_context(*, turn_id: str, user_id: str, selected: Sequence[MessageRecord],
                        refs: Sequence[dict[str, Any]], core_rows: Sequence[dict[str, Any]],
                        history_rows: Sequence[dict[str, Any]], max_reference_chars: int,
                        current_recalled: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    return {
        'turn_id': turn_id,
        'current_user': user_id,
        'long_term_memory': long_term_items(
            refs=refs, core_rows=core_rows, history_rows=history_rows,
            max_reference_chars=max_reference_chars, current_recalled=current_recalled),
        'short_term_memory': short_term_items(selected),
    }
