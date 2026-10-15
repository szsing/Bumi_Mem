"""Structured single-turn response built directly from persisted memory rows.

No model call and no parsing of the old prompt string. Internal scores, jobs,
message IDs and database keys are intentionally excluded from the Agent API.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Sequence

from bumi_mem.schemas import MessageRecord
from bumi_mem.turn_prompt import _core_description, _event_description, _memory_body, _STATUS_TEXT

MAX_REFERENCE_CHARS = 6000
MAX_HISTORY_ROWS = 16


def _unique(values):
    return list(dict.fromkeys(str(value) for value in values if value not in (None, '')))


def memory_times(row: dict[str, Any]) -> list[str]:
    """Expose semantic/source times, never storage-only record update time."""
    metadata = row.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    values = [metadata.get('valid_at'), metadata.get('invalid_at'), metadata.get('expired_at'), row.get('event_at'), metadata.get('event_at'), row.get('event_date'),
              metadata.get('event_date'), row.get('observed_at'), metadata.get('observed_at')]
    sources = metadata.get('source_messages')
    if isinstance(sources, list):
        values.extend(source.get('observed_at') for source in sources if isinstance(source, dict))
    return _unique(values)


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


def _reference_item(row: dict[str, Any]) -> dict[str, Any] | None:
    content = _memory_body(row)
    if not content or str(row.get('reference_status', 'active')) != 'active':
        return None
    fact_status = str(row.get('fact_status') or '')
    if fact_status in {'historical_superseded', 'historical_version'}:
        content = '历史：' + content
    return {'user_id': str(row.get('user_id') or ''), 'content': content, 'time': memory_times(row)}


def _bounded_references(refs, max_chars):
    selected, seen, used = [], set(), 0
    for row in reversed(refs):
        item = _reference_item(row)
        if item is None:
            continue
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        size = len(key) + (1 if selected else 0)
        if used + size > max_chars:
            continue
        selected.append(item)
        used += size
    selected.reverse()
    return selected


def long_term_items(*, refs: Sequence[dict[str, Any]], core_rows: Sequence[dict[str, Any]],
                    history_rows: Sequence[dict[str, Any]], max_reference_chars: int) -> list[dict[str, Any]]:
    if max_reference_chars < 0:
        raise ValueError('max_reference_chars must be nonnegative')
    items = _bounded_references(refs, max_reference_chars)
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
                        history_rows: Sequence[dict[str, Any]], max_reference_chars: int) -> dict[str, Any]:
    return {
        'turn_id': turn_id,
        'current_user': user_id,
        'long_term_memory': long_term_items(
            refs=refs, core_rows=core_rows, history_rows=history_rows,
            max_reference_chars=max_reference_chars),
        'short_term_memory': short_term_items(selected),
    }
