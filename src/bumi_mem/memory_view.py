"""One compact model-facing view of visible long-term facts."""
from __future__ import annotations

from typing import Any

from bumi_mem.provenance import display_memory_body


def memory_times(row: dict[str, Any], *, labelled: bool = False) -> list[str]:
    """Keep event/source anchors distinct; never expose storage update times.

    The structured API retains its existing timestamp-list contract. Chat uses
    short labels because it has no external schema explaining temporal roles.
    """
    metadata = row.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    result = []
    for key in ("valid_at", "invalid_at", "expired_at"):
        value = metadata.get(key)
        if value:
            result.append(f"{key}={value}" if labelled else str(value))
    for key, label in (('event_at', 'event_at'), ('event_date', 'event_date'),
                       ('observed_at', 'source_at')):
        for value in (row.get(key), metadata.get(key)):
            if value not in (None, ''):
                result.append(f'{label}={value}' if labelled else str(value))
    sources = metadata.get('fact_source_messages') or metadata.get('source_messages')
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict) and source.get('observed_at'):
                value = str(source['observed_at'])
                label = 'source_at' if metadata.get('fact_source_messages') else 'input_at'
                result.append(f'{label}={value}' if labelled else value)
    return list(dict.fromkeys(result))


def reference_item(row: dict[str, Any], *, labelled_times: bool = False) -> dict[str, Any] | None:
    """Rendering cannot add visibility or change owner identity."""
    from bumi_mem.recall_context import reference_body
    projected = dict(row, memory=display_memory_body(row))
    content = reference_body(projected)
    if not content or row.get('reference_status', 'active') != 'active':
        return None
    if row.get('fact_status') in {'historical_superseded', 'historical_version'}:
        content = '历史：' + content
    else:
        status = {'planned':'计划中','promised':'已承诺','completed':'已完成',
                  'cancelled':'已取消','retracted':'已撤回'}.get(row.get('fact_status'))
        if status:
            content = status + '：' + content
    return {'user_id': str(row.get('user_id') or ''), 'content': content,
            'time': memory_times(row, labelled=labelled_times)}
