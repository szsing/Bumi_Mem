"""按本轮召回优先级选择可见引用；不改变引用锚点或扩大可见范围。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from bumi_mem.shared_memory import reference_key

logger = logging.getLogger(__name__)
ReferenceIdentity = tuple[str, str]
ReferenceSource = Literal['current', 'historical']


def _identity(row: dict[str, Any]) -> ReferenceIdentity:
    # 记忆 ID 必须与所有者一起匹配，避免别人的共享引用获得当前用户优先级。
    return str(row.get('user_id') or ''), reference_key(row)


@dataclass(frozen=True)
class ReferenceOmission:
    identity: ReferenceIdentity
    source: ReferenceSource
    reason: Literal['reference_char_budget', 'not_visible', 'empty_content', 'duplicate_content']
    item_chars: int


@dataclass
class ReferenceSelection:
    items: list[dict[str, Any]]
    selected_identities: list[ReferenceIdentity]
    current_identities: set[ReferenceIdentity]
    omissions: list[ReferenceOmission]
    used_chars: int
    max_chars: int


def select_reference_items(
    refs: Sequence[dict[str, Any]], *, current_recalled: Sequence[dict[str, Any]],
    max_chars: int, project: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> ReferenceSelection:
    """先按检索返回顺序选本轮引用，剩余预算再选最新历史引用。

    current_recalled 只提供优先级，不能直接成为输出事实；事实只能来自
    refs 中已通过可见性检查的行。重复召回的旧锚点无需移动或续期。
    完整记录与 JSON 分隔符计入预算，不截断事实句子。
    """
    if max_chars < 0:
        raise ValueError('max_reference_chars must be nonnegative')
    from bumi_mem.recall_context import selection_order
    ranked_current = selection_order(current_recalled) if any((r.get("metadata") or {}).get("backend")=="graphiti" for r in current_recalled) else current_recalled
    current_order = list(dict.fromkeys(_identity(row) for row in ranked_current))
    if not current_recalled and any((r.get("metadata") or {}).get("backend")=="graphiti" for r in refs):
        current_order = list(dict.fromkeys(_identity(row) for row in selection_order(refs)))
    current_keys = set(current_order)
    visible = {_identity(row): row for row in refs
               if row.get('reference_status', 'active') == 'active'}
    omissions = [ReferenceOmission(key, 'current', 'not_visible', 0)
                 for key in current_order if key not in visible]
    candidates: list[tuple[dict[str, Any], ReferenceSource]] = [
        (visible[key], 'current') for key in current_order if key in visible]
    candidates.extend((row, 'historical') for row in selection_order(refs)
                      if _identity(row) not in current_keys and
                      row.get('reference_status', 'active') == 'active')
    current_items, historical_items, selected_keys = [], [], []
    seen, used = set(), 0
    for row, source in candidates:
        identity = _identity(row)
        item = project(row)
        if item is None:
            omissions.append(ReferenceOmission(identity, source, 'empty_content', 0))
            continue
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if encoded in seen:
            omissions.append(ReferenceOmission(identity, source, 'duplicate_content', len(encoded)))
            continue
        seen.add(encoded)
        # 第一条包含数组括号，后续条目包含 ", " 分隔符，均占两个字符。
        # 没有事实时允许输出空数组，即使事实预算为零。
        size = len(encoded) + 2
        if used + size > max_chars:
            omissions.append(ReferenceOmission(identity, source, 'reference_char_budget', size))
            continue
        (current_items if source == 'current' else historical_items).append(item)
        selected_keys.append(identity)
        used += size
    # 本轮按相关性顺序展示；未命中的历史引用保持原有对话顺序。
    # 诊断 ID 必须与实际输出的每条记录一一对应，不能保留历史选取时的逆序。
    current_count = len(current_items)
    output_keys = selected_keys[:current_count] + list(reversed(selected_keys[current_count:]))
    return ReferenceSelection(current_items + list(reversed(historical_items)), output_keys,
                              current_keys, omissions, used, max_chars)


def log_reference_selection(selection: ReferenceSelection) -> None:
    """诊断只记录 ID、数量和预算，不记录事实正文或来源消息内容。"""
    logger.info('memory_reference_selection current_ids=%s selected_ids=%s '
                'used_chars=%s max_chars=%s',
                sorted(selection.current_identities), selection.selected_identities,
                selection.used_chars, selection.max_chars)
    for omission in selection.omissions:
        level = logging.DEBUG if omission.reason == 'duplicate_content' else logging.WARNING
        logger.log(level, 'memory_reference_omitted owner_user_id=%s memory_id=%s '
                   'source=%s reason=%s item_chars=%s used_chars=%s max_chars=%s',
                   *omission.identity, omission.source, omission.reason, omission.item_chars,
                   selection.used_chars, selection.max_chars)
