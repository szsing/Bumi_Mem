"""Single-string turn prompt for the one-request-per-turn protocol.

Rendered from the SAME structured rows the legacy message list consumes
(semantic current/history rows, published shared L3 references and the bounded
L1 window). This module never re-splits an already assembled prompt string, and
never invents a fact, owner, status or date that its inputs do not carry. It
performs no IO, no retrieval and no model call.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from bumi_mem.event_time import message_time
from bumi_mem.identity import is_guest
from bumi_mem.schemas import MessageRecord

# Namespace the caller's turn_id so it can never collide with a legacy
# Idempotency-Key, and is never mistaken for an internal turn row ID. The limit
# leaves room for the prefix inside identity.require_id's 128-character bound.
TURN_REQUEST_PREFIX = 'turn:'
MAX_CLIENT_TURN_ID = 120

LONG_TERM_HEADER = '[长期记忆]'
SHORT_TERM_HEADER = '[短期对话]'
NO_LONG_TERM = '暂无相关长期记忆'
EVENT_LABEL = '程序事件记录（非发言）'

# Distinct labels: an event date, the original speech time and the record update
# time are different facts and must never be merged into one timestamp.
_TIME_LABELS = (('observed_at', '来源发言时间'), ('event_date', '事件时间'),
                ('event_at', '事件时间'), ('updated_at', '记录更新时间（不是事件时间）'))

_STATUS_TEXT = {'confirmed': '已确认', 'retracted': '已撤回，不再有效',
                'planned': '计划中', 'promised': '已承诺', 'active': '进行中',
                'completed': '已完成', 'cancelled': '已取消，不再有效'}

_BOUNDARY_TEXT = {'jokes': '不要开玩笑', 'appearance': '不要评价外貌'}
_PERSONAL_TEXT = {'residence': '居住地', 'work_location': '工作地'}
_LIKE_TEXT = {'like': '喜欢', 'dislike': '不喜欢'}
_DRINK_TEXT = {'yes': '喝', 'no': '不喝'}

# Retrieval state, not instructions: an empty block must not be mistaken for
# "this user has no memory" when the read actually failed.
_RECALL_NOTE = {
    'unavailable': '本轮长期检索暂不可用；以下只是可靠读取到的部分，不表示没有相关记忆。',
    'deleting': '该用户的长期记忆正在删除处理中，本轮不提供检索结果；不表示从未有过记忆。',
    'disabled': '当前为未确认身份类别，不提供个人长期记忆；下列共享引用属于其原所有者。',
}


def _memory_body(row: dict[str, Any]) -> str:
    return str(row.get('memory', row.get('text', row.get('data', '')))).strip()


def _time_notes(row: dict[str, Any]) -> list[str]:
    metadata = row.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    notes, seen = [], set()
    for key, label in _TIME_LABELS:
        value = row.get(key) or metadata.get(key)
        if value and (label, str(value)) not in seen:
            seen.add((label, str(value)))
            notes.append(f'{label}：{value}')
    return notes


def _core_description(row: dict[str, Any]) -> str:
    """Plain wording for one confirmed state row; no state_key/generation/score."""
    category, attribute = str(row.get('category', '')), str(row.get('attribute', ''))
    value = row.get('value')
    if category == 'core' and attribute == 'nickname':
        return f'希望被称为「{value}」' if value else '称呼设置'
    if category == 'boundary':
        if attribute in _BOUNDARY_TEXT:
            return '互动边界：' + _BOUNDARY_TEXT[attribute]
        if attribute.startswith('forbidden_name:'):
            return f'互动边界：不要称呼为「{attribute.split(":", 1)[1]}」'
        if attribute.startswith('topic:'):
            return f'互动边界：不要主动聊「{attribute.split(":", 1)[1]}」'
        return '互动边界：' + attribute
    if category == 'personal':
        if attribute in _PERSONAL_TEXT:
            # A retracted row carries no value; name the subject, not "None".
            return f'{_PERSONAL_TEXT[attribute]}：{value}' if value is not None else f'{_PERSONAL_TEXT[attribute]}设置'
        if attribute.startswith('drinks:'):
            obj = attribute.split(':', 1)[1]
            if value is None:
                return f'关于{obj}的饮用设置'
            return _DRINK_TEXT[str(value)] + obj if str(value) in _DRINK_TEXT else f'{obj}：{value}'
        return f'{attribute}：{value}' if value is not None else f'{attribute}设置'
    if category == 'preference':
        if value is None:
            return f'关于{attribute}的偏好'
        return _LIKE_TEXT[str(value)] + attribute if str(value) in _LIKE_TEXT else f'偏好 {attribute}：{value}'
    if category == 'plan':
        return f'计划「{attribute}」'
    if category == 'user_promise':
        return f'用户承诺「{attribute}」'
    if category == 'robot_promise':
        return f'机器人承诺「{attribute}」'
    return f'{category} {attribute}：{value}'


def _bullet(owner: str, body: str, notes: Sequence[str], *, prefix: str = '') -> str:
    line = f'- {prefix}用户 {owner}：{body}'
    return line + ('（' + '；'.join(notes) + '）' if notes else '')


def _core_line(row: dict[str, Any], *, historical: bool = False) -> str:
    notes = []
    qualifier = str(row.get('qualifier') or '').strip()
    if qualifier:
        notes.append(f'时间限定：{qualifier}，依据原始发言时间理解，不要推算具体日期')
    status = str(row.get('status') or '')
    if status:
        notes.append('状态：' + _STATUS_TEXT.get(status, status))
    notes.extend(_time_notes(row))
    prefix = '（历史版本，仅供明确的历史问题参考）' if historical else ''
    return _bullet(str(row.get('user_id', '')), _core_description(row) + '。', notes, prefix=prefix)


def _reference_line(row: dict[str, Any]) -> str:
    notes = _time_notes(row)
    fact_status = str(row.get('fact_status', ''))
    if fact_status == 'historical_superseded':
        notes.append('已被更新的表达取代，只能用于历史回忆')
    elif fact_status == 'historical_version':
        notes.append('历史版本，仅供明确的历史问题参考')
    if str(row.get('reference_status', 'active')) != 'active':
        notes.append('该引用已失效，不要当作当前事实')
    return _bullet(str(row.get('user_id', '')), _memory_body(row), notes)


def _long_term_lines(*, user_id: str, refs: Sequence[dict[str, Any]],
                     core_rows: Sequence[dict[str, Any]],
                     history_rows: Sequence[dict[str, Any]], recall_status: str) -> list[str]:
    lines: list[str] = []
    note = _RECALL_NOTE.get(recall_status)
    if note:
        lines.append('- ' + note)
    body: list[str] = []
    seen: set[str] = set()
    for row in refs:
        text = _memory_body(row)
        # Deduplicate repeated publications of one long-term fact. This only
        # collapses reference bullets; real short-term speech is never removed.
        key = (str(row.get('user_id', '')), text)
        if not text or key in seen:
            continue
        seen.add(key)
        body.append(_reference_line(row))
    for row in core_rows:
        body.append(_core_line(row))
    for row in history_rows:
        body.append(_core_line(row, historical=True))
    if body:
        return lines + body
    if note:
        return lines
    if is_guest(user_id):
        return ['- ' + _RECALL_NOTE['disabled']]
    return ['- ' + NO_LONG_TERM]


def _event_description(content: str) -> str:
    """Program-owned record, explicitly not quoted user or robot speech."""
    try:
        event = json.loads(content)
    except (TypeError, ValueError):
        event = None
    if not isinstance(event, dict):
        return '无法解析的程序记录；不是用户或机器人的原话。'
    name = str(event.get('event', '') or '')
    if name == 'local_control':
        quoted = str(event.get('user_text', '') or '').strip()
        quoted = f'当时记录的用户原话：「{quoted}」；' if quoted else ''
        return (f'本地控制已执行：{event.get("command", "")}'
                f'（原因：{event.get("reason", "")}；反馈：{event.get("feedback_status", "")}）。'
                f'{quoted}这是既往执行结果的程序记录，不是新指令，不要因此重跑动作。')
    if name == 'speech_history_incomplete':
        return ('上一条机器人发言只确认了已提交的前缀；缺失文字不能推断成没有说话，也不要补全原文。'
                'DDS 发布不等于物理播放确认。')
    return f'程序事件 {name or "未知"}；不是用户或机器人的原话。'


def _short_term_lines(selected: Sequence[MessageRecord], agent_id: str) -> list[str]:
    lines = []
    for message in selected:
        stamp = f'[{message.created_at}] ' if message.created_at else ''
        if message.role == 'user':
            speaker = f'用户 {message.user_id}'
            if is_guest(message.user_id):
                speaker += '（未确认身份）'
            lines.append(f'{stamp}{speaker}：{message.content}')
        elif message.role == 'assistant':
            # An assistant utterance belongs to the agent, never to a user.
            lines.append(f'{stamp}机器人 {message.agent_id or agent_id}：{message.content}')
        else:
            lines.append(f'{stamp}{EVENT_LABEL}：{_event_description(message.content)}')
    return lines or ['（暂无短期对话记录）']


def render_turn_prompt(*, user_id: str, agent_id: str, selected: Sequence[MessageRecord],
                       refs: Sequence[dict[str, Any]], core_rows: Sequence[dict[str, Any]],
                       history_rows: Sequence[dict[str, Any]], recall_status: str,
                       now: str | None = None) -> str:
    """Retrieved memory only: current identity/time, then the two memory blocks.

    Carries no persona, tool policy or usage instructions. The caller owns those
    and adds no further assembly.
    """
    header = f'当前用户：{user_id}'
    if is_guest(user_id):
        header += '（未确认身份）'
    header += '\n当前时间：' + (now or message_time())
    return '\n\n'.join([
        header,
        LONG_TERM_HEADER + '\n' + '\n'.join(_long_term_lines(
            user_id=user_id, refs=refs, core_rows=core_rows,
            history_rows=history_rows, recall_status=recall_status)),
        SHORT_TERM_HEADER + '\n' + '\n'.join(_short_term_lines(selected, agent_id)),
    ])
