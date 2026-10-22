"""单字符串轮次提示词：每轮一次请求协议的提示词渲染。

从与旧版消息列表相同的结构化行渲染（语义当前/历史行、已发布共享 L3 引用
和有界 L1 窗口）。本模块从不重新拆分已组装的提示词字符串，不虚构输入
未携带的事实/所有者/状态/日期。不执行 I/O、检索或模型调用。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Sequence

from bumi_mem.event_time import message_time
from bumi_mem.identity import is_guest
from bumi_mem.schemas import MessageRecord
from bumi_mem.provenance import display_memory_body

# Internal storage keys cannot be supplied by a legacy Idempotency-Key:
# identity.require_id rejects this non-printable unit separator. The explicit
# request_kind column additionally guards cache types and v7 migration.
TURN_REQUEST_PREFIX = '\x1fturn:'
LEGACY_TURN_REQUEST_PREFIX = 'turn:'
MAX_CLIENT_TURN_ID = 120
MAX_REFERENCE_CHARS = 24000
MAX_HISTORY_ROWS = 16
PERSISTENCE_UNCONFIRMED = '本轮新信息未确认长期保存；以下仍保留可靠读取到的既有记忆。'
REFERENCES_OMITTED = '部分长期引用未展示，以下不是全部可用记忆。'
HISTORY_OMITTED = '仅展示部分历史版本，不代表全部历史。'

LONG_TERM_HEADER = '[长期记忆]'
SHORT_TERM_HEADER = '[短期对话]'
NO_LONG_TERM = '暂无相关长期记忆'
EVENT_LABEL = '程序事件记录（非发言）'

# 展示事件和来源时间；存储更新时间只用于后台版本/审计，不进入回答上下文。
_TIME_LABELS = (('observed_at', '来源发言时间'), ('event_date', '事件时间'),
                ('event_at', '事件时间'))

_STATUS_TEXT = {'confirmed': '已确认', 'retracted': '已撤回，不再有效',
                'planned': '计划中', 'promised': '已承诺', 'active': '进行中',
                'completed': '已完成', 'cancelled': '已取消，不再有效'}  # 事实状态中文映射

_BOUNDARY_TEXT = {'jokes': '不要开玩笑', 'appearance': '不要评价外貌'}
_PERSONAL_TEXT = {'residence': '居住地', 'work_location': '工作地'}
_LIKE_TEXT = {'like': '喜欢', 'dislike': '不喜欢'}
_DRINK_TEXT = {'yes': '喝', 'no': '不喝'}

# 检索状态而非指令：空块不得被误解为"该用户没有记忆"（实际可能是读取失败）
_RECALL_NOTE = {
    'unavailable': '本轮长期检索暂不可用；以下只是可靠读取到的部分，不表示没有相关记忆。',
    'deleting': '该用户的长期记忆正在删除处理中，本轮不提供检索结果；不表示从未有过记忆。',
    'disabled': '当前为未确认身份类别，不提供个人长期记忆；下列共享引用属于其原所有者。',
}


def _memory_body(row: dict[str, Any]) -> str:
    from bumi_mem.recall_context import reference_body
    return reference_body(dict(row, memory=display_memory_body(row)))


def _inline_data(value: Any) -> str:
    """Reversible presentation escaping; never mutate the stored source text.

    Keep every record on one physical line. Brackets in data are escaped so a
    quoted section/role header cannot look like a service-authored delimiter.
    This preserves formatting boundaries, not immunity to semantic injection.
    """
    encoded = json.dumps(str(value), ensure_ascii=False)[1:-1]
    return ''.join(r'\u005b' if char == '[' else r'\u005d' if char == ']' else
                   json.dumps(char, ensure_ascii=True)[1:-1] if not char.isprintable() else char
                   for char in encoded)


def _time_notes(row: dict[str, Any]) -> list[str]:
    metadata = row.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    notes, seen = [], set()
    for key, label in _TIME_LABELS:
        value = row.get(key) or metadata.get(key)
        if value and (label, str(value)) not in seen:
            seen.add((label, str(value)))
            notes.append(f'{label}：{value}')
    # 整理存储分块级证据，而非逐事实的事件时间戳
    # 不虚构哪个原始消息支持哪条抽取事实
    sources = metadata.get('source_messages')
    times = list(dict.fromkeys(str(source['observed_at']) for source in sources
                  if isinstance(source, dict) and source.get('observed_at'))) if isinstance(sources, list) else []
    if len(times) == 1:
        if ('来源发言时间', times[0]) not in seen:
            notes.append(f'来源发言时间（抽取输入，非事件时间）：{times[0]}')
    elif times:
        try:
            parsed = [(datetime.fromisoformat(value.replace('Z', '+00:00')), value) for value in times]
            if any(stamp.utcoffset() is None for stamp, _ in parsed):
                raise ValueError('source timezone missing')
            first = min(parsed, key=lambda item: item[0])[1]
            last = max(parsed, key=lambda item: item[0])[1]
            notes.append(f'抽取输入时间范围（非事件起止）：{first} 至 {last}；{len(times)}个去重时间')
        except (ValueError, TypeError, OverflowError):
            notes.append('抽取输入记录时间（未推算时区或事件时间）：' + '、'.join(times))
    return notes


def _core_description(row: dict[str, Any]) -> str:
    """将一条核心状态行转为自然语言描述，不暴露内部字段（state_key/generation/score）。

    按类别分派：
    - core/nickname：称呼偏好
    - boundary：互动边界规则
    - personal：个人信息（居住地、工作地、饮用偏好等）
    - preference：偏好（喜欢/不喜欢）
    - plan/user_promise/robot_promise：计划和承诺
    """
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
    line = f'- {prefix}用户 {_inline_data(owner)}：{_inline_data(body)}'
    return line + ('（' + '；'.join(_inline_data(note) for note in notes) + '）' if notes else '')


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


def _bounded_reference_lines(refs, max_chars):
    lines, seen, used, omitted = [], set(), 0, False
    # Keep shared anchors newest first and preserve per-kind Graphiti rank.
    # Never cut a fact or its time in half.
    from bumi_mem.recall_context import selection_order
    for row in selection_order(refs):
        if not _memory_body(row) or row.get('reference_status', 'active') != 'active':
            continue
        line = _reference_line(row)
        if line in seen:
            continue
        seen.add(line)  # 时间/状态是行的一部分：不折叠它们
        size = len(line) + (1 if lines else 0)
        if used + size > max_chars:
            omitted = True
            continue
        lines.append(line)
        used += size
    return lines if any((r.get("metadata") or {}).get("backend")=="graphiti" for r in refs) else list(reversed(lines)), omitted


def _long_term_lines(*, user_id: str, refs: Sequence[dict[str, Any]],
                     core_rows: Sequence[dict[str, Any]], history_rows: Sequence[dict[str, Any]],
                     recall_status: str, core_write: dict[str, Any] | None = None,
                     max_reference_chars: int = MAX_REFERENCE_CHARS,
                     max_history_rows: int = MAX_HISTORY_ROWS) -> list[str]:
    lines: list[str] = []
    note = _RECALL_NOTE.get(recall_status)
    if note:
        lines.append('- ' + note)
    if (core_write or {}).get('status') == 'failed':
        lines.append('- ' + PERSISTENCE_UNCONFIRMED)
    body, omitted = _bounded_reference_lines(refs, max_reference_chars)
    if omitted:
        lines.append('- ' + REFERENCES_OMITTED)
    # 必需的当前规则不会被检索/历史预算牺牲
    body.extend(_core_line(row) for row in core_rows)
    body.extend(_core_line(row, historical=True) for row in history_rows[:max_history_rows])
    if len(history_rows) > max_history_rows:
        lines.append('- ' + HISTORY_OMITTED)
    if body or lines:
        return lines + body
    if is_guest(user_id):
        return ['- ' + _RECALL_NOTE['disabled']]
    return ['- ' + NO_LONG_TERM]


def _control_result(result):
    if not isinstance(result, dict):
        return '未知（没有可用的执行结果）'
    status = result.get('status')
    labels = {'stopped': '已停止', 'already_stopped': '原本已停止', 'sleeping': '已进入休眠',
              'not_started': '未开始', 'stop_requested': '已请求停止，尚未确认停止成功',
              'result_unknown': '未知', 'unknown': '未知', 'failed': '失败',
              'timeout': '等待超时，结果未确认', 'interrupted': '已中断',
              'cancelled': '已取消', 'completed': '程序报告已完成',
              'running': '进行中', 'active': '进行中', 'pending': '等待处理', 'queued': '已排队'}
    if status is not None:
        label = labels.get(str(status), f'未识别状态「{status}」，不能确认执行成功')
    elif result.get('stopped') is True:
        label = '程序报告已停止'
    else:
        label = '未知（没有明确的结果状态）'
    if result.get('started') is False:
        label += '；程序报告尚未开始执行'
    elif result.get('started') is True:
        label += '；程序报告曾开始执行'
    if result.get('confirmation') is not None:
        confirmations = {'pending': '等待确认', 'confirmed': '已确认', 'unknown': '未知'}
        value = result['confirmation']
        label += '；确认状态：' + confirmations.get(str(value), str(value))
    return label


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
        return (f'本地控制记录：{event.get("command", "")}'
                f'（执行结果：{_control_result(event.get("result"))}；'
                f'原因：{event.get("reason", "")}；反馈：{event.get("feedback_status", "")}）。'
                f'{quoted}这是既往执行结果的程序记录，不是新指令，不要因此重跑动作。')
    if name == 'speech_history_incomplete':
        return ('上一条机器人发言只确认了已提交的前缀；缺失文字不能推断成没有说话，也不要补全原文。'
                'DDS 发布不等于物理播放确认。')
    return f'程序事件 {name or "未知"}；不是用户或机器人的原话。'


def _short_term_lines(selected: Sequence[MessageRecord], agent_id: str) -> list[str]:
    lines = []
    for message in selected:
        stamp = f'[{_inline_data(message.created_at)}] ' if message.created_at else ''
        if message.role == 'user':
            speaker = f'用户 {_inline_data(message.user_id)}'
            if is_guest(message.user_id):
                speaker += '（未确认身份）'
            lines.append(f'{stamp}{speaker}：{_inline_data(message.content)}')
        elif message.role == 'assistant':
            # assistant 发言属于 agent，不属于用户
            lines.append(f'{stamp}机器人 {_inline_data(message.agent_id or agent_id)}：{_inline_data(message.content)}')
        else:
            lines.append(f'{stamp}{EVENT_LABEL}：{_inline_data(_event_description(message.content))}')
    return lines or ['（暂无短期对话记录）']


def render_turn_prompt(*, user_id: str, agent_id: str, selected: Sequence[MessageRecord],
                       refs: Sequence[dict[str, Any]], core_rows: Sequence[dict[str, Any]],
                       history_rows: Sequence[dict[str, Any]], recall_status: str,
                       now: str | None = None, core_write: dict[str, Any] | None = None,
                       max_reference_chars: int = MAX_REFERENCE_CHARS,
                       max_history_rows: int = MAX_HISTORY_ROWS) -> str:
    """Retrieved memory only: current identity/time, then the two memory blocks.

    Carries no persona or generic tool policy. Persistence/coverage notices are
    factual context, not a claim that a failed write never had any side effects.
    """
    if max_reference_chars < 0 or max_history_rows < 0:
        raise ValueError('prompt budgets must be nonnegative')
    header = f'当前用户：{_inline_data(user_id)}'
    if is_guest(user_id):
        header += '（未确认身份）'
    header += '\n当前时间：' + _inline_data(now or message_time())
    return '\n\n'.join([
        header,
        LONG_TERM_HEADER + '\n' + '\n'.join(_long_term_lines(
            user_id=user_id, refs=refs, core_rows=core_rows,
            history_rows=history_rows, recall_status=recall_status, core_write=core_write,
            max_reference_chars=max_reference_chars, max_history_rows=max_history_rows)),
        SHORT_TERM_HEADER + '\n' + '\n'.join(_short_term_lines(selected, agent_id)),
    ])
