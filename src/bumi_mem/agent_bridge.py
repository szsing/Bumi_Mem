"""Agent 桥接辅助函数：无依赖、无网络 I/O 的载荷构建和上下文合并。

不导入 Pipecat，不初始化音频/模型，不在 speech/control 回调中执行网络 I/O。
将返回的冻结载荷入队，由 worker/outbox 发送。
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, tzinfo
from collections.abc import Mapping

GUEST_USER_ID = '__guest__'              # 陌生/未识别用户的保留 ID
MEMORY_BLOCK_PREFIX = '[Bumi memory supplement]\n'  # 记忆补充块的标识前缀


def _id(value):
    """校验并返回合法的稳定 ID：1~128 个可打印字符，且首尾不能是空白。"""
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 128 or not value.isprintable():
        raise ValueError('a stable printable ID of 1..128 characters is required')
    return value


def client_turn_id(session_id: str, timeline_turn_id: str) -> str:
    """组合会话 ID 和时间线轮次 ID 为全局唯一的客户端轮次 ID。

    时间线计数器会在重启后归零，因此需要用持久化的 session_id 作为前缀。
    """
    _id(session_id)
    _id(timeline_turn_id)
    return _id(session_id + ':' + timeline_turn_id)


def timeline_occurred_at(timeline, *, timezone_info: tzinfo | None = None) -> str:
    """保留对话的本地日期，用于 昨天/今天 归一化。

    如果机器人 OS 时区与用户时区不同，需显式提供 timezone_info。
    永远不要用单调秒表替代挂钟输入。
    """
    timestamp = timeline.speech_started_wall_time
    if timezone_info is not None:
        return datetime.fromtimestamp(timestamp, timezone_info).isoformat()
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


def speaker_for_memory(identity, *, enabled: bool, mode: str) -> dict:
    """构建说话人身份载荷。shadow/disabled 模式不得静默启用个人记忆。"""
    if not enabled or mode != 'label':
        return {'status': 'disabled'}
    if identity is None:
        return {'status': 'unknown'}
    data = identity if isinstance(identity, Mapping) else identity.history_metadata()
    status = data.get('status', 'unknown')
    allowed = {'matched', 'unknown', 'timeout', 'unavailable', 'insufficient_audio', 'disabled'}
    if status not in allowed:
        raise ValueError('unknown speaker status')
    if status != 'matched':
        return {'status': status}
    profile = _id(data.get('profile_id'))
    if profile == GUEST_USER_ID:
        raise ValueError('a matched speaker cannot use the reserved guest category')
    # 不发送生物识别分数、模型内部信息或可变显示标签
    return {'status': 'matched', 'profile_id': profile}


def prepare_payload(*, query: str, request_turn_id: str, identity=None,
                    identity_enabled: bool = False, identity_mode: str = 'shadow',
                    occurred_at: str | None = None, prefetch_id: str | None = None) -> dict:
    """构建 prepare 操作载荷：终稿文本提交给记忆服务进行检索和上下文组装。

    参数:
        query:            终稿文本（已稳定的完整转录，不支持原始音频）
        request_turn_id:  请求轮次 ID
        identity:         说话人身份对象
        identity_enabled: 是否启用身份识别
        identity_mode:    身份模式（'shadow'/'label'）
        occurred_at:      事件发生时间（ISO 8601）
        prefetch_id:      预取 ID（若前缀预取成功则复用其结果）
    返回:
        prepare 操作载荷字典
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError('memory requires a finalized text transcript; raw audio bypass is unsupported')
    result = {'operation': 'prepare', 'client_turn_id': _id(request_turn_id), 'query': query.strip(),
              'speaker': speaker_for_memory(identity, enabled=identity_enabled, mode=identity_mode)}
    if occurred_at is not None:
        result['occurred_at'] = occurred_at
    if prefetch_id is not None:
        result['prefetch_id'] = _id(prefetch_id)  # 复用预取结果
    return result


def prefetch_payload(*, query_prefix: str, request_turn_id: str, identity=None,
                     identity_enabled: bool = False, identity_mode: str = 'shadow',
                     prefix_revision: int | None = None) -> dict:
    """流式前缀预取：在 ASR 稳定前缀上提前发起检索，不写 L1、不建轮次。

    只传已经稳定的前缀（稳定判据由 Agent 的 ASR 决定），不要传还在变的尾字。
    返回的 prefetch_id 交给同一 client_turn_id 的 prepare_payload 复用。
    """
    if not isinstance(query_prefix, str) or not query_prefix.strip():
        raise ValueError('prefetch requires a non-empty stabilized prefix')
    result = {'operation': 'prefetch', 'client_turn_id': _id(request_turn_id),
              'query_prefix': query_prefix.strip(),
              'speaker': speaker_for_memory(identity, enabled=identity_enabled, mode=identity_mode)}
    if prefix_revision is not None:
        result['prefix_revision'] = int(prefix_revision)
    return result


def settled_speech_snapshot(speech, *, occurred_at: str | None = None) -> dict:
    """Call ONLY from SpeechProgress.on_complete (after commit), not on LLM end.

    settled is set before commit in the Agent's implementation, so checking that
    flag outside its callback is NOT by itself a sufficient synchronization point.
    DDS publication is the measured boundary, not physical speaker playback.
    """
    if not speech.settled:
        raise ValueError('speech has not settled; do not submit generated text')
    # 语音结算不建立最终 LLM/工具任务结果
    # 将此证据与协调器的显式结果合并后再发送
    result = {'reply': speech.committed_text,
              'speech': {'delivery_basis': 'dds_successful_publish',
                         'alignment_covers_output': bool(speech.alignment_covers_output()),
                         'alignment_status': speech.alignment_status,
                         'output_samples': speech.output_samples, 'sample_rate': speech.sample_rate}}
    if occurred_at is not None:
        result['occurred_at'] = occurred_at
    return deepcopy(result)


def complete_payload_from_speech(*, turn_id: str, speech, status: str,
                                 occurred_at: str | None = None) -> dict:
    if status not in {'completed', 'interrupted', 'cancelled', 'failed'}:
        raise ValueError('an explicit terminal task outcome is required')
    return dict(operation='complete', turn_id=_id(turn_id), status=status,
                **settled_speech_snapshot(speech, occurred_at=occurred_at))


def local_control_payload(*, event_id: str, history_message: dict, occurred_at=None) -> dict:
    """构建本地控制事件载荷：将程序拥有的 settled 事件转为记忆服务可接受的格式。

    只接受 ConversationState 中已确认的 program-owned 事件（role='system'），
    不接受用户或 LLM 伪造的控制指令。

    参数:
        event_id:         事件唯一 ID
        history_message:  历史消息字典（必须 role='system'，content 为 JSON）
        occurred_at:      事件发生时间
    返回:
        agent_event 操作载荷字典
    """
    import json
    if history_message.get('role') != 'system':
        raise ValueError('expected a program-owned local control event')
    event = json.loads(history_message['content'])
    if event.pop('event', None) != 'local_control':
        raise ValueError('not a local_control event')
    # 剥离 event 字段后剩余部分作为 control 内容
    result = {'operation': 'agent_event', 'event_id': _id(event_id), 'kind': 'local_control', 'control': event}
    if occurred_at is not None:
        result['occurred_at'] = occurred_at
    return result


def merge_agent_context(agent_messages: list[dict], response: dict, *, expected_user_id: str,
                        expected_client_turn_id: str) -> list[dict]:
    """将记忆补充块合并到 Agent 消息列表中，返回只读视图，不修改原始 history/tools。

    同一用户轮次的每个 tool 迭代应复用同一个已接受的响应，
    不要在 _stream_current_turn 进入下一个 tool 迭代时再次调用 prepare。
    """
    if (response.get('operation') != 'prepare' or not response.get('has_prompt') or
            response.get('user_id') != expected_user_id or
            response.get('client_turn_id') != expected_client_turn_id):
        raise ValueError('stale or foreign memory response')
    blocks = response.get('memory_context')
    if not isinstance(blocks, list) or not blocks:
        raise ValueError('missing memory supplement')
    if any(not isinstance(m, dict) or m.get('role') != 'system' or not isinstance(m.get('content'), str) for m in blocks):
        raise ValueError('unexpected memory supplement shape')
    messages = deepcopy([m for m in agent_messages if not (
        m.get('role') == 'system' and isinstance(m.get('content'), str) and
        m['content'].startswith(MEMORY_BLOCK_PREFIX))])
    insert_at = 1 if messages and messages[0].get('role') == 'system' else 0
    messages[insert_at:insert_at] = [{'role': 'system', 'content': MEMORY_BLOCK_PREFIX + '\n\n'.join(m['content'] for m in blocks)}]
    return messages


def cold_start_spoken_history(response: dict, *, existing_history: list[dict], expected_user_id: str,
                              expected_client_turn_id: str) -> list[dict]:
    """冷启动口语历史恢复：仅在 Agent 历史为空时，从服务响应中恢复 L1 口语消息。

    在 add_user_text 之前调用。当前查询已在服务响应末尾，故意排除。
    不重建 tool 执行状态；绝不用此覆盖活跃的 Agent 历史。
    """
    # 安全检查：绝不允许用 L1 口语历史覆盖已有 Agent/tool 历史
    if existing_history:
        raise ValueError('never replace existing Agent/tool history with spoken L1')
    if (response.get('operation') != 'prepare' or response.get('user_id') != expected_user_id or
            response.get('client_turn_id') != expected_client_turn_id or not response.get('has_prompt')):
        raise ValueError('stale or foreign recovery context')
    messages = response.get('messages')
    if (not isinstance(messages, list) or len(messages) < 2 or messages[0].get('role') != 'system' or
            messages[-1].get('role') != 'user'):
        raise ValueError('unexpected spoken history shape')
    recovered = messages[1:-1]
    if any(m.get('role') not in {'user', 'assistant', 'system'} or not isinstance(m.get('content'), str) for m in recovered):
        raise ValueError('spoken history cannot restore tool execution state')
    return deepcopy(recovered)