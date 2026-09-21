"""Small, dependency-free helpers for the actual Noetix bumi-agent repository.

No Pipecat imports, no audio/model initialization, no network IO in speech/control
callbacks. Queue the returned frozen payload and send it from a worker/outbox.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, tzinfo
from collections.abc import Mapping

GUEST_USER_ID = '__guest__'
MEMORY_BLOCK_PREFIX = '[Bumi memory supplement]\n'


def _id(value):
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 128 or not value.isprintable():
        raise ValueError('a stable printable ID of 1..128 characters is required')
    return value


def client_turn_id(session_id: str, timeline_turn_id: str) -> str:
    """Timeline counters restart: prefix them with a persisted run/session ID."""
    _id(session_id)
    _id(timeline_turn_id)
    return _id(session_id + ':' + timeline_turn_id)


def timeline_occurred_at(timeline, *, timezone_info: tzinfo | None = None) -> str:
    """Keep the conversation's LOCAL day for yesterday/today normalization.

    Supply its timezone explicitly if the robot OS timezone is not the user's
    timezone. Never replace the wall-clock input with a monotonic stopwatch.
    """
    timestamp = timeline.speech_started_wall_time
    if timezone_info is not None:
        return datetime.fromtimestamp(timestamp, timezone_info).isoformat()
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


def speaker_for_memory(identity, *, enabled: bool, mode: str) -> dict:
    """Shadow/disabled identification must not silently enable personal memory."""
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
    # Do not send biometric scores, model internals, or mutable display labels.
    return {'status': 'matched', 'profile_id': profile}


def prepare_payload(*, query: str, request_turn_id: str, identity=None,
                    identity_enabled: bool = False, identity_mode: str = 'shadow',
                    occurred_at: str | None = None) -> dict:
    if not isinstance(query, str) or not query.strip():
        raise ValueError('memory requires a finalized text transcript; raw audio bypass is unsupported')
    result = {'operation': 'prepare', 'client_turn_id': _id(request_turn_id), 'query': query.strip(),
              'speaker': speaker_for_memory(identity, enabled=identity_enabled, mode=identity_mode)}
    if occurred_at is not None:
        result['occurred_at'] = occurred_at
    return result


def settled_speech_snapshot(speech, *, occurred_at: str | None = None) -> dict:
    """Call ONLY from SpeechProgress.on_complete (after commit), not on LLM end.

    settled is set before commit in the Agent's implementation, so checking that
    flag outside its callback is NOT by itself a sufficient synchronization point.
    DDS publication is the measured boundary, not physical speaker playback.
    """
    if not speech.settled:
        raise ValueError('speech has not settled; do not submit generated text')
    # Speech settlement does not establish the final LLM/tool task outcome.
    # Join this evidence with the coordinator's explicit outcome before sending.
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
    """Use ConversationState's settled program-owned event, not user/LLM claims."""
    import json
    if history_message.get('role') != 'system':
        raise ValueError('expected a program-owned local control event')
    event = json.loads(history_message['content'])
    if event.pop('event', None) != 'local_control':
        raise ValueError('not a local_control event')
    result = {'operation': 'agent_event', 'event_id': _id(event_id), 'kind': 'local_control', 'control': event}
    if occurred_at is not None:
        result['occurred_at'] = occurred_at
    return result


def merge_agent_context(agent_messages: list[dict], response: dict, *, expected_user_id: str,
                        expected_client_turn_id: str) -> list[dict]:
    """Return a request-only view; never mutate ConversationAgent.history/tools.

    Reuse the SAME accepted response on every tool round of one user turn. Do not
    call prepare again when _stream_current_turn enters another tool iteration.
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
    """Optional recovery of spoken L1 into an EMPTY Agent history only.

    Call before add_user_text. The current query is already last in the service
    response and is deliberately excluded. No tools or execution state are
    reconstructed; never use this to overwrite an active Agent history.
    """
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
