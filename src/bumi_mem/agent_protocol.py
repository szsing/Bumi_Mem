"""Explicit, retry-safe wire protocol for bumi-agent's custom Pipecat 1.8 loop.

These models do not import Pipecat or any robot/ASR/TTS dependency. The legacy
three-field exchange remains available on the same route.
"""
from __future__ import annotations

import json
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bumi_mem.event_time import message_time
from bumi_mem.identity import GUEST_USER_ID, require_id


class AgentModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    @field_validator('client_turn_id', 'turn_id', 'event_id', 'user_id', 'prefetch_id',
                     check_fields=False)
    @classmethod
    def validate_id(cls, value):
        return require_id(value) if value is not None else None

    @field_validator('occurred_at', check_fields=False)
    @classmethod
    def validate_time(cls, value):
        return message_time(value) if value is not None else None


class SpeakerIdentityInput(BaseModel):
    # Match the agent's history_metadata(); discard scores/biometric diagnostics.
    model_config = ConfigDict(extra='ignore', strict=True)
    status: Literal['matched', 'unknown', 'timeout', 'unavailable', 'insufficient_audio', 'disabled']
    profile_id: str | None = Field(default=None, max_length=128)
    label: str | None = Field(default=None, max_length=128)

    @model_validator(mode='after')
    def valid_matched(self):
        if self.status == 'matched':
            require_id(self.profile_id)
            if self.profile_id == GUEST_USER_ID:
                raise ValueError('matched identity cannot be the reserved guest category')
        return self

    @property
    def memory_user_id(self):
        return self.profile_id if self.status == 'matched' else GUEST_USER_ID


class SpeechEvidence(AgentModel):
    """Text coverage of published output, NOT a physical loudspeaker acknowledgement."""
    delivery_basis: Literal['dds_successful_publish', 'text_only', 'unknown'] = 'text_only'
    alignment_covers_output: bool = True
    alignment_status: str = Field(default='text_only', min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    output_samples: int | None = Field(default=None, ge=0)
    sample_rate: int | None = Field(default=None, ge=1, le=384000)

    @model_validator(mode='after')
    def coherent_evidence(self):
        if (self.output_samples is None) != (self.sample_rate is None):
            raise ValueError('output sample count and sample rate must be supplied together')
        if self.delivery_basis == 'dds_successful_publish' and self.output_samples is None:
            raise ValueError('DDS evidence requires sample count and sample rate')
        if self.delivery_basis == 'unknown' and self.alignment_covers_output:
            raise ValueError('unknown delivery cannot assert full transcript coverage')
        return self


class AgentPrepareRequest(AgentModel):
    operation: Literal['prepare']
    client_turn_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=16000)
    user_id: str | None = Field(default=None, max_length=128)
    speaker: SpeakerIdentityInput | None = None
    occurred_at: str | None = Field(default=None, max_length=64)
    # 可选：来自 operation=prefetch 的候选集；不匹配/过期/删除后都会静默重算。
    prefetch_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator('query')
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError('a finalized transcript is required')
        return value.strip()

    @model_validator(mode='after')
    def coherent_owner(self):
        if self.speaker is None and self.user_id is None:
            raise ValueError('provide a trusted user_id or speaker result')
        if self.speaker is not None and self.user_id is not None and self.user_id != self.speaker.memory_user_id:
            raise ValueError('user_id disagrees with the speaker identity')
        return self

    @property
    def memory_user_id(self):
        return self.speaker.memory_user_id if self.speaker is not None else self.user_id


class AgentPrefetchRequest(AgentModel):
    """流式前缀预取：在 ASR 稳定前缀上提前发起检索。

    不写 L1、不建轮次、不发布引用；返回的 prefetch_id 交给随后的 prepare 复用。
    """
    operation: Literal['prefetch']
    client_turn_id: str = Field(min_length=1, max_length=128)
    query_prefix: str = Field(min_length=1, max_length=16000)
    user_id: str | None = Field(default=None, max_length=128)
    speaker: SpeakerIdentityInput | None = None
    prefix_revision: int | None = Field(default=None, ge=0)

    @field_validator('query_prefix')
    @classmethod
    def nonblank_prefix(cls, value):
        if not value.strip():
            raise ValueError('a non-empty streaming prefix is required')
        return value.strip()

    @model_validator(mode='after')
    def coherent_owner(self):
        if self.speaker is None and self.user_id is None:
            raise ValueError('provide a trusted user_id or speaker result')
        if self.speaker is not None and self.user_id is not None and self.user_id != self.speaker.memory_user_id:
            raise ValueError('user_id disagrees with the speaker identity')
        return self

    @property
    def memory_user_id(self):
        return self.speaker.memory_user_id if self.speaker is not None else self.user_id


class AgentCompleteRequest(AgentModel):
    operation: Literal['complete']
    turn_id: str = Field(min_length=1, max_length=128)
    status: Literal['completed', 'interrupted', 'cancelled', 'failed']
    reply: str = Field(default='', max_length=64000)
    occurred_at: str | None = Field(default=None, max_length=64)
    speech: SpeechEvidence = Field(default_factory=SpeechEvidence)

    @model_validator(mode='after')
    def coherent_output(self):
        if self.status == 'cancelled' and self.reply.strip():
            raise ValueError('cancelled means no speech; use interrupted/failed for a partial reply')
        if (self.reply.strip() and self.speech.delivery_basis == 'dds_successful_publish'
                and self.speech.output_samples == 0):
            raise ValueError('a DDS transcript cannot precede every published sample')
        return self


class LocalControlEvent(AgentModel):
    command: Literal['stop', 'sleep']
    user_text: str = Field(default='', max_length=4000)
    reason: Literal['user_requested_stop', 'user_requested_sleep']
    result: dict[str, Any]
    feedback_status: str = Field(min_length=1, max_length=64)
    llm_followup: Literal[False] = False

    @model_validator(mode='after')
    def reason_matches(self):
        if len(json.dumps(self.result, ensure_ascii=False, allow_nan=False)) > 8000:
            raise ValueError('control result must be a compact structured summary, not audio/images')
        if self.reason != 'user_requested_' + self.command:
            raise ValueError('control command and reason disagree')
        return self


class AgentEventRequest(AgentModel):
    operation: Literal['agent_event']
    event_id: str = Field(min_length=1, max_length=128)
    kind: Literal['speech', 'local_control']
    occurred_at: str | None = Field(default=None, max_length=64)
    text: str | None = Field(default=None, max_length=64000)
    speech: SpeechEvidence | None = None
    control: LocalControlEvent | None = None

    @model_validator(mode='after')
    def coherent_event(self):
        if self.kind == 'speech':
            if self.control is not None or self.speech is None:
                raise ValueError('speech events require evidence and cannot contain control data')
            if not (self.text or '').strip() and self.speech.alignment_covers_output:
                raise ValueError('empty complete speech is not an independent utterance')
            if (self.text or '').strip() and self.speech.delivery_basis == 'dds_successful_publish' and not self.speech.output_samples:
                raise ValueError('DDS speech requires published samples')
        elif self.control is None or self.text is not None or self.speech is not None:
            raise ValueError('local_control requires only control data')
        return self


class AgentChatMessage(AgentModel):
    role: Literal['system', 'user', 'assistant']
    content: str


class MemorySupplement(AgentModel):
    role: Literal['system'] = 'system'
    content: str


class AgentPrepareResponse(AgentModel):
    operation: Literal['prepare'] = 'prepare'
    turn_id: str
    client_turn_id: str
    user_id: str
    agent_id: str
    source_message_id: str
    occurred_at: str
    messages: list[AgentChatMessage]
    memory_context: list[MemorySupplement]
    has_prompt: Literal[True] = True
    recall_status: Literal['ok', 'unavailable', 'deleting', 'disabled']
    core_write: dict[str, Any]


class AgentPrefetchResponse(AgentModel):
    """预取回执：只给状态与候选数，**不含 messages**，避免被误用为回答上下文。"""
    operation: Literal['prefetch'] = 'prefetch'
    prefetch_id: str | None = None
    client_turn_id: str
    user_id: str
    agent_id: str
    recall_status: Literal['ok', 'unavailable', 'deleting', 'not_requested']
    candidate_count: int = Field(ge=0)
    expires_in_ms: int = Field(ge=0)
    took_ms: float = Field(ge=0)


class AgentCompleteResponse(AgentModel):
    operation: Literal['complete'] = 'complete'
    turn_id: str
    user_id: str
    agent_id: str
    status: Literal['completed', 'interrupted', 'cancelled', 'failed']
    has_prompt: Literal[False] = False
    reply_saved: bool
    transcript_complete: bool
    consolidation_status: Literal['not_triggered', 'queued', 'succeeded']


class AgentEventResponse(AgentModel):
    operation: Literal['agent_event'] = 'agent_event'
    event_id: str
    agent_id: str
    kind: Literal['speech', 'local_control']
    message_id: str
    has_prompt: Literal[False] = False
    saved: Literal[True] = True


class AgentPendingRequest(AgentModel):
    operation: Literal['pending']


class PendingAgentTurn(AgentModel):
    turn_id: str
    client_turn_id: str | None
    user_id: str
    protocol: Literal['agent', 'context', 'direct']
    status: Literal['preparing', 'awaiting_reply']


class AgentPendingResponse(AgentModel):
    operation: Literal['pending'] = 'pending'
    agent_id: str
    has_prompt: Literal[False] = False
    pending: PendingAgentTurn | None
