"""Stage-1 类型化合约：强类型的数据模型和校验规则。

认证身份由可信的上游代码提供，本模块不负责身份验证。
所有模型使用 Pydantic strict 模式，禁止额外字段，确保数据边界清晰。
"""
from __future__ import annotations
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
Purpose = Literal["conversation_continuity", "interaction_adaptation", "coordination.availability", "coordination.shared_activity", "explicit_memory_review"]
Kind = Literal["identity", "behavior", "episode", "record_index"]


def timestamp(value: str | None) -> float | None:
    """将 ISO 8601 字符串转为 Unix 时间戳；None 返回 None。要求带时区偏移。"""
    if value is None:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timestamps require a timezone offset")
    return dt.timestamp()


def iso(value: float | None) -> str | None:
    """将 Unix 时间戳转为 UTC ISO 8601 字符串；None 返回 None。"""
    return None if value is None else datetime.fromtimestamp(value, timezone.utc).isoformat()


class Contract(BaseModel):
    """合约基类：strict 模式、禁止额外字段、不可变、禁止 NaN。"""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False, revalidate_instances="always")


class Scope(Contract):
    """访问作用域：定义一次记忆访问的空间、主体、受众和用途。

    属性:
        space_id:          空间 ID
        actor_id:          行为者 ID（未知身份时为 None）
        audience_ids:      受众 ID 列表
        identity_status:   身份状态（'verified_by_upstream'/'unknown'）
        purpose:           用途
        audience_confirmed: 受众是否已确认
    """
    space_id: Identifier
    actor_id: Identifier | None
    audience_ids: list[Identifier]
    identity_status: Literal["verified_by_upstream", "unknown"]
    purpose: Purpose = "conversation_continuity"
    audience_confirmed: bool = True

    @model_validator(mode="after")
    def coherent(self):
        if (self.identity_status == "unknown") != (self.actor_id is None):
            raise ValueError("unknown identity has no actor_id; verified identity requires one")
        if len(self.audience_ids) != len(set(self.audience_ids)):
            raise ValueError("duplicate audience ID")
        return self

    @property
    def recipients(self) -> set[str]:
        return set(self.audience_ids) | ({self.actor_id} if self.actor_id else set())


class EventTime(Contract):
    """事件时间：支持未知/天/天部分/区间四种精度。

    属性:
        precision:  时间精度（'unknown'/'day'/'day_part'/'interval'）
        local_date: 本地日期（ISO 8601）
        day_part:   天的部分（'morning'/'afternoon'/'evening'/'night'）
        start_at:   区间起始时间
        end_at:     区间结束时间
    """
    precision: Literal["unknown", "day", "day_part", "interval"] = "unknown"
    local_date: str | None = None
    day_part: Literal["morning", "afternoon", "evening", "night"] | None = None
    start_at: str | None = None
    end_at: str | None = None

    @field_validator("start_at", "end_at")
    @classmethod
    def canonical_timestamp(cls, value):
        return iso(timestamp(value))

    @field_validator("local_date")
    @classmethod
    def canonical_date(cls, value):
        return None if value is None else date.fromisoformat(value).isoformat()

    @model_validator(mode="after")
    def valid_time(self):
        if self.local_date is not None:
            date.fromisoformat(self.local_date)
        start, end = timestamp(self.start_at), timestamp(self.end_at)
        if self.precision == "unknown" and any(v is not None for v in (self.local_date, self.day_part, start, end)):
            raise ValueError("unknown time cannot contain guessed precision")
        if self.precision == "day" and (self.local_date is None or self.day_part is not None or start is not None or end is not None):
            raise ValueError("day requires only local_date")
        if self.precision == "day_part" and (not self.local_date or not self.day_part or start is not None or end is not None):
            raise ValueError("day_part requires date and day part only")
        if self.precision == "interval" and (start is None or end is None or start >= end):
            raise ValueError("interval requires start < end")
        return self


class IdentityPayload(Contract):
    """身份载荷：存储身份断言（如称呼、别名）。"""
    predicate: Identifier
    value: str = Field(min_length=1)
    context: str = "default"


class BehaviorPayload(Contract):
    """行为载荷：存储交互偏好、互动边界等行为模式。"""
    pattern_type: Literal["explicit_interaction_preference", "interaction_pattern", "decision_pattern", "interaction_boundary"]
    preference: str = Field(min_length=1)
    topic_tags: list[str] = Field(default_factory=list)
    excluded_intents: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)


class EpisodePayload(Contract):
    """事件载荷：存储生活事件、共享经历、助手承诺等。"""
    event_type: Literal["activity_plan", "life_event", "shared_experience", "assistant_commitment"]
    activity: Literal["meal", "movie", "outing", "meeting", "exercise", "other"] = "other"
    event_status: Literal["planned", "tentative", "completed", "cancelled", "disputed"]
    availability: Literal["busy", "free", "unknown"] = "unknown"
    time: EventTime = Field(default_factory=EventTime)
    location: str | None = None
    details: str | None = None

    @model_validator(mode="after")
    def cancelled_is_not_busy(self):
        if self.event_status in {"cancelled", "disputed"} and self.availability != "unknown":
            raise ValueError("cancelled/disputed plans do not establish availability")
        return self


class RecordPayload(Contract):
    """记录索引载荷：存储事实摘要和关键词。"""
    gist: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)


PAYLOADS = {"identity": IdentityPayload, "behavior": BehaviorPayload, "episode": EpisodePayload, "record_index": RecordPayload}


class ObservationInput(Contract):
    """观测输入：一条对话消息的来源和内容。"""
    observation_id: Identifier
    conversation_id: Identifier
    revision: int = Field(default=1, ge=1)
    role: Literal["user", "assistant"] = "user"
    text: str = Field(min_length=1)
    source_timestamp: str
    is_final: bool = True
    storage_allowed: bool = False
    data_profile: Literal["synthetic_demo_14d"] = "synthetic_demo_14d"

    @field_validator("source_timestamp")
    @classmethod
    def aware_timestamp(cls, value):
        timestamp(value)
        return value


class SpanInput(Contract):
    observation_id: Identifier
    revision: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    quote: str = Field(min_length=1)
    offset_unit: Literal["unicode_codepoint"] = "unicode_codepoint"

    @model_validator(mode="after")
    def increasing(self):
        if self.end <= self.start:
            raise ValueError("span is a nonempty half-open interval")
        return self


class MemoryRef(Contract):
    memory_id: Identifier
    version: int = Field(ge=1)
    attributes: list[str] = Field(min_length=1)


class MemoryDraft(Contract):
    memory_id: Identifier
    kind: Kind
    subject_ids: list[Identifier] = Field(min_length=1)
    assertion_kind: Literal["explicit_statement", "reported_statement", "observed", "inferred", "assistant_commitment"]
    status: Literal["candidate", "active"] = "active"
    payload: dict[str, Any]
    span_ids: list[Identifier] = Field(default_factory=list)
    parent_memories: list[MemoryRef] = Field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None

    @model_validator(mode="after")
    def validate_payload(self):
        if len(self.subject_ids) != len(set(self.subject_ids)):
            raise ValueError("subject IDs must be unique")
        if len({(p.memory_id,p.version) for p in self.parent_memories}) != len(self.parent_memories):
            raise ValueError("duplicate parent evidence; merge declared attributes explicitly")
        PAYLOADS[self.kind].model_validate(self.payload)
        if not self.span_ids and not self.parent_memories:
            raise ValueError("at least one source or parent memory is required")
        if self.assertion_kind == "inferred" and self.status != "candidate":
            raise ValueError("inferred objects stay candidate until explicitly confirmed")
        start, end = timestamp(self.valid_from), timestamp(self.valid_until)
        if start is not None and end is not None and start >= end:
            raise ValueError("valid_from must precede valid_until")
        return self


class GrantInput(Contract):
    subject_id: Identifier
    audience_ids: list[Identifier] = Field(min_length=1)
    effect: Literal["allow", "deny"]
    category: Literal["coordination.availability", "coordination.shared_activity", "private", "*"]
    attributes: list[str] = Field(min_length=1)
    purposes: list[Purpose] = Field(min_length=1)
    memory_id: Identifier | None = None
    starts_at: str | None = None
    expires_at: str | None = None
    block_inference: bool = False

    @model_validator(mode="after")
    def validate_grant(self):
        if len(self.audience_ids) != len(set(self.audience_ids)):
            raise ValueError("duplicate recipient")
        valid_attrs = {"subject_ids", "text", "*"} | set().union(*(set(p.model_fields) for p in PAYLOADS.values()))
        if not set(self.attributes) <= valid_attrs:
            raise ValueError("unknown policy attribute")
        if self.effect == "allow" and (self.category in {"private", "*"} or "*" in self.attributes or self.block_inference):
            raise ValueError("allow must explicitly name a supported sharing category and fields")
        start, end = timestamp(self.starts_at), timestamp(self.expires_at)
        if start is not None and end is not None and start >= end:
            raise ValueError("grant start must precede expiry")
        return self


class ViewDraft(Contract):
    memory_id: Identifier
    source_version: int = Field(ge=1)
    snapshot_kind: Literal["current", "historical"] = "current"
    projection_type: Literal["private", "availability", "shared_activity"]
    attributes: list[str] = Field(min_length=1)
    audience_ids: list[Identifier] = Field(min_length=1)
    purposes: list[Purpose] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_fields(self):
        for values in (self.attributes, self.audience_ids, self.purposes):
            if len(values) != len(set(values)):
                raise ValueError("view fields, audience and purposes must be unique")
        return self


class Epochs(Contract):
    content: int
    policy: int
    deletion: int


class ReadTicket(Contract):
    context_fingerprint: str
    epochs: Epochs
    view_ids: list[str]
    temporal_mode: Literal["current", "historical"]
    as_of: str | None


class VisibleView(Contract):
    view_id: str
    view_revision: int
    kind: Kind
    snapshot_kind: Literal["current", "historical"]
    projected_fields: dict[str, Any]
    display_text: str
    validity_status: str
    assertion_kind: str
    status: str
    valid_from: str | None
    valid_until: str | None