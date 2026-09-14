"""Stage-0 structural contracts only; no policy engine or memory implementation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Purpose = Literal["conversation_continuity", "interaction_adaptation", "coordination.availability", "coordination.shared_activity", "explicit_memory_review"]

class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

class Scope(Contract):
    space_id: str = Field(min_length=1)
    actor_id: str | None
    audience_ids: list[str]
    identity_status: Literal["verified_by_upstream", "unknown"]
    purpose: Purpose
    policy_epoch: int = Field(default=0, ge=0)
    deletion_epoch: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def coherent_actor(self):
        if self.identity_status == "verified_by_upstream" and not self.actor_id:
            raise ValueError("verified identity requires actor_id")
        if self.identity_status == "unknown" and self.actor_id is not None:
            raise ValueError("unknown identity must not claim a person")
        if len(self.audience_ids) != len(set(self.audience_ids)):
            raise ValueError("audience IDs must be unique")
        return self

class Observation(Contract):
    observation_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    conversation_id: str = Field(min_length=1)
    scope: Scope
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)
    source_timestamp: str
    is_final: bool = True
    data_profile: Literal["synthetic_demo_14d"] = "synthetic_demo_14d"

    @model_validator(mode="after")
    def timestamp_has_zone(self):
        if datetime.fromisoformat(self.source_timestamp).tzinfo is None:
            raise ValueError("source_timestamp must include an offset")
        return self

class SourceSpan(Contract):
    observation_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    offset_unit: Literal["unicode_codepoint"] = "unicode_codepoint"

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self

class Memory(Contract):
    memory_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    kind: Literal["identity", "behavior", "episode", "record_index"]
    status: Literal["candidate", "active", "contested", "superseded", "expired", "deleted"]
    assertion_kind: Literal["explicit_statement", "reported_statement", "observed", "inferred", "assistant_commitment"]
    subject_ids: list[str]
    payload: dict[str, Any]
    evidence: list[SourceSpan]

class MemoryView(Contract):
    view_id: str = Field(min_length=1)
    view_revision: int = Field(ge=1)
    source_memory_id: str = Field(min_length=1)
    source_memory_version: int = Field(ge=1)
    snapshot_kind: Literal["current", "historical"]
    projection_type: Literal["private", "availability", "shared_activity"]
    display_text: str
    allowed_audience_ids: list[str]
    allowed_purposes: list[Purpose]
    policy_version: int = Field(ge=0)

class QueryView(Contract):
    raw_query: str = Field(min_length=1)
    retrieval_text: str = Field(min_length=1)
    subject_ids: list[str]
    temporal_mode: Literal["current", "historical", "timeline", "uncertain"]
    as_of_time: str | None = None
    purposes: list[Purpose]

class TimingBreakdown(Contract):
    ingest_ms: float | None = Field(default=None, ge=0)
    query_compose_ms: float | None = Field(default=None, ge=0)
    embedding_ms: float | None = Field(default=None, ge=0)
    sql_ms: float | None = Field(default=None, ge=0)
    fts_ms: float | None = Field(default=None, ge=0)
    vector_ms: float | None = Field(default=None, ge=0)
    merge_ms: float | None = Field(default=None, ge=0)
    bundle_pack_ms: float | None = Field(default=None, ge=0)
    memory_bundle_ms: float | None = Field(default=None, ge=0)
    extraction_visibility_ms: float | None = Field(default=None, ge=0)
    index_lag_ms: float | None = Field(default=None, ge=0)

class MemoryBundle(Contract):
    request_id: str = Field(min_length=1)
    scope: Scope
    hard_constraints: list[str] = Field(default_factory=list)
    identity_context: list[MemoryView] = Field(default_factory=list)
    active_preferences: list[MemoryView] = Field(default_factory=list)
    active_behavior_patterns: list[MemoryView] = Field(default_factory=list)
    context_events: list[MemoryView] = Field(default_factory=list)
    shared_views: list[MemoryView] = Field(default_factory=list)
    query_inferences: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    evidence_refs: list[SourceSpan] = Field(default_factory=list)
    retrieval_status: Literal["ok", "no_relevant_memory", "partial", "degraded", "identity_unknown", "policy_blocked", "pending_extraction"]
    persistence_status: Literal["persisted", "not_stored", "failed", "pending"]
    metrics: TimingBreakdown = Field(default_factory=TimingBreakdown)

class Expectations(Contract):
    must_support: list[str]
    must_not_disclose: list[str]
    uncertainty_notes: list[str]
    # Requirements for later stages, never counted as successful Stage-0 behavior.
    target_phase: int = Field(ge=1, le=6)

class SeedCase(Contract):
    case_id: str
    title: str
    setup: dict[str, Any]
    observations: list[Observation]
    query: Observation
    expected: Expectations

ALL_CONTRACTS = (Scope, Observation, SourceSpan, Memory, MemoryView, QueryView, TimingBreakdown, MemoryBundle, SeedCase)
