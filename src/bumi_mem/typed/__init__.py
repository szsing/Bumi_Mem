"""Stage-1 类型化记忆基础：可选的强类型记忆存储，不替代旧版对话引擎。"""
from .store import TypedMemoryStore
from .models import (Scope, EventTime, ObservationInput, SpanInput, MemoryDraft,
                     MemoryRef, GrantInput, ViewDraft, ReadTicket, VisibleView)
from .errors import AccessDenied, ConflictError, EvidenceError, SchemaMismatch, StaleRead

__all__ = ['TypedMemoryStore', 'Scope', 'EventTime', 'ObservationInput', 'SpanInput',
           'MemoryDraft', 'MemoryRef', 'GrantInput', 'ViewDraft', 'ReadTicket',
           'VisibleView', 'AccessDenied', 'ConflictError', 'EvidenceError',
           'SchemaMismatch', 'StaleRead']