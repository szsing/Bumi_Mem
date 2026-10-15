"""Submit immutable session snapshots and execute them outside the chat path."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

from bumi_mem.long_term import LongTermMemory
from bumi_mem.identity import is_guest
from bumi_mem.prompts import MEMORY_EXTRACTION_INSTRUCTIONS, TEMPORAL_EXTRACTION_INSTRUCTIONS, summary_messages
from bumi_mem.providers import TextProvider
from bumi_mem.schemas import (
    ConsolidationSubmission, JobCancelledError, MemoryJob, MessageRecord,
    MemoryWriteUncertainError, RetryableMemoryError,
)
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import TokenCounter


@dataclass(slots=True)
class ConsolidatorConfig:
    backend: str = 'legacy'
    extraction_turns: int = 10
    extraction_tokens: int = 8000
    carryover_messages: int = 20
    # Effective L1 window for the Graphiti backend. Extraction frequency and the
    # L1 window are deliberately separate knobs: changing one must not silently
    # shrink the other. This travels in the persisted job snapshot, so a queued
    # batch keeps the window it was submitted under.
    l1_max_messages: int | None = None
    local_chunk_tokens: int = 3500
    summary_max_tokens: int = 512
    max_attempts: int = 3
    retry_base_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.extraction_turns <= 0 or self.extraction_tokens <= 0:
            raise ValueError('extraction limits must be positive')
        if self.l1_max_messages is not None and self.l1_max_messages <= 0:
            raise ValueError('l1_max_messages must be positive when set')
        if self.carryover_messages < 0 or self.local_chunk_tokens <= 0 or self.summary_max_tokens <= 0:
            raise ValueError('invalid consolidation size limits')
        if self.max_attempts < 1 or self.retry_base_seconds < 0:
            raise ValueError('invalid retry configuration')


class SessionConsolidator:
    def __init__(self, *, store: SQLiteSessionStore, long_term: LongTermMemory,
                 summary_provider: TextProvider, memory_provider_name: str,
                 memory_token_counter: TokenCounter, config: ConsolidatorConfig | None = None) -> None:
        self.store = store
        self.long_term = long_term
        self.summary_provider = summary_provider
        self.memory_provider_name = memory_provider_name
        self.memory_token_counter = memory_token_counter
        self.config = config or ConsolidatorConfig()

    def _settings(self) -> dict[str, Any]:
        # No API keys/endpoints are persisted. Model identity prevents silent
        # resumption with a different provider after a deployment change.
        return {**asdict(self.config), 'provider': self.memory_provider_name,
                'summary_model': self.summary_provider.config.model,
                'extraction_prompt': None if self.config.backend == 'graphiti' else MEMORY_EXTRACTION_INSTRUCTIONS}

    @staticmethod
    def _chat_messages(messages: Sequence[MessageRecord]) -> list[dict[str, str]]:
        return [m.to_extraction_message() for m in messages
                if not m.context_only and m.role in ("user", "assistant")]

    @staticmethod
    def _restore_extraction_chunks(
        chunks: Sequence[dict[str, Any]], sources: Sequence[MessageRecord],
    ) -> list[list[dict[str, str]]]:
        """Recover pending pre-provenance chunks from their immutable snapshot.

        Match in source order (including successful chunks and duplicate text).
        Never synthesize an anchor from a job or guess for an unmatched message.
        """
        cursor = 0
        restored = []
        for chunk in chunks:
            records = []
            for rendered in json.loads(chunk['messages_json']):
                found = None
                while cursor < len(sources):
                    source = sources[cursor]
                    cursor += 1
                    if rendered.get('source_message_id'):
                        matches = rendered['source_message_id'] == source.message_id
                    else:
                        matches = (rendered.get('role') == source.role and
                                   rendered.get('content') == source.content)
                    if matches:
                        found = source
                        break
                if found is None:
                    raise MemoryWriteUncertainError('Persisted extraction message has no original source anchor; explicit source repair required')
                records.append(found)
            restored.append(SessionConsolidator._chat_messages(records))
        return restored

    def _local_chunks(self, messages: Sequence[MessageRecord], *, token_limit: int | None = None) -> list[list[MessageRecord]]:
        limit = self.config.local_chunk_tokens if token_limit is None else token_limit
        rounds: list[list[MessageRecord]] = []
        for message in messages:
            if message.role == 'user' or not rounds:
                rounds.append([])
            rounds[-1].append(message)
        chunks: list[list[MessageRecord]] = []
        chunk: list[MessageRecord] = []
        tokens = 0
        for turn in rounds:
            size = sum(self.memory_token_counter.count_text(m.to_extraction_message()["content"]) + 4
                       for m in turn)
            if self.config.backend == 'graphiti' and size > limit:
                if chunk:
                    chunks.append(chunk)
                    chunk, tokens = [], 0
                for message in turn:
                    cost = self.memory_token_counter.count_text(message.to_extraction_message()['content']) + 4
                    if chunk and tokens + cost > limit:
                        chunks.append(chunk)
                        chunk, tokens = [], 0
                    chunk.append(message)
                    tokens += cost
                continue
            if chunk and tokens + size > limit:
                chunks.append(chunk)
                chunk, tokens = [], 0
            chunk.extend(turn)
            tokens += size
        if chunk:
            chunks.append(chunk)
        return chunks

    def _bounded_graph_messages(self, messages, limit):
        result = []
        for message in messages:
            rest = message.content
            while rest:
                lo, hi = 1, len(rest)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    view = replace(message, content=rest[:mid]).to_extraction_message()
                    if self.memory_token_counter.count_text(view['content']) + 4 <= limit:
                        lo = mid
                    else:
                        hi = mid - 1
                fragment = replace(message, content=rest[:lo])
                if self.memory_token_counter.count_text(fragment.to_extraction_message()['content']) + 4 > limit:
                    raise ValueError('Graph extraction token budget is too small for a source header')
                result.append(fragment)
                rest = rest[lo:]
        return result

    def submit(self, session_id: str, *, reason: str) -> ConsolidationSubmission:
        return self.store.submit_shared(session_id, reason=reason, settings=self._settings(),
                                 carryover_messages=self.config.carryover_messages,
                                 max_attempts=self.config.max_attempts)

    def recover_legacy(self) -> None:
        for session in self.store.legacy_sessions():
            self.store.submit(session.session_id, reason='legacy_recovery', settings=self._settings(),
                              carryover_messages=self.config.carryover_messages,
                              max_attempts=self.config.max_attempts, legacy=True)

    def execute(self, job: MemoryJob, owner: str) -> None:
        """Worker-only. Submission/rotation never calls this method."""
        try:
            self.store.assert_current(job.job_id, owner)
            if job.kind == 'delete':
                # delete_all is idempotent; reads stay suppressed until this is
                # durably acknowledged, including on restart/failure.
                self.long_term.delete_all(user_id=job.user_id, agent_id=job.agent_id)
                self.store.finish_job(job.job_id, owner)
                return
            if is_guest(job.user_id):
                self.store.finish_job(job.job_id, owner)
                return
            payload = json.loads(job.payload_json)
            settings = payload['settings']
            if settings['provider'] != self.memory_provider_name or settings['summary_model'] != self.summary_provider.config.model:
                self.store.fail_job(job.job_id, owner, 'Provider/model differs from submitted job; restore configuration before retry.',
                                    retry_delay=0, retryable=False)
                return
            sources = [MessageRecord(**m) for m in payload['messages']]
            messages = sources
            # Carryover and synthetic recall context are never fresh L3 evidence.
            messages = [m for m in messages if not m.context_only and m.role in ('user', 'assistant')]
            if any((m.user_id and m.user_id != job.user_id) or
                   (m.agent_id and m.agent_id != job.agent_id) for m in messages):
                self.store.fail_job(job.job_id, owner, 'Foreign speaker in owner snapshot',
                                    retry_delay=0, retryable=False)
                return
            session = self.store.get_session(job.session_id)
            if settings.get('backend') != 'graphiti' and session.summary_status not in ('success', 'fallback'):
                try:
                    summary = self.summary_provider.generate(summary_messages(self._chat_messages(messages)),
                                                             max_tokens=settings['summary_max_tokens'], temperature=0.0).strip()
                    if not summary:
                        raise RuntimeError('summary provider returned an empty response')
                    self.store.save_job_summary(job.job_id, owner, summary=summary,
                                                provider=self.summary_provider.name, status='success')
                except JobCancelledError:
                    raise
                except Exception as exc:
                    user_count = sum(m.role == 'user' for m in messages)
                    assistant_count = sum(m.role == 'assistant' for m in messages)
                    self.store.save_job_summary(job.job_id, owner,
                                                summary=f'会话包含 {user_count} 条用户消息和 {assistant_count} 条机器人消息',
                                                provider=self.summary_provider.name, status='fallback', error=str(exc))
            if settings.get('backend') == 'graphiti':
                messages = self._bounded_graph_messages(messages, settings['local_chunk_tokens'])
            chunks = (self._local_chunks(messages, token_limit=settings['local_chunk_tokens'])
                      if settings['provider'] == 'local' or settings.get('backend') == 'graphiti' else ([messages] if messages else []))
            self.store.ensure_chunks(job.job_id, owner, [self._chat_messages(c) for c in chunks])
            persisted = self.store.chunks(job.job_id)
            if settings.get('backend') == 'graphiti':
                restored = [json.loads(c['messages_json']) for c in persisted]
                source_ids = {m.message_id for m in sources}
                if any(m.get('source_message_id') not in source_ids for chunk in restored for m in chunk):
                    raise MemoryWriteUncertainError('Graph fragment has no original source anchor')
            else:
                restored = self._restore_extraction_chunks(persisted, sources)
            for chunk, extraction_messages in zip(persisted, restored):
                self.store.assert_current(job.job_id, owner)
                if chunk['status'] == 'succeeded':
                    continue
                prior_status = chunk['status']
                if prior_status != 'pending':
                    raise MemoryWriteUncertainError('chunk write outcome unknown; explicit replay required')
                if not extraction_messages:
                    self.store.set_chunk(job.job_id, owner, chunk['chunk_id'], status='succeeded', events=[])
                    continue
                self.store.set_chunk(job.job_id, owner, chunk['chunk_id'], status='writing')
                metadata = {'source': 'l1_session', 'session_id': job.session_id,
                            'consolidation_id': chunk['chunk_id'], 'consolidation_attempt': job.attempts,
                            'chunk_index': chunk['chunk_index'], 'chunk_count': len(persisted),
                            'job_id': job.job_id, 'memory_generation': job.generation}
                metadata['source_message_ids'] = [m['source_message_id'] for m in extraction_messages
                                                  if m.get('source_message_id')]
                metadata['source_messages'] = [
                    {'source_message_id': m['source_message_id'], 'observed_at': m.get('observed_at', '')}
                    for m in extraction_messages if m.get('source_message_id')]
                prompt = settings['extraction_prompt']
                if settings.get('backend') != 'graphiti' and TEMPORAL_EXTRACTION_INSTRUCTIONS not in prompt:
                    prompt += '\n\n' + TEMPORAL_EXTRACTION_INSTRUCTIONS
                # An explicit replay authorization, if one was issued for exactly
                # this chunk, is claimed once here and then consumed or burnt.
                grant = None
                if settings.get('backend') == 'graphiti' and self.store.replay_authorization(chunk['chunk_id']):
                    grant = self.store.take_replay_authorization(
                        chunk_id=chunk['chunk_id'], user_id=job.user_id, agent_id=job.agent_id,
                        generation=job.generation,
                        input_digest=self.store.chunk_input_digest(extraction_messages))
                try:
                    events = self.long_term.add_session(user_id=job.user_id, agent_id=job.agent_id,
                                                       messages=extraction_messages, metadata=metadata,
                                                       prompt=prompt, replay_authorization=grant)
                except RetryableMemoryError as exc:
                    if grant:
                        self.store.record_replay_outcome(chunk_id=chunk['chunk_id'], success=False,
                                                         detail=f'retryable failure during replay: {exc}')
                    self.store.set_chunk(job.job_id, owner, chunk['chunk_id'], status='pending', error=str(exc))
                    raise
                except Exception as exc:
                    if grant:
                        self.store.record_replay_outcome(chunk_id=chunk['chunk_id'], success=False,
                                                         detail=f'{type(exc).__name__}: {exc}')
                    self.store.set_chunk(job.job_id, owner, chunk['chunk_id'], status='uncertain', error=str(exc))
                    raise MemoryWriteUncertainError(f'Write outcome unknown: {exc}') from exc
                if grant:
                    self.store.record_replay_outcome(chunk_id=chunk['chunk_id'], success=True,
                                                     detail='replay produced a receipt')
                self.store.set_chunk(job.job_id, owner, chunk['chunk_id'], status='succeeded', events=events)
            self.store.finish_job(job.job_id, owner)
        except JobCancelledError:
            # A queued deletion will remove any in-flight old-generation write.
            return
        except Exception as exc:
            delay = min(60.0, self.config.retry_base_seconds * (2 ** min(job.attempts - 1, 6)))
            try:
                self.store.fail_job(job.job_id, owner, str(exc), retry_delay=delay,
                                    retryable=not isinstance(exc, MemoryWriteUncertainError))
            except JobCancelledError:
                pass
