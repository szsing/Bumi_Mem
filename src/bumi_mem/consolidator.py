"""Submit immutable session snapshots and execute them outside the chat path."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from bumi_mem.long_term import LongTermMemory
from bumi_mem.prompts import MEMORY_EXTRACTION_INSTRUCTIONS, summary_messages
from bumi_mem.providers import TextProvider
from bumi_mem.schemas import (
    ConsolidationSubmission, JobCancelledError, MemoryJob, MessageRecord,
    MemoryWriteUncertainError, RetryableMemoryError,
)
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import TokenCounter


@dataclass(slots=True)
class ConsolidatorConfig:
    carryover_messages: int = 4
    local_chunk_tokens: int = 3500
    summary_max_tokens: int = 512
    max_attempts: int = 3
    retry_base_seconds: float = 5.0

    def __post_init__(self) -> None:
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
                'extraction_prompt': MEMORY_EXTRACTION_INSTRUCTIONS}

    @staticmethod
    def _chat_messages(messages: Sequence[MessageRecord]) -> list[dict[str, str]]:
        return [m.to_chat_message() for m in messages]

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
            size = sum(self.memory_token_counter.count_text(m.content) + 4 for m in turn)
            if chunk and tokens + size > limit:
                chunks.append(chunk)
                chunk, tokens = [], 0
            chunk.extend(turn)
            tokens += size
        if chunk:
            chunks.append(chunk)
        return chunks

    def submit(self, session_id: str, *, reason: str) -> ConsolidationSubmission:
        return self.store.submit(session_id, reason=reason, settings=self._settings(),
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
            payload = json.loads(job.payload_json)
            settings = payload['settings']
            if settings['provider'] != self.memory_provider_name or settings['summary_model'] != self.summary_provider.config.model:
                self.store.fail_job(job.job_id, owner, 'Provider/model differs from submitted job; restore configuration before retry.',
                                    retry_delay=0, retryable=False)
                return
            messages = [MessageRecord(**m) for m in payload['messages']]
            session = self.store.get_session(job.session_id)
            if session.summary_status not in ('success', 'fallback'):
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
            chunks = (self._local_chunks(messages, token_limit=settings['local_chunk_tokens'])
                      if settings['provider'] == 'local' else [messages])
            self.store.ensure_chunks(job.job_id, owner, [self._chat_messages(c) for c in chunks])
            persisted = self.store.chunks(job.job_id)
            for chunk in persisted:
                self.store.assert_current(job.job_id, owner)
                if chunk['status'] == 'succeeded':
                    continue
                if chunk['status'] != 'pending':
                    raise MemoryWriteUncertainError('chunk write outcome unknown; explicit replay required')
                self.store.set_chunk(job.job_id, owner, chunk['chunk_id'], status='writing')
                metadata = {'source': 'l1_session', 'session_id': job.session_id,
                            'consolidation_id': chunk['chunk_id'], 'consolidation_attempt': job.attempts,
                            'chunk_index': chunk['chunk_index'], 'chunk_count': len(persisted),
                            'job_id': job.job_id, 'memory_generation': job.generation}
                try:
                    events = self.long_term.add_session(user_id=job.user_id, agent_id=job.agent_id,
                                                       messages=json.loads(chunk['messages_json']), metadata=metadata,
                                                       prompt=settings['extraction_prompt'])
                except RetryableMemoryError as exc:
                    self.store.set_chunk(job.job_id, owner, chunk['chunk_id'], status='pending', error=str(exc))
                    raise
                except Exception as exc:
                    self.store.set_chunk(job.job_id, owner, chunk['chunk_id'], status='uncertain', error=str(exc))
                    raise MemoryWriteUncertainError(f'Write outcome unknown: {exc}') from exc
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
