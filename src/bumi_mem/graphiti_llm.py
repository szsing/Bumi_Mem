"""Schema validation inside Graphiti's existing bounded LLM retry boundary.

No prompt changes, result repair, fallback values or graph-write retries. Only
the individual generation is retried, using upstream's four-attempt policy.
"""
from __future__ import annotations

import json
import contextvars
import os
import time
import uuid
from pathlib import Path

from pydantic import ValidationError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient


class StructuredResponseError(json.JSONDecodeError):
    """Valid JSON with an invalid response shape; upstream retries JSON errors."""


class ValidatedGraphitiClient(OpenAIGenericClient):
    def __init__(self, *args, diagnostic_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_dir = Path(diagnostic_dir)
        self._active_record = contextvars.ContextVar('graphiti_llm_record', default=None)
        create = self.client.chat.completions.create
        async def capture(**request):
            response = await create(**request)
            record = self._active_record.get()
            if record is not None:
                record['request'] = {k: request[k] for k in
                    ('model','messages','temperature','max_tokens','response_format') if k in request}
                if hasattr(response, 'model_dump'):
                    record['raw_response'] = response.model_dump(mode='json')
                else:
                    record['raw_content'] = response.choices[0].message.content
            return response
        self.client.chat.completions.create = capture

    def _record(self, record):
        # Individual private files avoid interleaving concurrent model responses.
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.diagnostic_dir / (uuid.uuid4().hex + '.json')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, default=str)
            f.write('\n')

    async def _generate_response(self, messages, response_model=None, *args, **kwargs):
        started = time.perf_counter()
        record = {'response_model': getattr(response_model, '__name__', None),
                  'started_at': time.time()}
        token = self._active_record.set(record)
        try:
            result = await super()._generate_response(messages, response_model, *args, **kwargs)
            record['parsed_response'] = result
            if response_model is not None:
                try:
                    response_model.model_validate(result)
                except ValidationError as exc:
                    record['validation_errors'] = exc.errors(include_url=False)
                    # Reuse upstream's existing tenacity boundary. Do not add an
                    # outer retry around generate_response or add_episode_bulk.
                    raise StructuredResponseError(
                        f'LLM JSON does not match {response_model.__name__}',
                        json.dumps(result, ensure_ascii=False), 0) from exc
            record['status'] = 'ok'
            return result
        except BaseException as exc:
            record['status'] = 'failed'
            record['error_type'] = type(exc).__name__
            # Only validation/JSON payloads are retained; vendor exception strings
            # can contain credentials/headers and are deliberately not serialized.
            if isinstance(exc, json.JSONDecodeError):
                record['invalid_json_document'] = exc.doc
            raise
        finally:
            self._active_record.reset(token)
            record['elapsed_ms'] = (time.perf_counter() - started) * 1000
            self._record(record)
