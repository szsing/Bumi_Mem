"""Single-endpoint memory client; no model calls and no local conversation inference."""
from __future__ import annotations

import argparse
import json
import os
from uuid import uuid4

import httpx


class BumiMemoryClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 10.0,
                 transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=base_url.rstrip('/'),
                                  headers={'Authorization': 'Bearer ' + api_key},
                                  timeout=timeout, transport=transport)

    def get_context(self, user_id: str, query: str | None, *, previous_ai_reply: str | None = None,
                    idempotency_key: str | None = None, query_occurred_at: str | None = None,
                    previous_reply_occurred_at: str | None = None) -> dict:
        """Supply the same key/body on retry. query=None only flushes the last AI reply."""
        body = {'user_id': user_id, 'previous_ai_reply': previous_ai_reply, 'query': query}
        if query_occurred_at is not None:
            body['query_occurred_at'] = query_occurred_at
        if previous_reply_occurred_at is not None:
            body['previous_reply_occurred_at'] = previous_reply_occurred_at
        result = self._http.post('/v1/memory/context',
                                 headers={'Idempotency-Key': idempotency_key or uuid4().hex},
                                 json=body)
        result.raise_for_status()
        return result.json()

    def agent_request(self, payload: dict) -> dict:
        """Send prepare/complete/agent_event; persist payload unchanged before retry."""
        if payload.get('operation') not in {'prepare', 'complete', 'agent_event', 'pending'}:
            raise ValueError('expected an explicit Agent operation')
        result = self._http.post('/v1/memory/context', json=payload)
        result.raise_for_status()
        return result.json()

    def prepare_turn(self, *, client_turn_id: str, query: str, user_id: str | None = None,
                     speaker: dict | None = None, occurred_at: str | None = None) -> dict:
        return self.agent_request(_prepare_payload(client_turn_id, query, user_id, speaker, occurred_at))

    def complete_turn(self, *, turn_id: str, status: str, reply: str = '',
                      speech: dict | None = None, occurred_at: str | None = None) -> dict:
        return self.agent_request(_complete_payload(turn_id, status, reply, speech, occurred_at))

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncBumiMemoryClient:
    """Async version for Pipecat/asyncio; same single endpoint."""
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 10.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._http = httpx.AsyncClient(base_url=base_url.rstrip('/'),
                                       headers={'Authorization': 'Bearer ' + api_key},
                                       timeout=timeout, transport=transport)

    async def get_context(self, user_id: str, query: str | None, *, previous_ai_reply: str | None = None,
                          idempotency_key: str | None = None, query_occurred_at: str | None = None,
                    previous_reply_occurred_at: str | None = None) -> dict:
        body = {'user_id': user_id, 'previous_ai_reply': previous_ai_reply, 'query': query}
        if query_occurred_at is not None:
            body['query_occurred_at'] = query_occurred_at
        if previous_reply_occurred_at is not None:
            body['previous_reply_occurred_at'] = previous_reply_occurred_at
        result = await self._http.post('/v1/memory/context',
                                       headers={'Idempotency-Key': idempotency_key or uuid4().hex},
                                       json=body)
        result.raise_for_status()
        return result.json()

    async def agent_request(self, payload: dict) -> dict:
        """Never retry a 409 as a new operation or silently reassign a turn."""
        if payload.get('operation') not in {'prepare', 'complete', 'agent_event', 'pending'}:
            raise ValueError('expected an explicit Agent operation')
        result = await self._http.post('/v1/memory/context', json=payload)
        result.raise_for_status()
        return result.json()

    async def prepare_turn(self, *, client_turn_id: str, query: str, user_id: str | None = None,
                           speaker: dict | None = None, occurred_at: str | None = None) -> dict:
        return await self.agent_request(_prepare_payload(client_turn_id, query, user_id, speaker, occurred_at))

    async def complete_turn(self, *, turn_id: str, status: str, reply: str = '',
                            speech: dict | None = None, occurred_at: str | None = None) -> dict:
        return await self.agent_request(_complete_payload(turn_id, status, reply, speech, occurred_at))

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def _prepare_payload(client_turn_id, query, user_id, speaker, occurred_at):
    body = {'operation': 'prepare', 'client_turn_id': client_turn_id, 'query': query}
    body.update({k: v for k, v in {'user_id': user_id, 'speaker': speaker, 'occurred_at': occurred_at}.items() if v is not None})
    return body


def _complete_payload(turn_id, status, reply, speech, occurred_at):
    body = {'operation': 'complete', 'turn_id': turn_id, 'status': status, 'reply': reply}
    body.update({k: v for k, v in {'speech': speech, 'occurred_at': occurred_at}.items() if v is not None})
    return body


def main():
    parser = argparse.ArgumentParser(description='Single-context API client; never generates an AI reply')
    parser.add_argument('--user-id', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--query')
    mode.add_argument('--flush', action='store_true', help='Flush previous AI output without a new query')
    parser.add_argument('--previous-ai-reply', default=None)
    parser.add_argument('--idempotency-key', default=None)
    args = parser.parse_args()
    with BumiMemoryClient(os.environ['BUMI_SERVICE_URL'], os.environ['BUMI_SERVICE_API_KEY']) as client:
        result = client.get_context(args.user_id, None if args.flush else args.query,
                                    previous_ai_reply=args.previous_ai_reply,
                                    idempotency_key=args.idempotency_key)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
