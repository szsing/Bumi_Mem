"""Single-endpoint memory client; no model calls and no local conversation inference."""
from __future__ import annotations

import argparse
import json
import os
from uuid import uuid4

import httpx


class BumiMemoryClient:
    """单端点记忆客户端：不执行模型调用，不进行本地对话推断。

    使用方式：
    - turn_context()：主路径，每轮一次请求，返回四个结构化字段
    - get_context()：旧版接口，保持向后兼容
    - agent_request()：发送 prepare/complete/agent_event 操作
    - prepare_turn()/prefetch_turn()/complete_turn()：Agent 协议的便捷方法
    """
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 10.0,
                 transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=base_url.rstrip('/'),
                                  headers={'Authorization': 'Bearer ' + api_key},
                                  timeout=timeout, transport=transport)

    def turn_context(self, *, turn_id: str, user_id: str, query: str | None,
                     previous_ai_reply: str | None = None, query_occurred_at: str | None = None,
                     previous_reply_occurred_at: str | None = None) -> dict:
        """Main path: one request per turn, returning four structured fields.

        Reuse the SAME turn_id and body on a network retry; use a NEW turn_id for
        the next turn. query=None settles the last confirmed reply and returns an
        empty memory arrays. Consume result['long_term_memory'] and result['short_term_memory'].
        """
        result = self._http.post('/v1/memory/context', json=_turn_payload(
            turn_id, user_id, query, previous_ai_reply, query_occurred_at, previous_reply_occurred_at))
        result.raise_for_status()
        return result.json()

    def get_context(self, user_id: str, query: str | None, *, previous_ai_reply: str | None = None,
                    idempotency_key: str | None = None, query_occurred_at: str | None = None,
                    previous_reply_occurred_at: str | None = None) -> dict:
        """Legacy exchange kept for existing callers; prefer turn_context().

        Supply the same key/body on retry. query=None only flushes the last AI reply.
        """
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
        if payload.get('operation') not in {'prefetch', 'prepare', 'complete', 'agent_event', 'pending'}:
            raise ValueError('expected an explicit Agent operation')
        result = self._http.post('/v1/memory/context', json=payload)
        result.raise_for_status()
        return result.json()

    def prepare_turn(self, *, client_turn_id: str, query: str, user_id: str | None = None,
                     speaker: dict | None = None, occurred_at: str | None = None,
                     prefetch_id: str | None = None) -> dict:
        return self.agent_request(_prepare_payload(client_turn_id, query, user_id, speaker,
                                                   occurred_at, prefetch_id))

    def prefetch_turn(self, *, client_turn_id: str, query_prefix: str, user_id: str | None = None,
                      speaker: dict | None = None, prefix_revision: int | None = None) -> dict:
        """在稳定的 ASR 前缀上提前发起检索；把返回的 prefetch_id 交给 prepare_turn。"""
        return self.agent_request(_prefetch_payload(client_turn_id, query_prefix, user_id,
                                                    speaker, prefix_revision))

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

    async def turn_context(self, *, turn_id: str, user_id: str, query: str | None,
                           previous_ai_reply: str | None = None, query_occurred_at: str | None = None,
                           previous_reply_occurred_at: str | None = None) -> dict:
        """Main path: one request per turn, returning four structured fields."""
        result = await self._http.post('/v1/memory/context', json=_turn_payload(
            turn_id, user_id, query, previous_ai_reply, query_occurred_at, previous_reply_occurred_at))
        result.raise_for_status()
        return result.json()

    async def get_context(self, user_id: str, query: str | None, *, previous_ai_reply: str | None = None,
                          idempotency_key: str | None = None, query_occurred_at: str | None = None,
                    previous_reply_occurred_at: str | None = None) -> dict:
        """Legacy exchange kept for existing callers; prefer turn_context()."""
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
        if payload.get('operation') not in {'prefetch', 'prepare', 'complete', 'agent_event', 'pending'}:
            raise ValueError('expected an explicit Agent operation')
        result = await self._http.post('/v1/memory/context', json=payload)
        result.raise_for_status()
        return result.json()

    async def prepare_turn(self, *, client_turn_id: str, query: str, user_id: str | None = None,
                           speaker: dict | None = None, occurred_at: str | None = None,
                           prefetch_id: str | None = None) -> dict:
        return await self.agent_request(_prepare_payload(client_turn_id, query, user_id, speaker,
                                                         occurred_at, prefetch_id))

    async def prefetch_turn(self, *, client_turn_id: str, query_prefix: str, user_id: str | None = None,
                            speaker: dict | None = None, prefix_revision: int | None = None) -> dict:
        return await self.agent_request(_prefetch_payload(client_turn_id, query_prefix, user_id,
                                                          speaker, prefix_revision))

    async def complete_turn(self, *, turn_id: str, status: str, reply: str = '',
                            speech: dict | None = None, occurred_at: str | None = None) -> dict:
        return await self.agent_request(_complete_payload(turn_id, status, reply, speech, occurred_at))

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def _turn_payload(turn_id, user_id, query, previous_ai_reply, query_occurred_at, previous_reply_occurred_at):
    body = {'turn_id': turn_id, 'user_id': user_id, 'query': query}
    body.update({k: v for k, v in {'previous_ai_reply': previous_ai_reply,
                                   'query_occurred_at': query_occurred_at,
                                   'previous_reply_occurred_at': previous_reply_occurred_at}.items()
                 if v is not None})
    return body


def _prepare_payload(client_turn_id, query, user_id, speaker, occurred_at, prefetch_id=None):
    body = {'operation': 'prepare', 'client_turn_id': client_turn_id, 'query': query}
    body.update({k: v for k, v in {'user_id': user_id, 'speaker': speaker, 'occurred_at': occurred_at,
                                   'prefetch_id': prefetch_id}.items() if v is not None})
    return body


def _prefetch_payload(client_turn_id, query_prefix, user_id, speaker, prefix_revision):
    body = {'operation': 'prefetch', 'client_turn_id': client_turn_id, 'query_prefix': query_prefix}
    body.update({k: v for k, v in {'user_id': user_id, 'speaker': speaker,
                                   'prefix_revision': prefix_revision}.items() if v is not None})
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
    mode.add_argument('--flush', action='store_true', help='Settle previous AI output without a new query')
    parser.add_argument('--previous-ai-reply', default=None)
    parser.add_argument('--turn-id', default=None, help='Unique ID for this request; new ID per turn')
    parser.add_argument('--query-occurred-at', default=None)
    parser.add_argument('--previous-reply-occurred-at', default=None)
    parser.add_argument('--legacy', action='store_true',
                        help='Use the old multi-context exchange instead of the structured single-turn response')
    parser.add_argument('--idempotency-key', default=None, help='Legacy exchange only')
    args = parser.parse_args()
    with BumiMemoryClient(os.environ['BUMI_SERVICE_URL'], os.environ['BUMI_SERVICE_API_KEY']) as client:
        query = None if args.flush else args.query
        if args.legacy:
            result = client.get_context(args.user_id, query, previous_ai_reply=args.previous_ai_reply,
                                        idempotency_key=args.idempotency_key,
                                        query_occurred_at=args.query_occurred_at,
                                        previous_reply_occurred_at=args.previous_reply_occurred_at)
        else:
            result = client.turn_context(turn_id=args.turn_id or uuid4().hex, user_id=args.user_id,
                                         query=query, previous_ai_reply=args.previous_ai_reply,
                                         query_occurred_at=args.query_occurred_at,
                                         previous_reply_occurred_at=args.previous_reply_occurred_at)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
