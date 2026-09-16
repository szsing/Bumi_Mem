"""HTTP boundary for external Agents. Models/credentials are loaded in lifespan only."""

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bumi_mem.identity import require_id
from bumi_mem.schemas import IdentityRequiredError
from bumi_mem.service_settings import ServiceSettings, build_service_engine
from bumi_mem.turn_store import TurnConflict, TurnNotFound

logger = logging.getLogger(__name__)


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    user_id: str = Field(min_length=1, max_length=128)
    previous_ai_reply: str | None = Field(default=None, max_length=64000,
                                         description='Previous agent output; null/empty means no output')
    query: str | None = Field(max_length=16000,
                              description='Current user query; null flushes previous output without a new turn')

    @field_validator('user_id')
    @classmethod
    def validate_user(cls, value):
        try:
            return require_id(value)
        except IdentityRequiredError:
            raise ValueError('user_id must be an explicit stable ID') from None

    @field_validator('query')
    @classmethod
    def validate_query(cls, value):
        if value is not None and not value.strip():
            raise ValueError('query must be null or nonblank')
        return value


class ChatMessage(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str


class ContextResponse(BaseModel):
    request_id: str
    user_id: str
    agent_id: str
    messages: list[ChatMessage]
    has_prompt: bool
    previous_reply_saved: bool
    recall_status: Literal['ok', 'unavailable', 'deleting', 'disabled', 'not_requested']
    consolidation_status: Literal['not_triggered', 'queued', 'succeeded']


class RequestBodyLimit:
    """Bound JSON buffering, including chunked requests, before validation."""
    def __init__(self, app, limit=1024 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope.get('path', '').startswith('/v1/memory/'):
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event['type'] == 'http.disconnect':
                return
            body.extend(event.get('body', b''))
            if len(body) > self.limit:
                response = JSONResponse(status_code=413, content={'detail': {'code': 'request_too_large'}})
                return await response(scope, receive, send)
            if not event.get('more_body', False):
                break
        sent = False
        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()
        return await self.app(scope, replay, send)


def create_app(settings: ServiceSettings | None = None, *, engine_factory=build_service_engine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        config = settings if settings is not None else ServiceSettings.from_env()
        config.data_dir.mkdir(parents=True, exist_ok=True)
        # Fail fast on accidental multi-worker/multi-process startup against this store.
        lock = FileLock(str(config.db_path) + '.service.lock')
        try:
            lock.acquire(timeout=0)
        except Timeout:
            raise RuntimeError('another Bumi HTTP process already owns this data directory; use one worker') from None
        engine = None
        app.state.ready = False
        try:
            app.state.settings = config
            engine = await asyncio.to_thread(engine_factory, config)
            app.state.engine = engine
            await asyncio.to_thread(engine.start)
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            try:
                if engine is not None:
                    await asyncio.to_thread(engine.close)
            finally:
                lock.release()

    app = FastAPI(title='Bumi Agent Memory API', version='1.0.0', lifespan=lifespan)
    app.state.ready = False
    app.add_middleware(RequestBodyLimit)
    bearer = HTTPBearer(auto_error=False)

    def authorize(request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        config = getattr(request.app.state, 'settings', None)
        if (config is None or credentials is None or credentials.scheme.lower() != 'bearer'
                or not secrets.compare_digest(credentials.credentials.encode(), config.api_key.encode())):
            raise HTTPException(401, detail={'code': 'unauthorized'}, headers={'WWW-Authenticate': 'Bearer'})
        if not request.app.state.ready:
            raise HTTPException(503, detail={'code': 'not_ready'})

    @app.exception_handler(TurnConflict)
    async def conflict(_, exc):
        return JSONResponse(status_code=409, content={'detail': {'code': exc.code}})

    @app.exception_handler(TurnNotFound)
    async def not_found(_, exc):
        return JSONResponse(status_code=404, content={'detail': {'code': 'turn_not_found'}})

    @app.exception_handler(IdentityRequiredError)
    async def bad_id(_, exc):
        return JSONResponse(status_code=422, content={'detail': {'code': 'invalid_identity'}})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_, exc):
        # Do not echo query text, credentials, or internal Python exceptions in validation errors.
        return JSONResponse(status_code=422, content={'detail': {'code': 'invalid_request', 'fields': [
            {'loc': list(e['loc']), 'type': e['type']} for e in exc.errors()]}})

    @app.exception_handler(Exception)
    async def unexpected(_, exc):
        logger.error('Memory request failed (%s)', type(exc).__name__)
        return JSONResponse(status_code=503, content={'detail': {'code': 'memory_service_error'}})

    @app.get('/health/live', tags=['health'])
    def live():
        return {'status': 'alive'}

    @app.get('/health/ready', tags=['health'])
    def ready(request: Request):
        if not request.app.state.ready:
            raise HTTPException(503, detail={'code': 'not_ready'})
        return {'status': 'ready', 'warmed_up': request.app.state.engine._warmup_elapsed_ms is not None}

    @app.post('/v1/memory/context', response_model=ContextResponse, dependencies=[Depends(authorize)], tags=['memory'])
    def context(body: ContextRequest, request: Request,
                idempotency_key: Annotated[str | None, Header(alias='Idempotency-Key', max_length=128)] = None):
        config = request.app.state.settings
        return request.app.state.engine.process_context(
            user_id=body.user_id, agent_id=config.agent_id, query=body.query,
            previous_ai_reply=body.previous_ai_reply, idempotency_key=idempotency_key)

    return app


def main():
    import uvicorn
    uvicorn.run('bumi_mem.api:create_app', factory=True, host=os.getenv('BUMI_HOST', '127.0.0.1'),
                port=int(os.getenv('BUMI_PORT', '8088')), workers=1, access_log=False)


if __name__ == '__main__':
    main()
