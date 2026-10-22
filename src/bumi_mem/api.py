# 本次改动（2026-10-21）：将注册用户与共同事实的 HTTP 路由安装到现有 FastAPI 应用。
"""HTTP 边界层：外部 Agent 的 REST API。模型/凭据仅在 lifespan 中加载。"""

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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bumi_mem.identity import require_id
from bumi_mem.event_time import message_time
from bumi_mem.turn_prompt import MAX_CLIENT_TURN_ID
from bumi_mem.schemas import IdentityRequiredError
from bumi_mem.service_settings import ServiceSettings, build_service_engine
from bumi_mem.turn_store import TurnConflict, TurnNotFound
from bumi_mem.agent_protocol import (
    AgentPrepareRequest, AgentCompleteRequest, AgentEventRequest, AgentPrefetchRequest,
    AgentPrepareResponse, AgentCompleteResponse, AgentEventResponse,
    AgentPendingRequest, AgentPendingResponse, AgentPrefetchResponse,
)

logger = logging.getLogger(__name__)


class TurnContextRequest(BaseModel):
    """每轮一次请求：上一轮确认回复 + 当前问题。

    turn_id 是调用方对本轮请求的唯一标识，网络重试时保持不变。
    query=None 时仅结算上一轮回复，不开始新轮次。
    """
    model_config = ConfigDict(extra='forbid')
    turn_id: str = Field(min_length=1, max_length=MAX_CLIENT_TURN_ID,
                         description="Caller's unique ID for THIS request; keep it across network retries")
    user_id: str = Field(min_length=1, max_length=128,
                         description='Speaker of query only, never of the reply; __guest__ when unconfirmed')
    query: str | None = Field(max_length=16000,
                              description='Current user text; null settles the last reply without a new turn')
    previous_ai_reply: str | None = Field(default=None, max_length=64000,
                                          description='Previously CONFIRMED robot output, not an unspoken draft')
    query_occurred_at: str | None = Field(default=None, max_length=64)
    previous_reply_occurred_at: str | None = Field(default=None, max_length=64)

    @field_validator('query_occurred_at', 'previous_reply_occurred_at')
    @classmethod
    def validate_time(cls, value):
        return message_time(value) if value is not None else None

    @field_validator('turn_id', 'user_id')
    @classmethod
    def validate_ids(cls, value):
        try:
            return require_id(value)
        except IdentityRequiredError:
            raise ValueError('must be an explicit stable printable ID') from None

    @field_validator('query')
    @classmethod
    def validate_query(cls, value):
        # 空或空白查询不是仅结算请求；必须为 null
        if value is not None and not value.strip():
            raise ValueError('query must be null or nonblank')
        return value

    @model_validator(mode='after')
    def validate_time_pairing(self):
        if self.query is None and self.query_occurred_at is not None:
            raise ValueError('query timestamp requires a query')
        if not (self.previous_ai_reply or '').strip() and self.previous_reply_occurred_at is not None:
            raise ValueError('reply timestamp requires a nonempty reply')
        return self


class LongTermMemoryItem(BaseModel):
    """长期记忆条目：用户 ID、内容和时间标签列表。"""
    model_config = ConfigDict(extra='forbid')
    user_id: str
    content: str
    time: list[str]


class ShortTermMemoryItem(BaseModel):
    """短期记忆条目：角色、说话人 ID、内容和时间。"""
    model_config = ConfigDict(extra='forbid')
    role: Literal['user', 'assistant', 'system']
    speaker_id: str
    content: str
    time: str


class TurnContextResponse(BaseModel):
    """轮次上下文响应：返回当前用户、长期记忆和短期记忆。"""
    model_config = ConfigDict(extra='forbid')
    turn_id: str
    current_user: str
    long_term_memory: list[LongTermMemoryItem]
    short_term_memory: list[ShortTermMemoryItem]


class ContextRequest(BaseModel):
    """旧版上下文请求：仅为兼容已有调用方保留，返回旧版字段。"""
    model_config = ConfigDict(extra='forbid')
    user_id: str = Field(min_length=1, max_length=128)
    previous_ai_reply: str | None = Field(default=None, max_length=64000,
                                         description='Previous agent output; null/empty means no output')
    query: str | None = Field(max_length=16000,
                              description='Current user query; null flushes previous output without a new turn')

    query_occurred_at: str | None = Field(default=None, max_length=64)
    previous_reply_occurred_at: str | None = Field(default=None, max_length=64)

    @field_validator('query_occurred_at', 'previous_reply_occurred_at')
    @classmethod
    def validate_time(cls, value):
        return message_time(value) if value is not None else None

    @model_validator(mode='after')
    def validate_time_pairing(self):
        if self.query is None and self.query_occurred_at is not None:
            raise ValueError('query timestamp requires a query')
        if not (self.previous_ai_reply or '').strip() and self.previous_reply_occurred_at is not None:
            raise ValueError('reply timestamp requires a nonempty reply')
        return self

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
    """聊天消息：角色和内容。"""
    role: Literal['system', 'user', 'assistant']
    content: str


class ContextResponse(BaseModel):
    """旧版上下文响应：包含消息列表、检索状态和整理状态。"""
    core_write: dict = Field(default_factory=lambda: {'status': 'not_requested'})
    request_id: str
    user_id: str
    agent_id: str
    messages: list[ChatMessage]
    has_prompt: bool
    previous_reply_saved: bool
    recall_status: Literal['ok', 'unavailable', 'deleting', 'disabled', 'not_requested']
    consolidation_status: Literal['not_triggered', 'queued', 'succeeded']


class RequestBodyLimit:
    """请求体大小限制中间件：在验证前限制 JSON 缓冲区大小（含分块请求）。

    默认限制 1MB，仅对 /v1/memory/ 路径生效，超限返回 413。
    """
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
    """创建 FastAPI 应用实例。

    参数:
        settings:       服务配置（None 时从环境变量读取）
        engine_factory: 引擎工厂函数（用于依赖注入测试）

    返回:
        配置好的 FastAPI 应用。

    副作用:
        - lifespan 中创建数据目录、获取文件锁（防多进程启动）、初始化引擎
        - 注册请求体限制中间件和 Bearer 认证
    """
    @asynccontextmanager
    async def lifespan(app):
        config = settings if settings is not None else ServiceSettings.from_env()
        config.data_dir.mkdir(parents=True, exist_ok=True)
        # 快速失败：防止多 worker/多进程意外启动同一数据存储
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

    app = FastAPI(title='Bumi Agent Memory API', version='1.4.0', lifespan=lifespan)
    app.state.ready = False
    app.add_middleware(RequestBodyLimit)
    bearer = HTTPBearer(auto_error=False)

    def authorize(request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        """Bearer 认证：验证 API Key 并检查服务就绪状态。"""
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
        # 不回显查询文本、凭据或内部 Python 异常，仅返回字段位置和类型
        return JSONResponse(status_code=422, content={'detail': {'code': 'invalid_request', 'fields': [
            {'loc': list(e['loc']), 'type': e['type']} for e in exc.errors()]}})

    @app.exception_handler(Exception)
    async def unexpected(_, exc):
        logger.error('Memory request failed (%s)', type(exc).__name__)
        return JSONResponse(status_code=503, content={'detail': {'code': 'memory_service_error'}})

    @app.get('/health/live', tags=['health'])
    def live():
        """存活探针：始终返回 alive。"""
        return {'status': 'alive'}

    @app.get('/health/ready', tags=['health'])
    def ready(request: Request):
        """就绪探针：引擎初始化完成后返回 ready，否则 503。"""
        if not request.app.state.ready:
            raise HTTPException(503, detail={'code': 'not_ready'})
        engine = request.app.state.engine
        return {'status': 'ready', 'warmed_up': engine._warmup_elapsed_ms is not None,
                'long_term_status': engine.long_term_status}

    @app.get('/v1/memory/debug', dependencies=[Depends(authorize)], tags=['debug'])
    def graph_debug(user_id: str, request: Request):
        require_id(user_id, 'user_id')
        engine = request.app.state.engine
        if not hasattr(engine.long_term, 'debug'):
            raise HTTPException(404, detail={'code': 'graph_debug_unavailable'})
        return engine.long_term.debug(user_id=user_id, agent_id=request.app.state.settings.agent_id)

    from bumi_mem.admin_api import install_admin
    install_admin(app, authorize)
    from bumi_mem.control_api import install_control
    install_control(app)
    from bumi_mem.isolation_api import install_isolation
    install_isolation(app,authorize)

    @app.post('/v1/memory/context', response_model=TurnContextResponse | ContextResponse | AgentPrefetchResponse | AgentPrepareResponse | AgentCompleteResponse | AgentEventResponse | AgentPendingResponse, dependencies=[Depends(authorize)], tags=['memory'])
    def context(body: TurnContextRequest | ContextRequest | AgentPrefetchRequest | AgentPrepareRequest | AgentCompleteRequest | AgentEventRequest | AgentPendingRequest, request: Request,
                idempotency_key: Annotated[str | None, Header(alias='Idempotency-Key', max_length=128)] = None):
        """核心记忆上下文端点：处理轮次请求、Agent 协议请求和旧版上下文请求。

        根据 body 类型分派到不同的处理路径：
        - TurnContextRequest: 新版轮次上下文（每轮一次请求）
        - AgentPendingRequest: Agent 待处理查询
        - AgentPrefetchRequest: Agent 预取
        - AgentPrepareRequest/AgentCompleteRequest: Agent 准备/完成
        - AgentEventRequest: Agent 事件记录
        - ContextRequest: 旧版上下文请求
        """
        config = request.app.state.settings
        engine = request.app.state.engine
        # 主路径：每轮一次请求，由调用方的 turn_id 标识
        if isinstance(body, TurnContextRequest):
            return TurnContextResponse.model_validate(engine.process_turn_context(
                turn_id=body.turn_id, user_id=body.user_id, agent_id=config.agent_id,
                query=body.query, previous_ai_reply=body.previous_ai_reply,
                query_occurred_at=body.query_occurred_at,
                previous_reply_occurred_at=body.previous_reply_occurred_at))
        if isinstance(body, AgentPendingRequest):
            return dict(operation='pending', agent_id=config.agent_id, has_prompt=False,
                        pending=engine.store.pending_agent_turn(config.agent_id))
        if isinstance(body, AgentPrefetchRequest):
            return engine.prefetch_agent(request=body, agent_id=config.agent_id)
        if isinstance(body, AgentPrepareRequest):
            return engine.prepare_agent(request=body, agent_id=config.agent_id)
        if isinstance(body, AgentCompleteRequest):
            return engine.complete_agent(request=body, agent_id=config.agent_id)
        if isinstance(body, AgentEventRequest):
            return engine.record_agent_event(request=body, agent_id=config.agent_id)
        return engine.process_context(
            user_id=body.user_id, agent_id=config.agent_id, query=body.query,
            previous_ai_reply=body.previous_ai_reply, idempotency_key=idempotency_key,
            query_occurred_at=body.query_occurred_at, previous_reply_occurred_at=body.previous_reply_occurred_at)

    return app


def main():
    """启动 HTTP 服务：使用 uvicorn 运行 create_app 工厂。"""
    import uvicorn
    uvicorn.run('bumi_mem.api:create_app', factory=True, host=os.getenv('BUMI_HOST', '127.0.0.1'),
                port=int(os.getenv('BUMI_PORT', '8088')), workers=1, access_log=False)


if __name__ == '__main__':
    main()