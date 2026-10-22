# 本次改动（2026-10-21）：文字控制错误码补充 registered_identity_required，不将未注册身份视为可控长期用户。
"""Versioned fixed-principal text pilot; admin credentials are never accepted."""
import os
import secrets
from pathlib import Path
from typing import Literal
from fastapi import Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from bumi_mem.identity import require_id
from bumi_mem.turn_store import TurnConflict
from bumi_mem.memory_control import MemoryControl

class ControlReply(BaseModel):
    model_config = ConfigDict(extra='forbid')
    operation_id: str = Field(min_length=1,max_length=128)
    memory_id: str = Field(min_length=1,max_length=128)
    version: int = Field(ge=0)
    confirm_source_scope: Literal[True]

class ControlContext(BaseModel):
    model_config = ConfigDict(extra='forbid')
    turn_id: str = Field(min_length=1,max_length=120)
    user_id: str = Field(min_length=1,max_length=128)
    query: str | None = Field(default=None,max_length=16000)
    previous_ai_reply: str | None = Field(default=None,max_length=64000)
    memory_control_reply: ControlReply | None = None
    query_occurred_at: str | None = Field(default=None,max_length=64)
    previous_reply_occurred_at: str | None = Field(default=None,max_length=64)

    @field_validator('turn_id','user_id')
    @classmethod
    def identifiers(cls,value):
        return require_id(value)

    @field_validator('query')
    @classmethod
    def nonempty_query(cls,value):
        if value is not None and not value.strip():
            raise ValueError('query must be null or nonblank')
        return value

def install_control(app):
    def principal(request: Request, authorization: str | None = Header(default=None)):
        key = os.getenv('BUMI_CONTROL_API_KEY','')
        user = os.getenv('BUMI_CONTROL_USER_ID','')
        admin = request.app.state.settings.api_key
        if len(key)<32 or secrets.compare_digest(key,admin) or not user:
            raise HTTPException(503,detail={'code':'control_pilot_not_configured'})
        if not authorization or not secrets.compare_digest(authorization,'Bearer '+key):
            raise HTTPException(401,detail={'code':'invalid_control_key'})
        try: require_id(user,'user_id')
        except ValueError: raise HTTPException(503,detail={'code':'invalid_control_principal'}) from None
        return user

    def invoke(fn):
        try: return fn()
        except TurnConflict:
            raise HTTPException(409,detail={'code':'context_conflict'}) from None
        except ValueError as exc:
            codes = {'request_conflict','operation_not_found','selection_expired','candidate_changed',
                     'source_lineage_unavailable','source_scope_confirmation_required','graphiti_required','registered_identity_required'}
            raise HTTPException(409,detail={'code':str(exc) if str(exc) in codes else 'invalid_control_request'}) from None
        except Exception:
            raise HTTPException(503,detail={'code':'control_unavailable','message':'请用相同请求 ID 重试或查询状态；不要假定操作成功。'}) from None

    @app.get('/memory-control',response_class=HTMLResponse,include_in_schema=False)
    def page(): return HTMLResponse(Path(__file__).with_name('control.html').read_text())

    @app.post('/v2/memory/context',tags=['memory-control'])
    def context(body: ControlContext,request: Request,user=Depends(principal)):
        if body.user_id!=user: raise HTTPException(403,detail={'code':'principal_mismatch'})
        if body.memory_control_reply and body.query: raise HTTPException(422,detail={'code':'selection_requires_null_query'})
        controller = MemoryControl(request.app.state.engine)
        return invoke(lambda:controller.context(user_id=user,agent_id=request.app.state.settings.agent_id,body=body.model_dump(mode='json')))

    @app.get('/v2/memory/operations/{operation_id}',tags=['memory-control'])
    def operation(operation_id: str,request: Request,user=Depends(principal)):
        result = invoke(lambda:MemoryControl(request.app.state.engine).get(user,request.app.state.settings.agent_id,operation_id))
        if result is None: raise HTTPException(404,detail={'code':'operation_not_found'})
        return result
