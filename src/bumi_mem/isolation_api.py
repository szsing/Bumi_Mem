# 本次改动（2026-10-21）：新增注册表、共同事实与管理员操作 API，检查注册、来源范围确认及操作权限。
"""Trusted Agent identity; registration management remains administrator-only."""
from pathlib import Path
from typing import Literal
from fastapi import Depends, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from bumi_mem.identity import require_id, is_guest

class Registration(BaseModel):
    model_config=ConfigDict(extra='forbid')
    user_id: str=Field(min_length=1,max_length=128)
    display_name: str=Field(min_length=1,max_length=128)
    aliases: list[str]=Field(default_factory=list,max_length=20)
    active: bool=True
    @field_validator('user_id')
    @classmethod
    def identity(cls,value):
        try:return require_id(value)
        except Exception:raise ValueError('invalid user ID') from None

class SharedChange(BaseModel):
    model_config=ConfigDict(extra='forbid')
    user_id: str=Field(min_length=1,max_length=128)
    version: int=Field(ge=0)
    action: Literal['withdraw','edit','hide','unhide','dispute']
    content: str|None=Field(default=None,max_length=1000)
    confirm_source_scope: bool=False
    @field_validator('user_id')
    @classmethod
    def identity(cls,value):
        try:return require_id(value)
        except Exception:raise ValueError('invalid user ID') from None


def install_isolation(app,authorize):
    @app.get('/isolation',response_class=HTMLResponse,include_in_schema=False)
    def page():
        return HTMLResponse(Path(__file__).with_name('isolation.html').read_text())

    def call(request,method,kwargs,write=False):
        engine=request.app.state.engine
        if not engine.store.isolation_enabled():raise HTTPException(409,detail={'code':'isolation_not_enabled'})
        try:
            result=engine.long_term._call(method,kwargs,read=not write)
        except PermissionError:raise HTTPException(403,detail={'code':'owner_required'}) from None
        except LookupError:raise HTTPException(404,detail={'code':'shared_fact_not_found'}) from None
        except ValueError as exc:raise HTTPException(409,detail={'code':str(exc) if str(exc) in ('version_conflict','invalid_content','invalid_action','invalid_shared_content') else 'invalid_change'}) from None
        if write:
            if kwargs.get('action') in ('edit','withdraw') and result.get('withdrawn_sources'):
                engine.store.control_exclude(user_id=result['owner_id'],agent_id=kwargs['agent_id'],source_ids=result['withdrawn_sources'])
            with engine.store._connect(write=True) as conn:engine.store._registry_invalidate(conn)
            engine.prefetch.clear()
        return result

    @app.get('/v2/admin/users',dependencies=[Depends(authorize)],tags=['registration'])
    def users(request:Request):
        return dict(items=request.app.state.engine.store.registered_users(request.app.state.settings.agent_id,include_inactive=True))

    @app.put('/v2/admin/users',dependencies=[Depends(authorize)],tags=['registration'])
    def register(body:Registration,request:Request):
        engine=request.app.state.engine
        try:
            result=engine.store.register_user(agent_id=request.app.state.settings.agent_id,**body.model_dump())
        except ValueError:raise HTTPException(422,detail={'code':'invalid_registration'}) from None
        engine.prefetch.clear()
        return result

    @app.get('/v2/memory/shared',dependencies=[Depends(authorize)],tags=['shared-memory'])
    def shared(request:Request,user_id:str=Query(min_length=1,max_length=128)):
        engine=request.app.state.engine;agent=request.app.state.settings.agent_id
        user=engine.store.resolve_registered_user(user_id,agent)
        if is_guest(user):return {'items':[]}
        return dict(items=call(request,'isolation_shared_list',dict(user_id=user,agent_id=agent,users=engine.store.registered_users(agent))))

    @app.patch('/v2/memory/shared/{memory_id}',dependencies=[Depends(authorize)],tags=['shared-memory'])
    def change(memory_id:str,body:SharedChange,request:Request):
        engine=request.app.state.engine;agent=request.app.state.settings.agent_id
        user=engine.store.resolve_registered_user(body.user_id,agent)
        if is_guest(user):raise HTTPException(403,detail={'code':'registered_identity_required'})
        if body.action in ('edit','withdraw') and not body.confirm_source_scope:
            raise HTTPException(409,detail={'code':'source_scope_confirmation_required','message':'修改或撤回会屏蔽原始来源及其关联图谱批次，请确认影响范围。'})
        return call(request,'isolation_shared_change',dict(user_id=user,agent_id=agent,memory_id=memory_id,
            version=body.version,action=body.action,content=body.content,registered_users=engine.store.registered_users(agent)),write=True)

    @app.patch('/v2/admin/shared/{memory_id}',dependencies=[Depends(authorize)],tags=['admin'])
    def admin_change(memory_id:str,body:SharedChange,request:Request):
        if body.action not in ('edit','withdraw') or not body.confirm_source_scope:
            raise HTTPException(409,detail={'code':'source_scope_confirmation_required'})
        return call(request,'isolation_shared_change',dict(user_id='__administrator__',
            agent_id=request.app.state.settings.agent_id,memory_id=memory_id,version=body.version,
            action=body.action,content=body.content,administrator=True),write=True)
