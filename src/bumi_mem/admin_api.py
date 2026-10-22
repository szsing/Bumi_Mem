"""Local administrator UI and API; uses the service's existing bearer key."""
from pathlib import Path
from typing import Literal
from fastapi import Depends,HTTPException,Query,Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel,Field,ConfigDict


class AdminChange(BaseModel):
    model_config=ConfigDict(extra='forbid')
    group_id:str=Field(min_length=1,max_length=128)
    version:int=Field(ge=0)
    action:Literal['edit','delete','restore']
    content:str|None=Field(default=None,max_length=8000)


def install_admin(app,authorize):
    def call(request,method,kwargs,write=False):
        engine=request.app.state.engine
        if getattr(engine.long_term,'backend_name',None)!='graphiti':
            raise HTTPException(409,detail={'code':'graphiti_required'})
        try:
            result=engine.long_term._call(method,kwargs,read=not write,timeout=None if write else 120)
        except LookupError:
            raise HTTPException(404,detail={'code':'memory_not_found'}) from None
        except ValueError as exc:
            code='version_conflict' if str(exc)=='version_conflict' else 'invalid_edit'
            raise HTTPException(409 if code=='version_conflict' else 422,detail={'code':code}) from None
        if write:
            # Drop materialized references and cached context in THIS service store.
            with engine.store._connect(write=True) as conn:
                agents=[r[0] for r in conn.execute('SELECT DISTINCT agent_id FROM shared_memory_refs')]
                conn.execute("UPDATE shared_memory_refs SET status='deleted' WHERE status='active'")
                for agent in agents:engine.store._bump_context_epoch(conn,agent)
        return result

    @app.get('/admin',response_class=HTMLResponse,include_in_schema=False)
    def page():
        return HTMLResponse(Path(__file__).with_name('admin.html').read_text())

    @app.get('/v1/admin/groups',dependencies=[Depends(authorize)],tags=['admin'])
    def groups(request:Request):return {'items':call(request,'admin_groups',{})}

    @app.get('/v1/admin/graph',dependencies=[Depends(authorize)],tags=['admin'])
    def graph(request:Request,group_id:str|None=None,limit:int=Query(default=350,ge=1,le=800)):
        return call(request,'admin_graph',dict(group_id=group_id,limit=limit))

    @app.get('/v1/admin/memories',dependencies=[Depends(authorize)],tags=['admin'])
    def memories(request:Request,group_id:str|None=None,q:str=Query(default='',max_length=200),
                 status:Literal['active','deleted','all']='active',offset:int=Query(default=0,ge=0),
                 limit:int=Query(default=30,ge=1,le=100)):
        return call(request,'admin_list',dict(group_id=group_id,query=q,status=status,offset=offset,limit=limit))

    @app.get('/v1/admin/memories/{memory_id}',dependencies=[Depends(authorize)],tags=['admin'])
    def detail(request:Request,memory_id:str,group_id:str):
        return call(request,'admin_detail',dict(group_id=group_id,memory_id=memory_id))

    @app.patch('/v1/admin/memories/{memory_id}',dependencies=[Depends(authorize)],tags=['admin'])
    def change(request:Request,memory_id:str,body:AdminChange):
        return call(request,'admin_change',dict(group_id=body.group_id,memory_id=memory_id,
                    version=body.version,action=body.action,content=body.content),write=True)
