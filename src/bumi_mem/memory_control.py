# 本次改动（2026-10-21）：文字控制用户也必须注册，查询回执核验身份；隔离模式沿用共享短期对话。
"""Conservative current-user commands. This module never calls an intent LLM."""
import hashlib
import json
import re
import time

from bumi_mem.graphiti_memory import graph_group


def parse_control(text):
    text = (text or '').strip()
    if text in ('别记上一句','不要记上一句','别记上一句话','不要记上一句话'):
        return 'exclude_previous', ''
    for action, pattern in (
        ('remember', r'(?:请)?(?:帮我)?(?:记住|记下来|记一下)[：:，,\s]+(.+)'),
        ('forget', r'(?:请)?(?:帮我)?(?:忘掉|忘记|删除这条记忆)[：:，,\s]+(.+)'),
    ):
        match = re.fullmatch(pattern,text)
        if match:
            return action, match[1].strip()
    return None, ''


class MemoryControl:
    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store

    def call(self, method, user, agent, **kwargs):
        return self.engine.long_term._call(method,dict(group_id=graph_group(user,agent),**kwargs),
                                            read=method!='control_withdraw')

    @staticmethod
    def public(op):
        if not op:
            return None
        payload = op['payload']
        return dict(operation_id=op['operation_id'],action=op['action'],status=op['status'],
                    **{k:v for k,v in payload.items() if k in ('message','candidates','expires_at','result','pending_text')})

    def get(self, user, agent, operation_id):
        if self.store.resolve_registered_user(user,agent)!=user:
            raise ValueError('registered_identity_required')
        op = self.store.control_get(user_id=user,agent_id=agent,operation_id=operation_id)
        if op and op['action']=='forget' and op['status']=='uncertain':
            result = self.call('control_receipt',user,agent,operation_id=op['operation_id'])
            if result:
                self.store.control_exclude(user_id=user,agent_id=agent,source_ids=result['source_ids'])
                op = self.store.control_save(user_id=user,agent_id=agent,request_key=op['request_key'],action='forget',
                    status='applied',payload=dict(op['payload'],result=result,message='已核对图数据库回执：删除已生效。'),operation_id=op['operation_id'])
        if op and op['action']=='remember' and op['status']=='queued':
            status = self.store.control_source_status(op['payload']['source_id'])
            if status!='queued':
                payload = dict(op['payload'],message='已完成长期记忆抽取和写入。' if status=='applied' else '写入失败或结果不确定，请查看后台作业；不会自动盲目重放。')
                op = self.store.control_save(user_id=user,agent_id=agent,request_key=op['request_key'],
                    action=op['action'],status=status,payload=payload,operation_id=op['operation_id'])
        return self.public(op)

    def apply(self, request, user, agent, request_key, signature, text, reply):
        existing = self.store.control_get(user_id=user,agent_id=agent,request_key=request_key)
        if existing:
            if existing['payload'].get('signature')!=signature:
                raise ValueError('request_conflict')
            return self.get(user,agent,existing['operation_id'])
        action, argument = parse_control(text)
        if reply:
            action = 'selection'
        if not action:
            return None
        turn = self.store.get_turn(request['turn_id'],agent) if request['turn_id'] else None
        mid = turn['source_message_id'] if turn else None
        payload = dict(signature=signature)
        status = 'queued'
        if action!='remember' and mid:
            with self.store._connect(write=True) as conn:
                conn.execute('UPDATE messages SET memory_excluded=1 WHERE message_id=?',(mid,))
        if action=='remember':
            payload.update(source_id=mid,pending_text=argument,message='已排队；空闲 120 秒或达到批次阈值后整理，尚未写入知识图谱。')
        elif action=='exclude_previous':
            with self.store._connect() as conn:
                rows = conn.execute('''SELECT message_id FROM messages WHERE user_id=? AND agent_id=?
                    AND role='user' AND context_only=0 AND rowid < (SELECT rowid FROM messages WHERE message_id=?)
                    ORDER BY rowid DESC LIMIT 1''',(user,agent,mid)).fetchall()
            if rows:
                # Historical graph batches may contain more than this utterance. Never silently delete them.
                old_mid = rows[0]['message_id']
                if self.store.control_source_status(old_mid)=='applied':
                    status='failed';payload['message']='上一句已参与长期记忆构图，请用“忘记：具体内容”选择记忆并确认来源影响范围。'
                else:
                    source_ids = self.store.control_related_sources([old_mid])
                    self.call('control_block_sources',user,agent,source_ids=source_ids)
                    self.store.control_exclude(user_id=user,agent_id=agent,source_ids=source_ids)
                    status='applied';payload.update(message='上一句已排除，不再作为新抽取证据；原始聊天记录保留。')
            else:
                status='failed';payload['message']='找不到当前用户的上一句。'
        elif action=='forget':
            candidates = self.call('control_candidates',user,agent,query=argument)
            status='needs_selection' if candidates else 'failed'
            payload.update(candidates=[dict(id=r['id'],version=r['version'],content=r['content'],
                affected_facts=r['affected_facts'],source_count=len(r['source_ids']),can_delete=r['reliable']) for r in candidates],
                private_candidates=candidates,expires_at=time.time()+300,
                message='请选择候选；删除会禁止相同旧来源再次召回和抽取，不会擦除原始聊天。' if candidates else '没有找到候选，请提供更准确的记忆内容。')
        elif action=='selection':
            op = self.store.control_get(user_id=user,agent_id=agent,operation_id=reply['operation_id'])
            if not op or op['action']!='forget':
                raise ValueError('operation_not_found')
            chosen = next((r for r in op['payload'].get('private_candidates',[]) if r['id']==reply['memory_id'] and r['version']==reply['version']),None)
            if not chosen or (op['payload'].get('selected') and op['payload']['selected']!=chosen):
                raise ValueError('candidate_changed')
            if op['status']=='applied':
                payload.update(result=op['payload'].get('result'),message='该操作已执行。');status='applied'
            else:
                data = op['payload']
                if op['status'] not in ('needs_selection','uncertain') or data.get('expires_at',0)<time.time():
                    raise ValueError('selection_expired')
                candidate = next((r for r in data.get('private_candidates',[]) if r['id']==reply['memory_id'] and r['version']==reply['version']),None)
                if data.get('selected') and data['selected'] != candidate:
                    raise ValueError('candidate_changed')
                if not candidate or not candidate['reliable']:
                    raise ValueError('source_lineage_unavailable')
                if not reply.get('confirm_source_scope'):
                    raise ValueError('source_scope_confirmation_required')
                # Intent is durable before external mutation; uncertain outcomes are reconciled by operation ID.
                uncertain = dict(data,selected=candidate)
                self.store.control_save(user_id=user,agent_id=agent,request_key=op['request_key'],action=op['action'],
                    status='uncertain',payload=uncertain,operation_id=op['operation_id'])
                try:
                    result = self.call('control_withdraw',user,agent,operation_id=op['operation_id'],candidate=candidate,
                                       expires_at=data['expires_at'])
                except ValueError as exc:
                    if str(exc) in ('candidate_changed','selection_expired','source_lineage_unavailable'):
                        self.store.control_save(user_id=user,agent_id=agent,request_key=op['request_key'],action=op['action'],
                            status='failed',payload=dict(uncertain,message=str(exc)),operation_id=op['operation_id'])
                    raise
                self.store.control_exclude(user_id=user,agent_id=agent,source_ids=result['source_ids'])
                payload.update(result=result,message='已停止召回，并阻止这些旧来源再次进入长期记忆；原始聊天和备份保留。');status='applied'
                self.store.control_save(user_id=user,agent_id=agent,request_key=op['request_key'],action=op['action'],
                    status=status,payload=dict(data,result=result,message=payload['message']),operation_id=op['operation_id'])
        op = self.store.control_save(user_id=user,agent_id=agent,request_key=request_key,action=action,status=status,payload=payload)
        return self.public(op)

    def context(self, *, user_id, agent_id, body):
        if getattr(self.engine.long_term,'backend_name',None)!='graphiti':
            raise ValueError('graphiti_required')
        if self.store.resolve_registered_user(user_id,agent_id)!=user_id:
            raise ValueError('registered_identity_required')
        signature = hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        # Separate idempotency namespace without changing the old strict four-field protocol.
        internal_id = 'mc2_'+hashlib.sha256(body['turn_id'].encode()).hexdigest()
        receipt = None
        def hook(request):
            nonlocal receipt
            receipt = self.apply(request,user_id,agent_id,internal_id,signature,body.get('query'),body.get('memory_control_reply'))
            if request['turn_id']:
                with self.store._connect(write=True) as conn:
                    conn.execute('UPDATE interaction_turns SET context_epoch=? WHERE turn_id=?',
                                 (self.store._context_epoch(conn,agent_id),request['turn_id']))
        with self.engine._pair_lock(user_id,agent_id):
            self.store.control_register_request(user_id=user_id,agent_id=agent_id,request_key=internal_id,signature=signature)
            existing = self.store.control_get(user_id=user_id,agent_id=agent_id,request_key=internal_id)
            if existing and existing['payload'].get('signature') != signature:
                raise ValueError('request_conflict')
            result = self.engine.process_turn_context(turn_id=internal_id,user_id=user_id,agent_id=agent_id,
                query=body.get('query'),previous_ai_reply=body.get('previous_ai_reply'),
                query_occurred_at=body.get('query_occurred_at'),
                previous_reply_occurred_at=body.get('previous_reply_occurred_at'),_control_hook=hook)
            result = dict(result,turn_id=body['turn_id'])
            result['long_term_memory'] = [r for r in result['long_term_memory'] if r.get('user_id')==user_id]
            with self.store._connect() as conn:
                allowed = {(r['role'],r['content'],r['created_at']) for r in conn.execute(
                    'SELECT m.* FROM messages m JOIN sessions s USING(session_id) WHERE s.user_id=? AND s.agent_id=? AND m.memory_excluded=0', (user_id,agent_id))}
                pending = conn.execute('''SELECT operation_id,payload FROM memory_control_operations
                    WHERE user_id=? AND agent_id=? AND action='remember' AND status='queued'
                    ORDER BY created_at DESC LIMIT 20''', (user_id,agent_id)).fetchall()
            if not self.store.isolation_enabled():
                result['short_term_memory'] = [r for r in result['short_term_memory'] if (r['role'],r['content'],r['time']) in allowed]
            result['pending_memory'] = []
            for row in pending:
                data = json.loads(row['payload'])
                if self.store.control_source_status(data['source_id'])=='queued':
                    result['pending_memory'].append(dict(operation_id=row['operation_id'],user_id=user_id,
                        content=data['pending_text'],status='queued',persisted=False))
            if receipt:
                result['memory_control']=receipt
            return result
