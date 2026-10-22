# 本次改动（2026-10-21）：从原始用户陈述保守识别明确双人关系和共同经历，排除歧义名称、指代和推断。
"""Conservative explicit joint assertions; no inferred school/location relationships.

Registration is authoritative. Ambiguous names and pronouns fail closed. Only the
matched clause is shared: trailing personal statements never accompany it.
"""
import hashlib
import json
import re


def unique_names(users):
    names = {}
    for user in users:
        if not user.get('active',True):continue
        for name in {user['user_id'],user['display_name'],*user.get('aliases',[])}:
            if name in ('我','你','他','她','我们','你们','他们','她们','__guest__'):
                continue
            names.setdefault(name,set()).add(user['user_id'])
    return {name:next(iter(ids)) for name,ids in names.items() if len(ids)==1}


def original_source(message):
    content = message.get('content','')
    if content.startswith('[source '):
        try:
            data,_ = json.JSONDecoder().raw_decode(content[len('[source '):])
            return data.get('original_content',''), data
        except (ValueError,TypeError):return '',{}
    return '',{}


def joint_assertions(messages, *, owner_id, users):
    names = unique_names(users)
    active = {u['user_id'] for u in users if u.get('active',True)}
    if owner_id not in active:return []
    aliases = sorted((n for n,uid in names.items() if uid!=owner_id),key=len,reverse=True)
    if not aliases:return []
    person='(?:'+'|'.join(re.escape(n) for n in aliases)+')'
    # Reject questions, hypothetical speech and embedded quotations rather than guessing intent.
    relation = re.compile(r'^(?:我和|我跟|我与)('+person+r')(?:是|成为了?)(朋友|好朋友|同学|大学同学|同事|室友|兄弟|姐妹)$')
    same_school = re.compile(r'^(?:我和|我跟|我与)('+person+r')(?:在同一所大学(?:上学|读书)?|在同一个大学(?:上学|读书)?|就读于同一所大学)$')
    event = re.compile(r'^(?:我和|我跟|我与)('+person+r')(今天|昨天|前天|上周|上个月)?一起(去了|去过|参观了|游览了)([^，,；;。！？!?“”"「」]{1,40})$')
    result=[]
    for message in messages:
        if message.get('role')!='user':continue
        text, source = original_source(message)
        if source.get('user_id')!=owner_id or not source.get('source_message_id'):continue
        for clause in re.split(r'[。；;\n]',text):
            clause=clause.strip()
            if any(w in clause for w in ('如果','假如','假设','据说','听说','不是','并非','不要','能否','是否','？','?','“','”','"','「','」')):continue
            match=relation.fullmatch(clause) or same_school.fullmatch(clause) or event.fullmatch(clause)
            if not match:continue
            other=names[match[1]]
            participants=sorted({owner_id,other})
            if len(participants)!=2:continue
            identity=json.dumps([source['source_message_id'],clause],ensure_ascii=False)
            result.append(dict(id=hashlib.sha256(identity.encode()).hexdigest(),content=clause,
                owner_id=owner_id,participants=participants,source_id=source['source_message_id'],
                observed_at=source.get('observed_at',''),kind='relationship' if relation.fullmatch(clause) or same_school.fullmatch(clause) else 'joint_event'))
    return result
