"""Cheap presentation of Graphiti recall; no extra retrieval or model calls."""
import json
import re

_HEADER = re.compile(r'^(user|assistant): \[source (.*)$')


def compact_episode(text):
    """Remove only valid machine source headers; preserve body and source time.

    JSON raw_decode handles closing brackets and newlines escaped in source text.
    Malformed headers are left intact rather than risking deletion of evidence.
    """
    lines=[]
    for line in text.splitlines():
        match=_HEADER.match(line)
        if match:
            try:
                source,end=json.JSONDecoder().raw_decode(match[2])
                if not isinstance(source,dict) or match[2][end:].strip()!=']':
                    raise ValueError('not a source header')
                stamp=str(source.get('observed_at') or '')
                lines.append(f'{match[1]}'+(f' [{stamp}]' if stamp else '')+':')
                continue
            except (ValueError,TypeError):
                pass
        lines.append(line)
    return '\n'.join(lines)


def reference_body(row):
    text=str(row.get('memory',row.get('text',row.get('data','')))).strip()
    if text.startswith('[episode] '):
        text='[episode] '+compact_episode(text[len('[episode] '):])
    return text


def selection_order(refs):
    """Newest anchor first, Graphiti per-kind rank first within each anchor.

    Interleave equally ranked kinds instead of letting the end of a concatenated
    kind list win. Unranked legacy rows retain the original newest-first rule.
    No comparison of scores produced from different object representations.
    """
    refs=list(refs)
    groups={}
    for i,row in enumerate(refs):
        meta=row.get('metadata') or {}
        if meta.get('backend')=='graphiti':
            anchor=row.get('anchor_message_id','')
            key=(anchor,row.get('user_id'),row.get('agent_id'))
            groups.setdefault(key,[]).append((i,row))
    ordered=[]; emitted=set()
    for row in reversed(refs):
        meta=row.get('metadata') or {}
        if meta.get('backend')!='graphiti':
            ordered.append(row);continue
        key=(row.get('anchor_message_id',''),row.get('user_id'),row.get('agent_id'))
        if key in emitted:continue
        emitted.add(key)
        counts={}; ranked=[]
        for i,item in groups[key]:
            kind=(item.get('metadata') or {}).get('kind','')
            rank=counts.get(kind,0);counts[kind]=rank+1
            ranked.append((rank,i,item))
        ordered.extend(item for _,_,item in sorted(ranked,key=lambda x:(x[0],x[1])))
    return ordered
