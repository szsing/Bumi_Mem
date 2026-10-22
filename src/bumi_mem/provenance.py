"""Keep internal source anchors out of facts without losing their audit trail."""
from __future__ import annotations

from contextlib import contextmanager
import json
import re
from typing import Any, Iterator


def source_message_ids(metadata: dict[str, Any]) -> set[str]:
    """Read trusted chunk lineage; model-generated IDs never extend this set."""
    values = metadata.get('source_message_ids', [])
    values = [values] if isinstance(values, str) else values
    result = {value for value in values if isinstance(value, str) and value} if isinstance(values, (list, tuple)) else set()
    sources = metadata.get('source_messages')
    for source in sources if isinstance(sources, list) else []:
        if isinstance(source, dict) and isinstance(source.get('source_message_id'), str) and source['source_message_id']:
            result.add(source['source_message_id'])
    return result


def detach_source_annotations(text: str, known_ids: set[str]) -> tuple[str, set[str]]:
    """Remove only labelled, known anchors inside flat annotation brackets.

    Other bracket contents (especially dates) survive. Unknown identifiers and
    ordinary prose are not guessed to be provenance. This is also the read-only
    compatibility path for older extracted memories.
    """
    if not known_ids:
        return text, set()
    ids = '|'.join(re.escape(value) for value in sorted(known_ids, key=len, reverse=True))
    anchor = rf'(?:{ids})(?![\w-])'
    labelled = re.compile(
        rf'(?i:\b(?:source_message_ids?|source_ids?|source)|来源(?:消息)?ID)\s*[:=：]?\s*'
        rf'(?P<ids>{anchor}(?:\s*[,，;]\s*{anchor})*)')
    found: set[str] = set()

    def clean(match: re.Match) -> str:
        block = match.group(0)
        inner = block[1:-1]

        def detach(label: re.Match) -> str:
            prefix = inner[:label.start()].rstrip()
            if prefix and prefix[-1] not in ',，;；':
                return label.group(0)  # ordinary prose, not a separate provenance note
            for value in re.split(r'\s*[,，;]\s*', label['ids']):
                if value in known_ids:
                    found.add(value)
            return ''

        kept, count = labelled.subn(detach, inner)
        if not count or kept == inner:
            return block
        kept = kept.strip(' \t,，;；')
        return block[0] + kept + block[-1] if kept else ''

    cleaned = re.sub(r'\([^()\n]*\)|\[[^\[\]\n]*\]', clean, text)
    # Join only the gap created by a removed annotation; never normalize dates
    # or strip arbitrary parenthesized facts.
    if cleaned != text:
        cleaned = re.sub(r'[ \t]+([.,;!?。；，！？])', r'\1', cleaned)
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned).strip()
    return cleaned, found


def display_memory_body(row: dict[str, Any]) -> str:
    text = str(row.get('memory', row.get('text', row.get('data', '')))).strip()
    metadata = row.get('metadata')
    if not isinstance(metadata, dict) or metadata.get('source') == 'explicit_memory':
        return text
    return detach_source_annotations(text, source_message_ids(metadata))[0]


@contextmanager
def extraction_provenance(memory: Any, *, user_id: str, agent_id: str, run_id: str,
                          metadata: dict[str, Any]) -> Iterator[dict[str, list[str]]]:
    """Adapt the pinned SDK at its LLM and vector-insert boundaries.

    Clean text BEFORE embedding/hash dedup, then attach per-fact citations to
    the actual batch payloads. Chunk lineage remains intact for conservative
    withdrawal/deletion. The caller holds the adapter's serialized write scope.
    """
    from mem0.memory.utils import extract_json, remove_code_blocks

    known_ids = source_message_ids(metadata)
    facts: dict[str, list[str]] = {}
    original_generate = memory.llm.generate_response
    original_insert = memory.vector_store.insert

    def generate(*args, **kwargs):
        response = original_generate(*args, **kwargs)
        if not isinstance(response, str):
            return response
        try:
            raw = remove_code_blocks(response)
            try:
                output = json.loads(raw, strict=False)
            except json.JSONDecodeError:
                output = json.loads(extract_json(raw), strict=False)
        except (ValueError, TypeError):
            # Preserve the SDK's handling of responses unrelated to extraction.
            return response
        if not isinstance(output, dict) or not isinstance(output.get('memory'), list):
            return response
        for item in output['memory']:
            if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                continue
            citations = item.get('source_message_ids', [])
            if (not isinstance(citations, list)
                    or any(not isinstance(value, str) or value not in known_ids for value in citations)):
                raise ValueError('PROVENANCE_INVALID: extracted citations must belong to this input chunk')
            text, inline_ids = detach_source_annotations(item['text'], known_ids)
            leftover_annotation = re.search(r'[\[(][^\])\n]*\bsource(?:_message)?_ids?\s*[:=]', text, re.I)
            if leftover_annotation or any(re.search(r'(?<![\w-])' + re.escape(value) + r'(?![\w-])', text) for value in known_ids):
                raise ValueError('PROVENANCE_INLINE: source anchors belong in source_message_ids, not fact text')
            if item['text'].strip() and not text:
                raise ValueError('PROVENANCE_EMPTY_FACT: extracted memory contains only a source annotation')
            item['text'] = text
            facts[text] = sorted(set(facts.get(text, [])) | set(citations) | inline_ids)
        return json.dumps(output, ensure_ascii=False)

    def insert(*args, **kwargs):
        payloads = kwargs.get('payloads')
        if payloads is None:
            raise RuntimeError('Pinned Mem0 must insert batch payloads by keyword')
        prepared = []
        for payload in payloads:
            updated = dict(payload)
            if ((updated.get('user_id'), updated.get('agent_id'), updated.get('run_id'))
                    == (user_id, agent_id, run_id)):
                ids = facts.get(updated.get('data', ''), [])
                if ids:
                    updated['fact_source_message_ids'] = ids
                    sources = metadata.get('source_messages')
                    updated['fact_source_messages'] = [dict(source)
                        for source in sources if isinstance(source, dict) and source.get('source_message_id') in ids
                    ] if isinstance(sources, list) else []
            prepared.append(updated)
        return original_insert(*args, **{**kwargs, 'payloads': prepared})

    memory.llm.generate_response = generate
    memory.vector_store.insert = insert
    try:
        yield facts
    finally:
        memory.llm.generate_response = original_generate
        memory.vector_store.insert = original_insert
