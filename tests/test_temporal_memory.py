"""Offline contracts for original-message time provenance."""
from dataclasses import asdict
import json

import pytest

from bumi_mem.consolidator import SessionConsolidator
from bumi_mem.context import ContextBuilder
from bumi_mem.prompts import MEMORY_EXTRACTION_INSTRUCTIONS, memory_text
from bumi_mem.schemas import MessageRecord, MemoryWriteUncertainError
from bumi_mem.temporal import normalize_relative_dates
from test_bumi_memory import make_engine, chat_turn, submit


def record(text="昨天搬家了，明天入职。", **kwargs):
    values = dict(message_id="m1", session_id="s1", sequence_no=1, role="user",
                  content=text, token_count=10, user_id="u1", agent_id="a1",
                  created_at="2024-03-01T00:15:00+14:00")
    values.update(kwargs)
    return MessageRecord(**values)


@pytest.mark.parametrize("anchor, expected", [
    ("2024-03-01T00:15:00+14:00", "2024-02-29 2024-03-01 2024-03-02"),
    ("2025-12-31T23:59:00-12:00", "2025-12-30 2025-12-31 2026-01-01"),
    ("2026-01-01T00:00:00Z", "2025-12-31 2026-01-01 2026-01-02"),
])
@pytest.mark.parametrize("text", ["昨天 今天 明天", "Yesterday TODAY tomorrow"])
def test_normalizes_local_day_not_utc_or_worker_day(anchor, expected, text):
    result = normalize_relative_dates(text, anchor)
    assert result.text == expected
    assert result.original_text == text
    assert result.observed_at == anchor


@pytest.mark.parametrize("anchor", ["", "bad", "2024-03-01", "2024-03-01T00:15:00"])
def test_unknown_offset_never_invents_anchor(anchor):
    assert normalize_relative_dates("明天 tomorrow", anchor).text == "明天 tomorrow"


def test_vague_and_compound_terms_are_not_partially_normalized():
    text = "最近想换工作，recently I've been looking. day before yesterday; the day after tomorrow; todays"
    result = normalize_relative_dates(text, record().created_at)
    assert result.text == text
    assert result.observed_at == record().created_at


def test_rendering_is_immutable_and_preserves_role_and_original_wording():
    source = record(role="assistant", text='你问“昨天搬家了吗？”；I suggest moving tomorrow.')
    before = asdict(source)
    rendered = source.to_extraction_message()
    assert rendered['role'] == 'assistant'
    assert rendered['observed_at'] == source.created_at
    assert rendered['source_message_id'] == source.message_id
    header, body = rendered['content'].split('\n', 1)
    provenance = json.loads(header[len('[source '):-1])
    assert provenance['original_content'] == source.content
    assert provenance['speaker_role'] == 'assistant'
    assert 'user_id' not in provenance
    assert '“2024-02-29搬家了吗？”' in body
    assert 'I suggest moving 2024-03-02.' in body
    assert asdict(source) == before
    assert source.to_chat_message() == {'role': 'assistant', 'content': source.content}


def test_every_rendered_l1_message_has_exact_created_at():
    records = [record(), record(message_id='m2', role='assistant', created_at='2024-03-02T23:00:00-05:00')]
    rendered = ContextBuilder().build(recalled=[], l1_messages=records)
    for source, item in zip(records, rendered[1:]):
        header, body = item['content'].split('\n', 1)
        assert json.loads(header[len('[speaker '):-1])['created_at'] == source.created_at
        assert body == source.content


def test_only_real_messages_are_extracted():
    messages = [record(), record(message_id='ref', role='system', text='shared L3 reference'),
                record(message_id='carry', context_only=True)]
    assert [m['source_message_id'] for m in SessionConsolidator._chat_messages(messages)] == ['m1']


def test_legacy_chunks_recover_original_anchors_and_skip_context():
    sources = [record('today', message_id='m1'), record('today', message_id='m2',
               created_at='2026-09-01T22:00:00-08:00'), record('ref', role='system')]
    chunks = [{'messages_json': json.dumps([s.to_chat_message()])} for s in sources]
    restored = SessionConsolidator._restore_extraction_chunks(chunks, sources)
    assert restored[0][0]['content'].endswith('\n2024-03-01')
    assert restored[1][0]['content'].endswith('\n2026-09-01')
    assert restored[2] == []
    with pytest.raises(MemoryWriteUncertainError, match='original source anchor'):
        SessionConsolidator._restore_extraction_chunks(chunks, [])
    invalid_chunks = [{'messages_json': json.dumps([
        {'role': 'user', 'content': 'maybe already written'}])}]
    with pytest.raises(MemoryWriteUncertainError, match='original source anchor'):
        SessionConsolidator._restore_extraction_chunks(invalid_chunks, sources)


def test_memory_times_do_not_turn_record_updates_into_events():
    assert memory_text({'memory': 'plain'}) == 'plain'
    text = memory_text({'memory': 'plan', 'updated_at': '2026-09-21',
                        'metadata': {'observed_at': '2024-03-01T00:15:00+14:00',
                                     'event_date': '2024-03-02'}})
    assert 'source observed_at=2024-03-01T00:15:00+14:00' in text
    assert 'event_date=2024-03-02' in text
    assert 'record updated_at (not event time)=2026-09-21' in text
    assert 'event_date' not in memory_text({'memory': 'plain', 'updated_at': '2026-09-21'})
    assert 'event_date' not in memory_text({'memory': 'plain', 'created_at': '2026-09-21'})


def test_prompt_preserves_filtering_and_overrides_global_date():
    for rule in ('Do not store questions', 'Do not convert assistant suggestions into user facts',
                 'A quote, question about', 'Ignore the global observation date',
                 'Never convert assistant facts to user', 'invent no date or deadline',
                 'Do not assign an event_date to every memory'):
        assert rule in MEMORY_EXTRACTION_INSTRUCTIONS


@pytest.mark.parametrize('provider', ['local', 'deepseek'])
@pytest.mark.parametrize('legacy_chunks', [False, True])
def test_worker_uses_snapshot_times_in_content_and_metadata(tmp_path, monkeypatch, provider, legacy_chunks):
    import bumi_mem.session_store as store_module
    engine, store, memory, _, _ = make_engine(tmp_path, memory_provider=provider, local_chunk_tokens=10)
    monkeypatch.setattr(store_module, 'utcnow', lambda: '2024-03-01T00:15:00+14:00')
    monkeypatch.setattr(store_module, 'message_time', lambda value=None: value or store_module.utcnow())
    chat_turn(engine, '昨天搬家了')
    monkeypatch.setattr(store_module, 'utcnow', lambda: '2024-03-02T23:00:00-05:00')
    monkeypatch.setattr(store_module, 'message_time', lambda value=None: value or store_module.utcnow())
    chat_turn(engine, 'Tomorrow I start my job; recently I have been busy.')
    monkeypatch.setattr(store_module, 'utcnow', lambda: '2026-09-21T12:00:00+08:00')
    monkeypatch.setattr(store_module, 'message_time', lambda value=None: value or store_module.utcnow())
    receipt = submit(engine)
    snapshot = store.get_job(receipt.job_id).payload_json
    originals = json.loads(snapshot)['messages']
    if legacy_chunks:
        ensure_chunks = store.ensure_chunks
        source_by_id = {m['message_id']: m for m in originals}

        def persist_old_rendering(job_id, owner, chunks):
            raw = [[{'role': source_by_id[m['source_message_id']]['role'],
                     'content': source_by_id[m['source_message_id']]['content']}
                    for m in chunk] for chunk in chunks]
            return ensure_chunks(job_id, owner, raw)

        monkeypatch.setattr(store, 'ensure_chunks', persist_old_rendering)
    memory.fail_add_count = 1
    assert engine.worker.run_once()
    assert engine.worker.run_once()
    assert store.get_job(receipt.job_id).status == 'succeeded'
    assert store.get_job(receipt.job_id).payload_json == snapshot
    by_id = {m['message_id']: m for m in originals}
    for call in memory.add_calls:
        assert 'Ignore the global observation date' in call['prompt']
        assert call['metadata']['source_message_ids'] == [m['source_message_id'] for m in call['messages']]
        assert call['metadata']['source_messages'] == [
            {'source_message_id': m['source_message_id'], 'observed_at': m['observed_at']}
            for m in call['messages']]
        assert 'event_date' not in call['metadata']
        for msg in call['messages']:
            source = by_id[msg['source_message_id']]
            assert msg['observed_at'] == source['created_at']
            assert msg['role'] == source['role']
            body = msg['content'].split('\n', 1)[1]
            if source['content'] == '昨天搬家了':
                assert body == '2024-02-29搬家了'
            if source['content'].startswith('Tomorrow'):
                assert body == '2024-03-03 I start my job; recently I have been busy.'


def test_chunk_input_times_remain_distinct_from_event_time():
    text = memory_text({'memory': 'a durable preference', 'metadata': {
        'source_messages': [
            {'source_message_id': 'm1', 'observed_at': '2024-01-01T01:00:00+08:00'},
            {'source_message_id': 'm2', 'observed_at': '2024-01-02T01:00:00-08:00'}]}})
    assert 'input source m1 observed_at=2024-01-01T01:00:00+08:00' in text
    assert 'input source m2 observed_at=2024-01-02T01:00:00-08:00' in text
    assert 'event_date' not in text


def test_local_chunk_budget_counts_rendered_provenance(tmp_path):
    class CharacterCounter:
        def count_text(self, text):
            return len(text)

    engine, _, _, _, _ = make_engine(tmp_path)
    engine.consolidator.memory_token_counter = CharacterCounter()
    messages = [record('today', message_id='m1'), record('today', message_id='m2')]
    # Raw content fits in one chunk; the source headers do not.
    assert len(engine.consolidator._local_chunks(messages, token_limit=50)) == 2


@pytest.mark.parametrize('role', ['user', 'assistant'])
def test_guest_header_is_unconfirmed_without_changing_body(role):
    source = record('original query\nsecond line', user_id='__guest__', role=role)
    rendered = source.to_shared_chat_message()
    header, body = rendered['content'].split('\n', 1)
    speaker = json.loads(header[len('[speaker '):-1])
    assert body == source.content
    assert speaker['created_at'] == source.created_at
    extraction = source.to_extraction_message()
    extraction_header, extraction_body = extraction['content'].split('\n', 1)
    assert extraction_body == source.content
    assert extraction['source_message_id'] == source.message_id
    assert extraction['observed_at'] == source.created_at
    extraction_source = json.loads(extraction_header[len('[source '):-1])
    if role == 'user':
        assert speaker['user_id'] == '__guest__'
        assert speaker['identity_status'] == 'unconfirmed'
        assert extraction_source['identity_status'] == 'unconfirmed'
    else:
        assert 'user_id' not in speaker
        assert 'identity_status' not in speaker
        assert 'identity_status' not in extraction_source


def test_assistant_restatements_are_not_independent_evidence():
    assert 'never independent evidence of user facts' in MEMORY_EXTRACTION_INSTRUCTIONS
