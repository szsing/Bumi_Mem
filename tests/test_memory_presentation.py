"""Model views must be compact without losing identity, time or provenance."""
from copy import deepcopy
import json

import pytest

from bumi_mem.context import ContextBuilder
from bumi_mem.memory_view import reference_item
from bumi_mem.provenance import detach_source_annotations
from bumi_mem.turn_context import long_term_items
from test_reference_selection import reference_block


@pytest.mark.parametrize('text, expected', [
    ('Jon plans a fair (source_message_id: anchor-1).', 'Jon plans a fair.'),
    ('Jon plans a fair [source_message_ids: anchor-1, anchor-2].', 'Jon plans a fair.'),
    ('Jon plans a fair (source anchor-1, observed 2023-07-21).',
     'Jon plans a fair (observed 2023-07-21).'),
    ('Jon plans a fair (2023-07-21; source_message_id: anchor-1).',
     'Jon plans a fair (2023-07-21).'),
    ('Jon计划五月参赛（日期不变）。', 'Jon计划五月参赛（日期不变）。'),
    ('五月参赛 [来源ID：anchor-1]。', '五月参赛。'),
    ('Jon plans a fair (2023-07-21, ticket 123).', 'Jon plans a fair (2023-07-21, ticket 123).'),
    ('Business source_message_id: anchor-1 is part of a quoted field.',
     'Business source_message_id: anchor-1 is part of a quoted field.'),
    ('An explanation (the source anchor-1 is a business field).',
     'An explanation (the source anchor-1 is a business field).'),
    ('A reference (source_message_id: unknown-id).', 'A reference (source_message_id: unknown-id).'),
])
def test_only_known_provenance_annotations_are_detached(text, expected):
    cleaned, ids = detach_source_annotations(text, {'anchor-1','anchor-2'})
    assert cleaned == expected
    if cleaned == text:
        assert not ids
    else:
        assert 'anchor-1' in ids


def test_old_record_is_cleaned_only_for_display_and_source_dates_survive():
    row = {'id':'private-memory-uuid','user_id':'Jon','agent_id':'bot',
        'memory':'Jon plans a fair in May 2023 (source_message_id: anchor-1, observed 2023-04-03).',
        'updated_at':'2026-10-15T10:00:00+00:00', 'metadata':{
            'source':'l1_session','source_message_ids':['anchor-1'],
            'source_messages':[{'source_message_id':'anchor-1','observed_at':'2023-04-03T12:00:00+00:00'}]}}
    before = deepcopy(row)
    prompt = ContextBuilder().build(recalled=[], l1_messages=[], shared_refs=[row], current_recalled=[row])
    view = reference_block(prompt)[0]
    assert set(view) == {'user_id','content','time'}
    assert view['user_id'] == 'Jon' and 'May 2023' in view['content']
    assert 'observed 2023-04-03' in view['content']
    assert '2023-04-03T12:00:00+00:00' in ' '.join(view['time'])
    for field in ('private-memory-uuid','anchor-1','2026-10-15','reference_status','fact_status'):
        assert field not in json.dumps(view)
    api = long_term_items(refs=[row], current_recalled=[row], core_rows=[],history_rows=[],max_reference_chars=6000)
    assert api[0]['content'] == view['content']
    assert api[0]['time'] == ['2023-04-03T12:00:00+00:00']
    assert row == before  # backend lookup, deletion lineage and old snapshots remain intact


def test_explicit_user_text_and_meaningful_fact_status_are_not_discarded():
    raw = 'My ticket (source_message_id: anchor-1) is cancelled.'
    row = {'memory':raw,'user_id':'Jon','metadata':{
        'source':'explicit_memory','source_message_ids':['anchor-1']}}
    assert reference_item(row)['content'] == raw
    assert reference_item(dict(row, fact_status='historical_superseded'))['content'] == '历史：' + raw
    assert reference_item(dict(row, reference_status='deleted')) is None


def test_short_reference_budget_is_spent_on_facts_not_internal_uuid():
    rows = [{'id':f'private-{i}-'+'x'*100,'user_id':'A','memory':f'fact {i}',
             'updated_at':'2026-10-15T10:00:00+00:00'} for i in range(4)]
    builder = ContextBuilder(max_reference_chars=260)
    prompt = builder.build(recalled=[], l1_messages=[], shared_refs=rows, current_recalled=rows)
    refs = reference_block(prompt)
    assert len(refs) == 4
    assert len(json.dumps(refs,ensure_ascii=False)) <= 260
    assert [r['content'] for r in refs] == ['fact 0','fact 1','fact 2','fact 3']
    assert all(set(r) == {'user_id','content','time'} for r in refs)


@pytest.mark.parametrize('state,label', [('planned','计划中'),('promised','已承诺'),
    ('completed','已完成'),('cancelled','已取消'),('retracted','已撤回')])
def test_meaningful_status_is_expressed_in_content(state,label):
    row = reference_item({'user_id':'A','memory':'the dance event','fact_status':state})
    assert row == {'user_id':'A','content':label+'：the dance event','time':[]}


def test_empty_legacy_lineage_does_not_match_arbitrary_text():
    row = {'user_id':'A','memory':'A fact (source_message_id: actual-business-number).',
        'metadata':{'source_message_ids':None,'source_messages':[{'source_message_id':''}]}}
    assert reference_item(row)['content'] == row['memory']
