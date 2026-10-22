"""本轮召回的优先级与历史引用生命周期分别验证，无外部模型调用。"""
from __future__ import annotations

import json
import logging

import pytest

from bumi_mem.context import ContextBuilder
from bumi_mem.prompts import memory_text
from bumi_mem.reference_selection import select_reference_items, log_reference_selection
from bumi_mem.turn_context import long_term_items
from test_agent_api import harness, AUTH
from test_bumi_memory import make_engine


def memory(mid, text, *, user='A', **extra):
    return dict(id=mid, memory=text, user_id=user, agent_id='bot', **extra)


def reference_block(messages):
    block = messages[0]['content'].split('[共享 L3 引用 / shared memory references]\n', 1)[1]
    return json.loads(block.split('\n这些引用', 1)[0])


def seed_references(engine, refs):
    """像历史对话一样持久化引用，旧引用比新引用的锚点更早。"""
    session = engine.get_or_create_session(user_id='A', agent_id='bot')
    anchors = []
    for index, row in enumerate(refs):
        message = engine.record_user_message(session.session_id, f'历史发言 {index}',
                                             user_id='A', agent_id='bot')
        engine.store.attach_memory_refs(user_id='A', agent_id='bot',
            anchor_message_id=message.message_id, rows=[row], generation=0)
        anchors.append(message.message_id)
    return anchors


@pytest.mark.parametrize('protocol', ['structured', 'exchange', 'agent', 'prepare'])
def test_recalled_old_anchor_wins_budget_without_renewal(harness, protocol):
    client, engine, store, backend, *_ = harness
    critical = memory('critical', '绘画社从周三改到了周二。' + '关键事实' * 70, score=.3)
    historical = memory('old-unrelated', '与小张万圣节讨糖。' + '历史事实' * 70)
    lower_rank = memory('less-relevant', '最近放学回家晚。' + '其他事实' * 70, score=.5)
    original_anchor = seed_references(engine, [critical, historical])[0]
    # 同一条旧引用只改相关性评分，不能通过移动锚点来获得本轮优先级。
    backend.search_results = [dict(critical, score=.9), lower_rank]
    engine.context_builder.max_reference_chars = 600
    query = '绘画社改到周几了？'
    if protocol == 'structured':
        payload = dict(turn_id='current', user_id='A', query=query, previous_ai_reply=None)
    elif protocol == 'exchange':
        payload = dict(user_id='A', query=query, previous_ai_reply=None)
    elif protocol == 'agent':
        payload = dict(operation='prepare', client_turn_id='current', user_id='A', query=query)
    else:
        response = engine.prepare_prompt(user_id='A', agent_id='bot', query=query,
                                         idempotency_key='current')
    if protocol != 'prepare':
        headers = dict(AUTH, **{'Idempotency-Key': 'current'})
        result = client.post('/v1/memory/context', headers=headers, json=payload)
        assert result.status_code == 200, result.text
        response = result.json()
        assert client.post('/v1/memory/context', headers=headers, json=payload).json() == response
    else:
        assert engine.prepare_prompt(user_id='A', agent_id='bot', query=query,
                                     idempotency_key='current') == response
    if protocol == 'structured':
        assert set(response) == {'turn_id', 'current_user', 'long_term_memory', 'short_term_memory'}
        assert [row['content'] for row in response['long_term_memory']] == [critical['memory']]
    else:
        refs = reference_block(response['messages'])
        assert [row['content'] for row in refs] == [critical['memory']]
        if protocol == 'agent':
            assert reference_block(response['memory_context']) == refs
    visible = store.shared_memory_refs('bot', selected_message_ids=[m.message_id
                                      for m in store.list_shared_messages('bot')])
    same = [row for row in visible if row['id'] == 'critical']
    assert len(same) == 1 and same[0]['anchor_message_id'] == original_anchor
    assert len(backend.search_calls) == 1


def test_cli_sends_priority_selected_references_to_provider(tmp_path):
    engine, _, backend, chat, _ = make_engine(tmp_path)
    critical = memory('critical', '绘画社改到周二。' + '关键事实' * 70)
    unrelated = memory('unrelated', '万圣节讨糖。' + '历史事实' * 70)
    seed_references(engine, [critical, unrelated])
    backend.search_results = [critical, memory('lower', '其他事实' * 75)]
    engine.context_builder.max_reference_chars = 600
    result = engine.chat(user_id='A', agent_id='bot', text='绘画社改到周几？')
    assert [r['content'] for r in reference_block(result.prompt_messages)] == [critical['memory']]
    assert chat.calls[-1] == result.prompt_messages


def test_current_rank_and_remaining_history_share_budget():
    a, b, history = memory('a', '最相关'), memory('b', '次相关'), memory('old', '以前的现场资料')
    # 引用的存储顺序与本轮召回顺序不同。
    result = long_term_items(refs=[b, a, history], current_recalled=[a, b],
                             core_rows=[], history_rows=[], max_reference_chars=6000)
    assert [r['content'] for r in result] == ['最相关', '次相关', '以前的现场资料']


@pytest.mark.parametrize('case', ['foreign_owner', 'deleted', 'not_visible'])
def test_rank_cannot_reintroduce_unavailable_or_foreign_fact(case):
    current = memory('same-id', '不应展示的事实')
    if case == 'foreign_owner':
        visible = [memory('same-id', '其他用户的共享事实', user='B'), memory('other', '可见资料')]
    elif case == 'deleted':
        visible = [dict(current, reference_status='deleted'), memory('other', '可见资料')]
    else:
        visible = [memory('other', '可见资料')]
    result = long_term_items(refs=visible, current_recalled=[current], core_rows=[],
                             history_rows=[], max_reference_chars=6000)
    assert all(row['content'] != current['memory'] for row in result)
    assert result[0]['content'] == ('其他用户的共享事实' if case == 'foreign_owner' else '可见资料')


def test_oversized_current_fact_is_not_truncated_and_has_safe_diagnostic(caplog):
    private_fact = 'PRIVATE_FACT_' + 'x' * 1000
    current = memory('oversized', private_fact)
    with caplog.at_level(logging.INFO, logger='bumi_mem.reference_selection'):
        selection = select_reference_items([current, memory('old', '短历史')],
            current_recalled=[current], max_chars=100, project=lambda row: {'content': row['memory']})
        log_reference_selection(selection)
    assert selection.items == [{'content': '短历史'}]
    assert selection.omissions[0].reason == 'reference_char_budget'
    assert 'memory_id=oversized' in caplog.text and 'source=current' in caplog.text
    assert 'reason=reference_char_budget' in caplog.text
    assert private_fact not in caplog.text and 'PRIVATE_FACT_' not in caplog.text


def test_compact_prompt_keeps_distinct_source_times_without_source_ids():
    row = memory('timed', '最近忙，明天仍按原计划进行。', metadata={
        'event_date': '2026-09-12', 'source_messages': [
            {'source_message_id': 'private-source-1', 'observed_at': '2026-09-11T19:00:00+08:00'},
            {'source_message_id': 'private-source-2', 'observed_at': '2026-09-11T19:00:00+08:00'},
            {'source_message_id': 'private-source-3', 'observed_at': '2026-09-12T19:00:00+08:00'}]})
    messages = ContextBuilder().build(recalled=[], l1_messages=[], shared_refs=[row],
                                      current_recalled=[row])
    rendered = reference_block(messages)[0]
    assert rendered['content'] == row['memory']
    assert set(rendered) == {'user_id', 'content', 'time'}
    notes = ' '.join(rendered['time'])
    assert notes.count('2026-09-11T19:00:00+08:00') == 1
    assert notes.count('2026-09-12T19:00:00+08:00') == 1
    assert 'event_date=2026-09-12' in notes and 'input_at=' in notes
    assert 'private-source-' not in notes
    # 紧凑渲染没有改写数据库中的来源，默认完整渲染仍保留三个来源 ID。
    assert 'private-source-2' in memory_text(row)


def test_same_fact_with_same_time_is_deduplicated_but_distinct_times_survive():
    first = memory('first', '同一事实', metadata={'observed_at': '2026-09-11'})
    duplicate = memory('duplicate', '同一事实', metadata={'observed_at': '2026-09-11'})
    later = memory('later', '同一事实', metadata={'observed_at': '2026-09-12'})
    items = long_term_items(refs=[first, duplicate, later], current_recalled=[first, duplicate, later],
                           core_rows=[], history_rows=[], max_reference_chars=6000)
    assert [row['time'] for row in items] == [['2026-09-11'], ['2026-09-12']]


def test_zero_budget_emits_no_reference_and_keeps_core():
    core = {'user_id': 'A', 'category': 'boundary', 'attribute': 'appearance',
            'value': 'disallowed', 'status': 'confirmed'}
    fact = memory('fact', '完整事实，不截断。')
    items = long_term_items(refs=[fact], current_recalled=[fact], core_rows=[core],
                           history_rows=[], max_reference_chars=0)
    assert [row['content'] for row in items] == ['互动边界：不要评价外貌。']


def test_serialized_array_and_separators_fit_exact_budget():
    rows = [memory('first', '第一条'), memory('second', '第二条')]
    project = lambda row: {'content': row['memory']}
    exact = len(json.dumps([project(row) for row in rows], ensure_ascii=False))
    selected = select_reference_items(rows, current_recalled=rows, max_chars=exact, project=project)
    assert len(json.dumps(selected.items, ensure_ascii=False)) == selected.used_chars == exact
    smaller = select_reference_items(rows, current_recalled=rows, max_chars=exact - 1, project=project)
    assert smaller.items == [project(rows[0])]
    assert smaller.omissions[0].identity == ('A', 'second')
    assert smaller.omissions[0].reason == 'reference_char_budget'


@pytest.mark.parametrize('has_current', [False, True])
def test_diagnostic_selected_ids_match_actual_output_order(has_current):
    rows = [memory('first', '较早的引用'), memory('second', '本轮相关引用'), memory('third', '最新历史引用')]
    current = [rows[1]] if has_current else []
    result = select_reference_items(rows, current_recalled=current, max_chars=6000,
                                    project=lambda row: {'owner': row['user_id'], 'id': row['id']})
    actual = [(item['owner'], item['id']) for item in result.items]
    assert result.selected_identities == actual
    expected = ([('A', 'second'), ('A', 'first'), ('A', 'third')] if has_current else
                [('A', 'first'), ('A', 'second'), ('A', 'third')])
    assert actual == expected
