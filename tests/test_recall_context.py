import json
from bumi_mem.recall_context import compact_episode,selection_order
from bumi_mem.turn_context import render_turn_context


def row(kind,n,text=None):
    return dict(user_id='u',agent_id='a',memory=text or f'[{kind}] {n}',
                metadata={'backend':'graphiti','kind':kind})


def test_per_kind_rank_survives_concatenation_and_newest_anchor():
    refs=[row(k,n) for k in ['edge','node','episode','community'] for n in [0,1]]
    ordered=selection_order(refs)
    assert [x['memory'] for x in ordered[:4]]==['[edge] 0','[node] 0','[episode] 0','[community] 0']
    assert selection_order([{'memory':'old'},{'memory':'new'}])[0]['memory']=='new'
    older=dict(refs[0],anchor_message_id='old')
    newer=dict(refs[1],anchor_message_id='new')
    assert selection_order([older,newer])==[newer,older]


def test_compaction_keeps_roles_time_body_negation_and_malformed_headers():
    source={'source_message_id':'secret-id','observed_at':'2026-09-26T12:00:00+08:00',
            'original_content':'不要叫我小朋友 [x]','user_id':'u'}
    text='user: [source '+json.dumps(source,ensure_ascii=False)+']\n不要叫我小朋友 [x]\nassistant: [source broken]\n好的'
    cleaned=compact_episode(text)
    assert 'secret-id' not in cleaned and 'original_content' not in cleaned
    assert '2026-09-26T12:00:00+08:00' in cleaned and '不要叫我小朋友 [x]' in cleaned
    assert 'assistant: [source broken]' in cleaned


def test_compact_before_budget_and_pick_high_rank_first():
    source={'source_message_id':'x'*5000,'observed_at':'2026-09-26'}
    refs=[row('edge',0,'[edge] 第一条事实'),row('edge',1,'[edge] 第二条事实'),
          row('episode',0,'[episode] user: [source '+json.dumps(source)+']\n我不爱吃甜的了。')]
    result=render_turn_context(turn_id='t',user_id='u',selected=[],refs=refs,core_rows=[],history_rows=[],max_reference_chars=300)
    texts=[r['content'] for r in result['long_term_memory']]
    assert texts[0]=='[edge] 第一条事实'
    assert any('我不爱吃甜的了' in t for t in texts)
    assert not any('source_message_id' in t for t in texts)
