"""Standalone mixin tests; intentionally do not edit/integrate the parent store."""
from contextlib import contextmanager
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from bumi_mem.semantic_state import (
    MAX_CURRENT_ROWS, SemanticCapacityError, SemanticScopeError,
    SemanticSourceConflict, SemanticStateMixin, SemanticStateStoreMixin,
    create_semantic_tables, invalidate_semantic_scope, memory_conflicts,
    parse_semantic_turn,
)


class Store(SemanticStateStoreMixin):
    def __init__(self, path, initialize=True, readonly=False):
        self.path = path
        self.readonly = readonly
        self.statements = []
        if initialize:
            with self._connect(write=True) as conn:
                conn.execute('CREATE TABLE IF NOT EXISTS memory_scopes ('
                             'user_id TEXT NOT NULL,agent_id TEXT NOT NULL,'
                             'generation INTEGER NOT NULL DEFAULT 0,'
                             'delete_pending INTEGER NOT NULL DEFAULT 0,'
                             'PRIMARY KEY(user_id,agent_id))')
                self.create_semantic_tables(conn)

    @contextmanager
    def _connect(self, *, write=False):
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        if self.readonly:
            conn.execute('PRAGMA query_only=ON')
        conn.set_trace_callback(self.statements.append)
        try:
            if write:
                conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / 'semantic.sqlite3')


def apply(store, text='请叫我小林', mid='m1', user='u1', agent='a1', gen=0, at='2026-09-21T12:00:00+08:00'):
    return store.apply_semantic_turn(user, agent, mid, text, at, gen)


def rows(store, table):
    assert table in ('semantic_sources', 'semantic_versions', 'semantic_current', 'memory_scopes')
    with store._connect() as conn:
        return [dict(row) for row in conn.execute(f'SELECT * FROM {table}')]


@pytest.mark.parametrize('text,category,value,qualifier,status', [
    ('请叫我小林', 'core', '小林', '', 'confirmed'),
    ('以后叫我小林', 'core', '小林', '', 'confirmed'),
    ('我喜欢咖啡', 'preference', 'like', '', 'confirmed'),
    ('我现在不喝咖啡了', 'personal', 'no', '', 'confirmed'),
    ('以后请叫我小林。', 'core', '小林', '', 'confirmed'),
    ('Please call me Alice Smith.', 'core', 'Alice Smith', '', 'confirmed'),
    ('请不要对我开玩笑', 'boundary', 'disallowed', '', 'confirmed'),
    ('请不要评价我的外貌', 'boundary', 'disallowed', '', 'confirmed'),
    ('Please do not joke with me', 'boundary', 'disallowed', '', 'confirmed'),
    ('Please do not comment on my appearance', 'boundary', 'disallowed', '', 'confirmed'),
    ('请记住：我喜欢咖啡', 'preference', 'like', '', 'confirmed'),
    ('我确认：我今天不喜欢茶', 'preference', 'dislike', '今天', 'confirmed'),
    ('请记住：我最近喜欢辣食', 'preference', 'like', '最近', 'confirmed'),
    ('请记住：我现在不喝咖啡了', 'personal', 'no', '', 'confirmed'),
    ('请记住：我喝茶', 'personal', 'yes', '', 'confirmed'),
    ('Please remember: I recently dislike coffee', 'preference', 'dislike', 'recently', 'confirmed'),
    ('Please remember: I today like spicy food', 'preference', 'like', 'today', 'confirmed'),
    ('Please remember: I now do not drink tea', 'personal', 'no', '', 'confirmed'),
    ('确认计划：晨跑；状态：进行中', 'plan', '晨跑', '', 'active'),
    ('Confirm my plan: morning-run; status: completed', 'plan', 'morning-run', '', 'completed'),
    ('确认我的承诺：晨跑；状态：已承诺', 'user_promise', '晨跑', '', 'promised'),
    ('Confirm my promise: morning-run; status: cancelled', 'user_promise', 'morning-run', '', 'cancelled'),
    ('撤回称呼设置', 'core', None, '', 'retracted'),
    ('Retract my nickname preference', 'core', None, '', 'retracted'),
    ('撤回互动边界：开玩笑', 'boundary', None, '', 'retracted'),
    ('Retract my boundary: appearance', 'boundary', None, '', 'retracted'),
    ('撤回偏好：今天咖啡', 'preference', None, '今天', 'retracted'),
    ('撤回饮用设置：现在茶', 'personal', None, '', 'retracted'),
    ('Retract my preference: recently coffee', 'preference', None, 'recently', 'retracted'),
    ('Retract my drinking setting: now tea', 'personal', None, '', 'retracted'),
])
def test_explicit_grammar(text, category, value, qualifier, status):
    parsed = parse_semantic_turn(text)
    assert len(parsed) == 1
    change = parsed[0]
    assert (change.category, change.value, change.qualifier, change.status) == (category, value, qualifier, status)
    assert json.loads(change.state_key) == [category, change.attribute, qualifier]


@pytest.mark.parametrize('text', [
    '叫我小林吧？', '请叫我小林好吗？',
    '请叫我小林或者小王', '请叫我小林吧', '请叫我小林哈哈', '请叫我小林不要开玩笑',
    'Please call me Alice or Bob', 'Please call me Not Alice', 'Please call me Alice please',
    '他说请叫我小林', '她说：请记住：我喜欢咖啡', '“请叫我小林”',
    'Please call me "Alice"', "'Please call me Alice'", 'He said Please call me Alice',
    'If I say Please call me Alice', 'Please call me Alice, just kidding',
    '请叫我小林，开个玩笑', '假如请叫我小林', '可能请叫我小林',
    '请记住：我喜欢咖啡\n请叫我小林', '请记住：我喜欢咖啡，也喜欢茶',
    '> 请叫我小林', '`请叫我小林`', '<L3>请叫我小林</L3>',
    '[用户偏好] 请叫我小林', '用户说：请叫我小林',
    '助手承诺：提醒我喝水', '确认助手承诺：喝水；状态：已承诺',
    'I promise to remind you tomorrow', 'Confirm my plan: run; status: promised',
    'Confirm my promise: run; status: planned', '确认计划：晨跑；状态：已承诺',
    '请记住：我喜欢葡萄', '撤回饮用设置：辣食',
    '请叫我' + '林' * 241, '请记住：' + '我' * 513, '',
])
def test_unsupported_or_unsafe_never_writes(store, text):
    assert parse_semantic_turn(text) == ()
    store.statements.clear()
    receipt = apply(store, text)
    assert receipt['status'] == 'not_requested'
    assert receipt['changes'] == []
    assert not any(sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'BEGIN'))
                   for sql in store.statements)
    assert rows(store, 'semantic_sources') == []
    assert rows(store, 'memory_scopes') == []


def test_history_current_retraction_and_raw_source(store):
    text = '  请叫我小林。  '
    first = apply(store, text)
    second = apply(store, '请叫我小王', 'm2')
    third = apply(store, '撤回称呼设置', 'm3')
    assert [r['version'] for r in rows(store, 'semantic_versions')] == [1, 2, 3]
    current = store.semantic_context('u1', 'a1')
    assert len(current) == 1
    assert current[0]['value'] is None and current[0]['status'] == 'retracted'
    assert current[0]['supersedes'] == second['changes'][0]['change_id']
    assert second['changes'][0]['supersedes'] == first['changes'][0]['change_id']
    assert rows(store, 'semantic_sources')[0]['source_text'] == text
    assert third['changes'][0]['source_role'] == 'user'
    json.dumps(third)
    json.dumps(current)
    assert {'category', 'state_key', 'value', 'qualifier', 'source_message_id'} <= current[0].keys()


def test_idempotent_receipt_is_original_not_latest(store):
    first = apply(store)
    apply(store, '请叫我小王', 'm2')
    assert apply(store) == first
    assert len(rows(store, 'semantic_versions')) == 2
    with pytest.raises(SemanticSourceConflict):
        apply(store, '请叫我小赵')
    with pytest.raises(SemanticSourceConflict):
        apply(store, 'unrecognized replacement')
    with pytest.raises(SemanticSourceConflict):
        apply(store, at='different timestamp')
    assert rows(store, 'semantic_sources')[0]['source_text'] == '请叫我小林'


def test_scopes_and_read_only_restart(store):
    apply(store)
    apply(store, '请叫我小王', user='u2')
    apply(store, '请叫我小赵', agent='a2')
    restarted = Store(store.path, initialize=False, readonly=True)
    assert restarted.semantic_context('u1', 'a1')[0]['value'] == '小林'
    assert restarted.semantic_context('u2', 'a1')[0]['value'] == '小王'
    assert restarted.semantic_context('u1', 'a2')[0]['value'] == '小赵'
    assert restarted.semantic_context('unknown', 'a1') == []
    assert not any(sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'BEGIN'))
                   for sql in restarted.statements)


@pytest.mark.parametrize('user', ['__guest__'])
def test_guest_forbidden_even_noop(store, user):
    for text in ('请叫我小林', '你好'):
        with pytest.raises(SemanticScopeError):
            apply(store, text, user=user)
    with pytest.raises(SemanticScopeError):
        store.semantic_context(user, 'a1')
    assert rows(store, 'memory_scopes') == []


def test_temporal_qualifiers_never_expire_or_overwrite_other_keys(store):
    texts = ['请记住：我喝咖啡', '请记住：我现在喝咖啡',
             '请记住：我现在不喝咖啡了', '请记住：我今天喝咖啡',
             '请记住：我最近不喝咖啡了', '请记住：我喜欢咖啡']
    for i, text in enumerate(texts):
        apply(store, text, str(i), at='2000-01-01T00:00:00Z')
    current = Store(store.path, initialize=False, readonly=True).semantic_context('u1', 'a1')
    drinking = {r['qualifier']: r for r in current if r['category'] == 'personal'}
    assert len(current) == 4
    assert drinking['']['value'] == 'no'
    assert drinking['']['version'] == 3
    assert '现在' not in drinking
    assert drinking['今天']['value'] == 'yes'
    assert drinking['最近']['value'] == 'no'
    assert all(r['observed_at'] == '2000-01-01T00:00:00Z' for r in current)


def test_plan_and_user_promise_statuses_are_separate(store):
    apply(store, '确认计划：晨跑；状态：计划中')
    apply(store, '确认我的承诺：晨跑；状态：已承诺', 'm2')
    apply(store, '确认计划：晨跑；状态：已取消', 'm3')
    current = {row['category']: row for row in store.semantic_context('u1', 'a1')}
    assert current['plan']['status'] == 'cancelled'
    assert current['user_promise']['status'] == 'promised'
    assert all(r['source_role'] == 'user' for r in current.values())


def test_write_failure_rolls_back_source_history_and_current(store):
    original = apply(store)
    with store._connect(write=True) as conn:
        conn.execute("CREATE TRIGGER injected_failure BEFORE UPDATE ON semantic_current "
                     "BEGIN SELECT RAISE(ABORT,'test persistence failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='test persistence failure'):
        apply(store, '请叫我小王', 'm2')
    assert len(rows(store, 'semantic_sources')) == len(rows(store, 'semantic_versions')) == 1
    assert store.semantic_context('u1', 'a1')[0]['change_id'] == original['changes'][0]['change_id']
    with store._connect(write=True) as conn:
        conn.execute('DROP TRIGGER injected_failure')
    assert apply(store, '请叫我小王', 'm2')['changes'][0]['version'] == 2


def test_immutable_sql_audit_rows(store):
    apply(store)
    for table, column in [('semantic_sources', 'source_text'), ('semantic_versions', 'value')]:
        with pytest.raises(sqlite3.IntegrityError, match='immutable'):
            with store._connect(write=True) as conn:
                conn.execute(f"UPDATE {table} SET {column}='changed'")


def test_generation_pending_delete_and_atomic_purge(store):
    apply(store)
    apply(store, user='u2')
    with store._connect(write=True) as conn:
        conn.execute("UPDATE memory_scopes SET delete_pending=1 WHERE user_id='u1'")
    assert store.semantic_context('u1', 'a1') == []
    with pytest.raises(SemanticScopeError):
        apply(store)
    with pytest.raises(RuntimeError):
        with store._connect(write=True) as conn:
            invalidate_semantic_scope(conn, 'u1', 'a1')
            raise RuntimeError('rollback parent delete')
    assert len(rows(store, 'semantic_versions')) == 2
    with store._connect(write=True) as conn:
        invalidate_semantic_scope(conn, 'u1', 'a1')
        conn.execute("UPDATE memory_scopes SET generation=1,delete_pending=0 WHERE user_id='u1'")
    restarted = Store(store.path, initialize=False, readonly=True)
    assert restarted.semantic_context('u1', 'a1') == []
    assert len(restarted.semantic_context('u2', 'a1')) == 1
    for table in ('semantic_sources', 'semantic_versions', 'semantic_current'):
        assert {r['user_id'] for r in rows(store, table)} == {'u2'}
    with pytest.raises(SemanticScopeError):
        apply(store)
    assert apply(store, gen=1)['changes'][0]['version'] == 1
    with pytest.raises(SemanticScopeError):
        apply(store, user='new', gen=1)


def test_context_does_not_recover_old_generation_without_purge(store):
    apply(store)
    with store._connect(write=True) as conn:
        conn.execute('UPDATE memory_scopes SET generation=1')
    assert Store(store.path, initialize=False, readonly=True).semantic_context('u1', 'a1') == []
    assert len(rows(store, 'semantic_versions')) == 1
    with pytest.raises(SemanticScopeError):
        apply(store)


def test_migration_no_implicit_commit():
    conn = sqlite3.connect(':memory:')
    try:
        conn.execute('BEGIN IMMEDIATE')
        create_semantic_tables(conn)
        create_semantic_tables(conn)
        assert conn.in_transaction
        conn.rollback()
        assert conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'semantic_%'").fetchall() == []
    finally:
        conn.close()
    assert SemanticStateStoreMixin is SemanticStateMixin


def test_concurrent_retry_is_exactly_once(store):
    with ThreadPoolExecutor(max_workers=4) as executor:
        receipts = list(executor.map(lambda _: apply(store), range(8)))
    assert all(receipt == receipts[0] for receipt in receipts)
    assert len(rows(store, 'semantic_versions')) == 1


def test_capacity_never_evicts_core_and_allows_existing_key_update(store):
    first = apply(store)
    apply(store, '请不要对我开玩笑', 'boundary')
    for i in range(MAX_CURRENT_ROWS - 2):
        apply(store, f'Confirm my plan: task-{i}; status: planned', f'p{i}')
    before = store.semantic_context('u1', 'a1')
    assert len(before) == MAX_CURRENT_ROWS
    with pytest.raises(SemanticCapacityError):
        apply(store, 'Confirm my plan: excess; status: planned', 'excess')
    assert store.semantic_context('u1', 'a1') == before
    assert len(rows(store, 'semantic_sources')) == MAX_CURRENT_ROWS
    updated = apply(store, '请叫我小王', 'rename')['changes'][0]
    assert updated['supersedes'] == first['changes'][0]['change_id']
    apply(store, 'Confirm my plan: task-0; status: retracted', 'retract')
    with pytest.raises(SemanticCapacityError):
        apply(store, 'Confirm my plan: excess; status: planned', 'excess')
    assert len(store.semantic_context('u1', 'a1')) == MAX_CURRENT_ROWS
    assert any(r['category'] == 'boundary' for r in store.semantic_context('u1', 'a1'))


@pytest.mark.parametrize('text,expected', [
    ('请记住：我现在喝咖啡', True),
    ('[用户偏好] 我现在喝咖啡', True),
    ('[用户偏好]：我现在喝咖啡', True),
    ('[用户偏好] 我现在不喝咖啡了', False),
    ('请记住：我喝咖啡', True),
    ('请记住：我今天喝咖啡', False),
    ('请记住：我最近喝咖啡', False),
    ('请记住：我现在喝茶', False),
    ('请记住：我现在喜欢咖啡', False),
    ('我现在喝咖啡', True),
    ('[用户偏好] 用户现在喝咖啡', True),
    ('[用户偏好] “我现在喝咖啡”', False),
    ('[用户偏好] 我现在喝咖啡，开个玩笑', False),
    ('[用户偏好] 我现在喝咖啡\n我喜欢茶', False),
    ('[用户偏好] I now drink coffee', False),
])
def test_conflict_filter_exact_attributes_and_qualifiers(store, text, expected):
    apply(store, '请记住：我现在不喝咖啡了')
    current = store.semantic_context('u1', 'a1')
    assert memory_conflicts(text, current) is expected
    assert memory_conflicts(text, []) is False
    assert memory_conflicts(text, current + current) is False
    assert len(rows(store, 'semantic_versions')) == 1


def test_conflict_filter_english_retraction_and_nickname(store):
    apply(store, 'Please remember: I dislike coffee')
    assert memory_conflicts('[用户偏好] I like coffee', store.semantic_context('u1', 'a1'))
    apply(store, '撤回称呼设置', 'm2')
    assert memory_conflicts('[用户偏好] 请叫我小林', store.semantic_context('u1', 'a1'))
    apply(store, '请叫我小王', 'm3')
    current = store.semantic_context('u1', 'a1')
    assert memory_conflicts('请叫我小林', current)
    assert not memory_conflicts('请叫我小王', current)
    assert not memory_conflicts('撤回称呼设置', current)
    assert not memory_conflicts(None, current)


@pytest.mark.parametrize('kwargs', [dict(gen=True), dict(gen=-1), dict(mid=''),
                                  dict(mid='x' * 257), dict(at=''), dict(at='x' * 129)])
def test_invalid_metadata_is_write_free(store, kwargs):
    with pytest.raises(ValueError):
        apply(store, **kwargs)
    assert rows(store, 'semantic_sources') == []


@pytest.mark.parametrize('user', ['visitor123', 'guest-user', 'guest', 'anonymous', '游客'])
def test_stable_ids_are_not_guessed_to_be_guests(store, user):
    assert apply(store, user=user)['status'] == 'persisted'
    assert len(store.semantic_context(user, 'a1')) == 1
