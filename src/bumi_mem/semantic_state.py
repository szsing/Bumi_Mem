"""Conservative, model-free semantic state; not a general natural-language parser.

Parent integration contract
---------------------------
* Add SemanticStateStoreMixin to SQLiteSessionStore; its existing _connect(write=...)
  must yield sqlite3.Row connections, BEGIN IMMEDIATE for writes, and roll back
  on errors. Call create_semantic_tables(conn) AFTER creating memory_scopes in
  the parent migration. This function does not commit or change user_version.
* apply_semantic_turn(user_id, agent_id, source_message_id, text, observed_at,
  generation) takes ONLY the immutable original CURRENT USER message, never a
  prompt, retrieval/L3/shared reference, summary, assistant output or rewritten
  text. The caller must authenticate ownership/role and bind generation before
  calling: a string-only API cannot establish provenance or detect a bare joke.
* Receipt status is not_requested or persisted; changes contain change_id,
  category, attribute, qualifier, state_key, value, status, version, supersedes,
  source_message_id, source_role, source_text, observed_at and scope. Receipts
  are immutable application-time snapshots, not a fresh current-state query.
  Only recognized changes get a source ledger entry; not_requested is write-free.
  Retrying a source ID with different original text/time raises
  SemanticSourceConflict. Persistence errors propagate (never report success).
* semantic_context(user_id, agent_id) returns current rows for the live scope,
  including retracted/cancelled/completed status tombstones. It does not write,
  initialize, expire, resume plans, execute actions or schedule reminders.
* invalidate_semantic_scope(conn, user_id, agent_id) PURGES this pair's semantic
  sources/history/current rows, without committing, for explicit full deletion.
  The parent MUST also advance memory_scopes.generation in the same transaction
  so stale callers cannot recreate erased data. Ordinary retractions do NOT purge.
* Only identity.is_guest's reserved __guest__ identity is forbidden; other stable
  IDs are not guessed to be guests. IDs use identity.require_pair.

Accepted grammar (whole message, optionally one final period):
  请叫我小林 / 以后叫我林老师 / Please call me Alice
  今天叫我小林 / 别再叫我宝宝，我不喜欢 / 最近别聊工作
  我现在住在上海 / 我在北京工作
  撤回称呼设置 / Retract my nickname preference
  请不要对我开玩笑 / 请不要评价我的外貌
  撤回互动边界：开玩笑 / 撤回互动边界：评价外貌
  Please do not joke with me / Please do not comment on my appearance
  Retract my boundary: jokes / Retract my boundary: appearance
  请记住：我[今天|最近|现在]喜欢/不喜欢咖啡/茶/辣食/甜食
  请记住：我[今天|最近|现在]喝/不喝咖啡/茶[了]
  (我确认： may replace 请记住：; the prefix and qualifiers may be absent)
  Please remember: I [today|recently|now] like/dislike coffee/tea/spicy food/sweets
  Please remember: I [today|recently|now] drink/do not drink coffee/tea
  撤回偏好：[今天|最近|现在]咖啡/茶/辣食/甜食
  撤回饮用设置：[今天|最近|现在]咖啡/茶
  Retract my preference: [today|recently|now] coffee/tea/spicy food/sweets
  Retract my drinking setting: [today|recently|now] coffee/tea
  确认计划：晨跑；状态：计划中/进行中/已完成/已取消/已撤回
  确认我的承诺：晨跑；状态：已承诺/进行中/已完成/已取消/已撤回
  确认机器人承诺：陪你练习；状态：已完成
  record_assistant_promise handles actually spoken 我答应你陪你练习 separately
  Confirm my plan: morning-run; status: planned/active/completed/cancelled/retracted
  Confirm my promise: morning-run; status: promised/active/completed/cancelled/retracted

Limits: input grammar <=512 characters; values <=240 (the initial grammar is
stricter: names <=50, plan/promise labels <=32); source IDs <=256, timestamps
<=128, owner IDs <=128. Context holds at most 64 exact keys per live scope,
INCLUDING retracted/completed rows. New keys at capacity raise SemanticCapacityError
atomically; existing-key updates still work. No silent eviction or read truncation.
History is not truncated. Unsupported/oversized grammar returns not_requested.

Plan/promise labels are opaque short identifiers, NOT parsed task descriptions.
Nickname identifiers are similarly bounded. Quotes, reporting, questions,
conditional/joking markers, multiline/multiple requests and unrecognized forms
are rejected. No fuzzy matches, pronoun resolution, arbitrary personal facts,
general assistant-promise inference, relative-time expiry, forget-X parsing,
inferred contradictions, or cross-language entity resolution. Explicit bare
finite-object preference assertions are supported. Today/recently qualifiers
are kept literally (empty means unspecified, NOT permanent); 现在/now denotes
a current update on the unqualified key, never on a separate today key. Sources
and observed_at are stored exactly as supplied; time normalization/order belongs
upstream. Application order, not timestamp guessing, determines versions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
import sqlite3
from uuid import uuid4

from bumi_mem.identity import is_guest, require_pair


MAX_CURRENT_ROWS = 64    # 每个活跃作用域的最大当前状态行数（含已撤回/已完成）
MAX_SOURCE_CHARS = 512   # 输入语法最大字符数
MAX_VALUE_CHARS = 240    # 值字段最大字符数


class SemanticScopeError(ValueError):
    """语义作用域错误：陌生用户、过期代数或删除待处理时的访问。"""


class SemanticCapacityError(ValueError):
    """语义容量错误：新状态键将超过非驱逐的当前上下文上限。"""


class SemanticSourceConflict(ValueError):
    """语义源冲突：幂等键被复用于不同的不可变源输入。"""


@dataclass(frozen=True)
class SemanticChange:
    category: str
    attribute: str
    qualifier: str
    value: str | None
    status: str = 'confirmed'

    @property
    def state_key(self) -> str:
        return json.dumps([self.category, self.attribute, self.qualifier],
                          ensure_ascii=False, separators=(',', ':'))


# 刻意有限的对象/动作。不要扩展为宽松的文本尾部
_ZH_OBJECT = r'(咖啡|绿茶|红茶|茶|辣食|甜食)'
_EN_OBJECT = r'(coffee|tea|spicy food|sweets)'
_ZH_TIME = r'(今天|最近|现在)?'
_EN_TIME = r'(?:(today|recently|now) )?'
_LABEL = r'[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,31}'
_NAME = r'(?:[\u4e00-\u9fff]{1,12}|[A-Za-z]{1,16}(?: [A-Za-z]{1,16}){0,2})'
_REJECT = re.compile(
    r'[\r\n"\'“”‘’「」『』`<>?？]|'
    r'但是|不过|并且|而且|以及|或者|哈哈|开个玩笑|开玩笑的|只是玩笑|假如|如果|也许|可能|听说|据说|他说|她说|说过|比如|例如|转述|引用|'
    r'\b(?:joking|joke only|kidding|maybe|perhaps|if|said|says|reported|quote|example)\b', re.I)


def parse_semantic_turn(text: str) -> tuple[SemanticChange, ...]:
    """Pure exact-form recognizer. Returns zero or one change, never scans substrings."""
    if not isinstance(text, str):
        raise TypeError('text must be the original user string')
    if len(text) > MAX_SOURCE_CHARS or _REJECT.search(text):
        return ()
    body = text.strip()
    if body.endswith(('.', '。')):
        body = body[:-1]
    match = re.fullmatch(r'(?:以后)?(?:请)?叫我(' + _NAME + r')', body)
    if match is None:
        match = re.fullmatch(r'Please call me (' + _NAME + r')', body, re.I)
    if match:
        # 名字槽是不透明的，但明显的替代/助词不是名字
        if re.search(r'或者|还是|随便|也行|都可以|哈哈|谢谢|不要|请|[吧吗呢么]$|\b(?:or|and|not|please)\b', match[1], re.I):
            return ()
        return (SemanticChange('core', 'nickname', '', match[1]),)
    match = re.fullmatch(r'(今天|最近)(?:请)?叫我(' + _NAME + r')', body)
    if match:
        return (SemanticChange('core', 'nickname', match[1], match[2]),)
    match = re.fullmatch(r'(今天|最近|以后)?(?:请)?(?:不要|别再|别)叫我(' + _NAME + r')(?:[，,]我不喜欢)?', body)
    if match:
        qualifier = match[1] if match[1] in ('今天', '最近') else ''
        return (SemanticChange('boundary', 'forbidden_name:' + match[2], qualifier, 'disallowed'),)
    match = re.fullmatch(r'(今天|最近|以后)?(?:请)?(?:不要|别再|别)(?:和我|跟我)?(?:聊|谈)(工作|外貌|家庭|感情|收入|体重)(?:的事)?', body)
    if match:
        qualifier = match[1] if match[1] in ('今天', '最近') else ''
        return (SemanticChange('boundary', 'topic:' + match[2], qualifier, 'disallowed'),)
    match = re.fullmatch(r'(?:请记住[：:]|我确认[：:])?我(?:现在)?住在([\u4e00-\u9fff]{2,16})', body)
    if match:
        return (SemanticChange('personal', 'residence', '', match[1]),)
    match = re.fullmatch(r'(?:请记住[：:]|我确认[：:])?我(?:现在)?在([\u4e00-\u9fff]{2,16})工作', body)
    if match:
        return (SemanticChange('personal', 'work_location', '', match[1]),)
    fixed = {
        '撤回称呼设置': ('core', 'nickname', None, 'retracted'),
        'retract my nickname preference': ('core', 'nickname', None, 'retracted'),
        '请不要对我开玩笑': ('boundary', 'jokes', 'disallowed', 'confirmed'),
        '请不要评价我的外貌': ('boundary', 'appearance', 'disallowed', 'confirmed'),
        'please do not joke with me': ('boundary', 'jokes', 'disallowed', 'confirmed'),
        'please do not comment on my appearance': ('boundary', 'appearance', 'disallowed', 'confirmed'),
        '撤回互动边界：开玩笑': ('boundary', 'jokes', None, 'retracted'),
        '撤回互动边界：评价外貌': ('boundary', 'appearance', None, 'retracted'),
        'retract my boundary: jokes': ('boundary', 'jokes', None, 'retracted'),
        'retract my boundary: appearance': ('boundary', 'appearance', None, 'retracted'),
    }
    if body.lower() in fixed:
        category, attribute, value, status = fixed[body.lower()]
        return (SemanticChange(category, attribute, '', value, status),)
    match = re.fullmatch(r'(?:请记住：|我确认：)?我' + _ZH_TIME + r'(喜欢|不喜欢)' + _ZH_OBJECT, body)
    if match:
        return (SemanticChange('preference', match[3], match[1] if match[1] not in (None, '现在', 'now') else '',
                               'like' if match[2] == '喜欢' else 'dislike'),)
    match = re.fullmatch(r'(?:请记住：|我确认：)?我' + _ZH_TIME + r'(喝|不喝)(咖啡|茶)(?:了)?', body)
    if match:
        return (SemanticChange('personal', 'drinks:' + match[3], match[1] if match[1] not in (None, '现在', 'now') else '',
                               'yes' if match[2] == '喝' else 'no'),)
    match = re.fullmatch(r'Please remember: I ' + _EN_TIME + r'(like|dislike) ' + _EN_OBJECT, body, re.I)
    if match:
        return (SemanticChange('preference', match[3].lower(), match[1] if match[1] not in (None, '现在', 'now') else '', match[2].lower()),)
    match = re.fullmatch(r'Please remember: I ' + _EN_TIME + r'(drink|do not drink) (coffee|tea)', body, re.I)
    if match:
        return (SemanticChange('personal', 'drinks:' + match[3].lower(), match[1] if match[1] not in (None, '现在', 'now') else '',
                               'yes' if match[2].lower() == 'drink' else 'no'),)
    match = re.fullmatch(r'(撤回偏好：|撤回饮用设置：)' + _ZH_TIME + _ZH_OBJECT, body)
    if match:
        personal = match[1] == '撤回饮用设置：'
        if personal and match[3] not in ('咖啡', '茶'):
            return ()
        return (SemanticChange('personal' if personal else 'preference',
                               ('drinks:' if personal else '') + match[3],
                               match[2] if match[2] not in (None, '现在', 'now') else '', None, 'retracted'),)
    match = re.fullmatch(r'Retract my (preference|drinking setting): ' + _EN_TIME + _EN_OBJECT, body, re.I)
    if match:
        personal = match[1].lower() == 'drinking setting'
        obj = match[3].lower()
        if personal and obj not in ('coffee', 'tea'):
            return ()
        return (SemanticChange('personal' if personal else 'preference',
                               ('drinks:' if personal else '') + obj,
                               match[2] if match[2] not in (None, '现在', 'now') else '', None, 'retracted'),)
    match = re.fullmatch(r'确认(计划|我的承诺|机器人承诺)：(' + _LABEL + r')；状态：(计划中|已承诺|进行中|已完成|已取消|已撤回)', body)
    if match:
        category = {'计划': 'plan', '我的承诺': 'user_promise', '机器人承诺': 'robot_promise'}[match[1]]
        status = {'计划中': 'planned', '已承诺': 'promised', '进行中': 'active',
                  '已完成': 'completed', '已取消': 'cancelled', '已撤回': 'retracted'}[match[3]]
        label = match[2]
    else:
        match = re.fullmatch(r'Confirm my (plan|promise): (' + _LABEL + r'); status: '
                             r'(planned|promised|active|completed|cancelled|retracted)', body, re.I)
        if not match:
            return ()
        category = 'plan' if match[1].lower() == 'plan' else 'user_promise'
        label, status = match[2], match[3].lower()
    if (category == 'plan' and status == 'promised') or (category in ('user_promise', 'robot_promise') and status == 'planned'):
        return ()
    return (SemanticChange(category, label, '', label, status),)


def _owner(user_id: str, agent_id: str) -> None:
    require_pair(user_id, agent_id)
    if is_guest(user_id):
        raise SemanticScopeError('guest semantic state is forbidden')


class SemanticStateMixin:
    """SQLite-only persistence using the parent's transaction context manager."""

    @staticmethod
    def create_semantic_tables(conn: sqlite3.Connection) -> None:
        # execute (not executescript): preserve the parent's migration transaction.
        statements = (
            """CREATE TABLE IF NOT EXISTS semantic_withdrawals (
                user_id TEXT NOT NULL, agent_id TEXT NOT NULL, source_message_id TEXT NOT NULL,
                PRIMARY KEY(user_id,agent_id,source_message_id))""",
            """CREATE TABLE IF NOT EXISTS semantic_sources (
                user_id TEXT NOT NULL, agent_id TEXT NOT NULL, generation INTEGER NOT NULL,
                source_message_id TEXT NOT NULL, source_role TEXT NOT NULL CHECK(source_role IN ('user','assistant')),
                source_text TEXT NOT NULL, observed_at TEXT NOT NULL, receipt_json TEXT NOT NULL,
                PRIMARY KEY(user_id,agent_id,generation,source_message_id))""",
            """CREATE TABLE IF NOT EXISTS semantic_versions (
                change_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                generation INTEGER NOT NULL, state_key TEXT NOT NULL, category TEXT NOT NULL,
                attribute TEXT NOT NULL, qualifier TEXT NOT NULL, value TEXT, status TEXT NOT NULL,
                version INTEGER NOT NULL CHECK(version>0), supersedes TEXT,
                source_message_id TEXT NOT NULL,
                UNIQUE(user_id,agent_id,generation,state_key,version),
                UNIQUE(user_id,agent_id,generation,state_key,change_id),
                FOREIGN KEY(user_id,agent_id,generation,source_message_id)
                    REFERENCES semantic_sources(user_id,agent_id,generation,source_message_id))""",
            """CREATE TABLE IF NOT EXISTS semantic_current (
                user_id TEXT NOT NULL, agent_id TEXT NOT NULL, generation INTEGER NOT NULL,
                state_key TEXT NOT NULL, change_id TEXT NOT NULL,
                PRIMARY KEY(user_id,agent_id,generation,state_key),
                FOREIGN KEY(user_id,agent_id,generation,state_key,change_id)
                    REFERENCES semantic_versions(user_id,agent_id,generation,state_key,change_id))""",
            """CREATE TRIGGER IF NOT EXISTS semantic_sources_immutable
                BEFORE UPDATE ON semantic_sources BEGIN
                SELECT RAISE(ABORT, 'semantic source is immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS semantic_versions_immutable
                BEFORE UPDATE ON semantic_versions BEGIN
                SELECT RAISE(ABORT, 'semantic version is immutable'); END""",
        )
        for statement in statements:
            conn.execute(statement)

    def apply_semantic_turn(self, user_id: str, agent_id: str, source_message_id: str,
                            text: str, observed_at: str, generation: int) -> dict:
        return self._apply_semantic_source(user_id, agent_id, source_message_id, text,
                                           observed_at, generation, 'user', parse_semantic_turn(text))

    def record_assistant_promise(self, user_id: str, agent_id: str, source_message_id: str,
                                 text: str, observed_at: str, generation: int) -> dict:
        # 仅调用方提供的实际说出的话，不是模型草稿或用户引用
        match = None if _REJECT.search(text) else re.fullmatch(r'(?:我答应你|我承诺)([^。！？!?\n]{1,240})[。.]?', text.strip())
        changes = () if match is None else (SemanticChange('robot_promise', match[1], '', match[1], 'promised'),)
        return self._apply_semantic_source(user_id, agent_id, source_message_id, text,
                                           observed_at, generation, 'assistant', changes)

    def _apply_semantic_source(self, user_id: str, agent_id: str, source_message_id: str,
                               text: str, observed_at: str, generation: int, source_role: str,
                               parsed: tuple[SemanticChange, ...]) -> dict:
        _owner(user_id, agent_id)
        if (not isinstance(source_message_id, str) or not source_message_id.strip()
                or len(source_message_id) > 256):
            raise ValueError('source_message_id must be nonempty')
        if (not isinstance(observed_at, str) or not observed_at.strip()
                or len(observed_at) > 128):
            raise ValueError('observed_at must be an upstream-normalized nonempty string')
        if type(generation) is not int or generation < 0:
            raise ValueError('generation must be a nonnegative integer')
        with self._connect(write=bool(parsed)) as conn:
            scope = conn.execute('SELECT generation,delete_pending FROM memory_scopes '
                                 'WHERE user_id=? AND agent_id=?', (user_id, agent_id)).fetchone()
            if scope is None:
                if generation != 0:
                    raise SemanticScopeError('unknown semantic generation')
                if parsed:
                    conn.execute('INSERT INTO memory_scopes(user_id,agent_id) VALUES(?,?)', (user_id, agent_id))
            elif scope['generation'] != generation or scope['delete_pending']:
                raise SemanticScopeError('stale generation or deletion pending')
            if conn.execute('SELECT 1 FROM semantic_withdrawals WHERE user_id=? AND agent_id=? AND source_message_id=?',
                            (user_id, agent_id, source_message_id)).fetchone():
                raise SemanticScopeError('source has been withdrawn')
            owner = (user_id, agent_id, generation)
            source_key = (*owner, source_message_id)
            old = conn.execute('SELECT * FROM semantic_sources WHERE user_id=? AND agent_id=? '
                               'AND generation=? AND source_message_id=?', source_key).fetchone()
            if old is not None:
                if (old['source_text'], old['observed_at']) != (text, observed_at):
                    raise SemanticSourceConflict('source_message_id reused with changed text or observed_at')
                return json.loads(old['receipt_json'])
            source = dict(user_id=user_id, agent_id=agent_id, generation=generation,
                          source_message_id=source_message_id, source_role=source_role,
                          source_text=text, observed_at=observed_at)
            changes = []
            for change in parsed:
                previous = conn.execute('SELECT v.change_id,v.version FROM semantic_current c '
                                        'JOIN semantic_versions v ON v.change_id=c.change_id '
                                        'WHERE c.user_id=? AND c.agent_id=? AND c.generation=? '
                                        'AND c.state_key=?', (*owner, change.state_key)).fetchone()
                if previous is None:
                    count = conn.execute('SELECT COUNT(*) FROM semantic_current WHERE user_id=? '
                                         'AND agent_id=? AND generation=?', owner).fetchone()[0]
                    if count >= MAX_CURRENT_ROWS:
                        raise SemanticCapacityError('semantic current-state limit is 64; no rows were evicted')
                if change.value is not None and len(change.value) > MAX_VALUE_CHARS:
                    raise SemanticCapacityError('semantic value exceeds 240 characters')
                changes.append(dict(asdict(change), **source, state_key=change.state_key,
                                    change_id=uuid4().hex,
                                    version=previous['version'] + 1 if previous else 1,
                                    supersedes=previous['change_id'] if previous else None))
            receipt = dict(status='persisted' if changes else 'not_requested',
                           changes=changes, source=source)
            if not changes:
                return receipt
            conn.execute('INSERT INTO semantic_sources VALUES(?,?,?,?,?,?,?,?)',
                         (*source_key, source_role, text, observed_at, json.dumps(receipt, ensure_ascii=False)))
            for row in changes:
                conn.execute('INSERT INTO semantic_versions '
                             '(change_id,user_id,agent_id,generation,state_key,category,attribute,qualifier,'
                             'value,status,version,supersedes,source_message_id) '
                             'VALUES(:change_id,:user_id,:agent_id,:generation,:state_key,:category,:attribute,'
                             ':qualifier,:value,:status,:version,:supersedes,:source_message_id)', row)
                conn.execute('INSERT INTO semantic_current VALUES(?,?,?,?,?) '
                             'ON CONFLICT(user_id,agent_id,generation,state_key) '
                             'DO UPDATE SET change_id=excluded.change_id',
                             (*owner, row['state_key'], row['change_id']))
            # 父集成：发布状态、使冲突引用失效、在同一提交中撤销缓存上下文（仅新写入时）
            if changes and hasattr(self, '_bump_context_epoch'):
                refs = conn.execute("SELECT ref_id,payload_json FROM shared_memory_refs WHERE user_id=? AND agent_id=? AND status='active'",
                                    (user_id, agent_id)).fetchall()
                for ref in refs:
                    if memory_conflicts(str(json.loads(ref['payload_json']).get('memory', '')), changes):
                        conn.execute("UPDATE shared_memory_refs SET status='superseded' WHERE ref_id=?", (ref['ref_id'],))
                self._bump_context_epoch(conn, agent_id)
        return receipt

    def semantic_context(self, user_id: str, agent_id: str) -> list[dict]:
        _owner(user_id, agent_id)
        with self._connect() as conn:
            rows = conn.execute('SELECT v.*,s.source_role,s.source_text,s.observed_at '
                                'FROM semantic_current c JOIN semantic_versions v ON v.change_id=c.change_id '
                                'JOIN semantic_sources s ON s.user_id=v.user_id AND s.agent_id=v.agent_id '
                                'AND s.generation=v.generation AND s.source_message_id=v.source_message_id '
                                'JOIN memory_scopes m ON m.user_id=c.user_id AND m.agent_id=c.agent_id '
                                'AND m.generation=c.generation '
                                'WHERE c.user_id=? AND c.agent_id=? AND m.delete_pending=0 '
                                'AND NOT EXISTS (SELECT 1 FROM semantic_withdrawals w WHERE w.user_id=c.user_id '
                                'AND w.agent_id=c.agent_id AND w.source_message_id=v.source_message_id) '
                                'ORDER BY v.category,v.state_key', (user_id, agent_id)).fetchall()
            return [dict(row) for row in rows]

    def semantic_history(self, user_id: str, agent_id: str, limit: int = 64) -> list[dict]:
        """Explicit history queries only; never replace the always-current core block."""
        _owner(user_id, agent_id)
        with self._connect() as conn:
            rows = conn.execute('SELECT v.*,s.source_role,s.source_text,s.observed_at '
                                'FROM semantic_versions v JOIN semantic_sources s USING(user_id,agent_id,generation,source_message_id) '
                                'JOIN memory_scopes m USING(user_id,agent_id,generation) '
                                'WHERE v.user_id=? AND v.agent_id=? AND m.delete_pending=0 '
                                'AND NOT EXISTS (SELECT 1 FROM semantic_withdrawals w WHERE w.user_id=v.user_id '
                                'AND w.agent_id=v.agent_id AND w.source_message_id=v.source_message_id) '
                                'ORDER BY s.observed_at DESC,v.version DESC LIMIT ?',
                                (user_id, agent_id, min(max(limit, 1), 64))).fetchall()
            return [dict(r, fact_status='historical_version') for r in rows]

    @staticmethod
    def invalidate_semantic_scope(conn: sqlite3.Connection, user_id: str, agent_id: str) -> None:
        # 清理可删除旧版访客数据；与读/写不同，不需要身份验证为认证所有者
        require_pair(user_id, agent_id)
        for table in ('semantic_current', 'semantic_versions', 'semantic_sources', 'semantic_withdrawals'):
            conn.execute(f'DELETE FROM {table} WHERE user_id=? AND agent_id=?', (user_id, agent_id))


def create_semantic_tables(conn: sqlite3.Connection) -> None:
    """Module-level migration entry point; no implicit commit."""
    SemanticStateMixin.create_semantic_tables(conn)


def invalidate_semantic_scope(conn: sqlite3.Connection, user_id: str, agent_id: str) -> None:
    """Module-level full-delete hook; parent must advance generation atomically."""
    SemanticStateMixin.invalidate_semantic_scope(conn, user_id, agent_id)


def memory_conflicts(text: str, current_rows: list[dict]) -> bool:
    """Recognize exact-key superseded L3 statements, never infer contradictions.

    Supply ONLY rows from semantic_context for the authenticated owner/agent.
    Text is data, not an update: this helper never persists or returns facts to
    ingest. Accepts the turn grammar, optionally prefixed with [用户偏好], plus
    '[用户偏好] 我喜欢咖啡', '[用户偏好] 我现在不喝咖啡了' and the equivalent
    '[用户偏好] I like coffee' / 'I now do not drink coffee' forms. Prefix may
    have one following colon (ASCII or Chinese). Unsupported unlabelled assertions,
    noncanonical third-person summaries, paraphrases, multiple statements, unknown attributes,
    different/lost qualifiers and inconsistent duplicate current keys return
    False. Retractions invalidate positive claims, not vice versa automatically.
    No cross-language object matching; explicit status changes compare literally.
    """
    if not isinstance(text, str):
        return False
    candidate = text.strip()
    prefix = next((p for p in ('[用户偏好]', '[用户事实]', '[用户边界]', '[用户计划]', '[机器人承诺]') if candidate.startswith(p)), None)
    if prefix is not None:
        candidate = candidate[len(prefix):].lstrip()
        if candidate.startswith((':', '：')):
            candidate = candidate[1:].lstrip()
        if candidate.startswith('用户'):
            candidate = '我' + candidate[2:]
        parsed = parse_semantic_turn(candidate)
        if not parsed and candidate.startswith('我'):
            parsed = parse_semantic_turn('请记住：' + candidate)
        elif not parsed and candidate.lower().startswith('i '):
            parsed = parse_semantic_turn('Please remember: ' + candidate)
    else:
        parsed = parse_semantic_turn(candidate)
    if not parsed:
        return False
    change = parsed[0]
    matching = [row for row in current_rows
                if row.get('state_key') == change.state_key
                and row.get('category') == change.category
                and row.get('qualifier') == change.qualifier]
    if len(matching) != 1:
        return False
    current = matching[0]
    if 'value' not in current or 'status' not in current:
        return False
    # A historical retraction contains no positive fact to suppress.
    if change.status == 'retracted':
        return False
    if current['status'] == 'retracted':
        return True
    return (current['value'], current['status']) != (change.value, change.status)



# 父面向名称；保留较短名称供独立用户/测试使用
SemanticStateStoreMixin = SemanticStateMixin