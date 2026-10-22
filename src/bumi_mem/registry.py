# 本次改动（2026-10-21）：新增注册表持久化、陌生人映射、身份切换、策略及长期引用失效处理。
"""Trusted registration is separate from entity extraction and caller-supplied names."""
import json
from bumi_mem.identity import GUEST_USER_ID, require_id


class RegistryStoreMixin:
    def create_registry_tables(self, conn):
        for sql in (
            '''CREATE TABLE IF NOT EXISTS registered_users (
                agent_id TEXT NOT NULL,user_id TEXT NOT NULL,display_name TEXT NOT NULL,
                aliases_json TEXT NOT NULL,active INTEGER NOT NULL DEFAULT 1,
                version INTEGER NOT NULL DEFAULT 1,PRIMARY KEY(agent_id,user_id))''',
            '''CREATE TABLE IF NOT EXISTS isolation_policy (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),enabled INTEGER NOT NULL)''',
            '''CREATE TABLE IF NOT EXISTS isolation_actor (
                agent_id TEXT PRIMARY KEY,user_id TEXT NOT NULL)''',
        ):
            conn.execute(sql)

    def isolation_enabled(self):
        with self._connect() as conn:
            row = conn.execute('SELECT enabled FROM isolation_policy WHERE singleton=1').fetchone()
            return bool(row and row[0])

    def enable_isolation(self):
        with self._connect(write=True) as conn:
            conn.execute('INSERT OR REPLACE INTO isolation_policy VALUES(1,1)')
            self._registry_invalidate(conn)

    def _registry_invalidate(self, conn):
        agents = {r[0] for r in conn.execute('SELECT agent_id FROM sessions')}
        conn.execute("UPDATE shared_memory_refs SET status='deleted' WHERE status='active'")
        for agent in agents:
            self._bump_context_epoch(conn, agent)

    def register_user(self, *, agent_id, user_id, display_name, aliases=(), active=True):
        require_id(agent_id, 'agent_id');require_id(user_id)
        if user_id==GUEST_USER_ID:
            raise ValueError('guest_cannot_register')
        names = [display_name, *aliases]
        if len(names)>21 or any(not isinstance(n,str) or not n.strip() or len(n)>128 for n in names):
            raise ValueError('invalid_registered_name')
        with self._connect(write=True) as conn:
            conn.execute('''INSERT INTO registered_users(agent_id,user_id,display_name,aliases_json,active)
                VALUES(?,?,?,?,?) ON CONFLICT(agent_id,user_id) DO UPDATE SET
                display_name=excluded.display_name,aliases_json=excluded.aliases_json,
                active=excluded.active,version=registered_users.version+1''',
                (agent_id,user_id,display_name.strip(),json.dumps(sorted(set(a.strip() for a in aliases)),ensure_ascii=False),int(active)))
            self._registry_invalidate(conn)
        return next(r for r in self.registered_users(agent_id,include_inactive=True) if r['user_id']==user_id)

    def registered_users(self, agent_id, *, include_inactive=False):
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM registered_users WHERE agent_id=? ORDER BY user_id',(agent_id,)).fetchall()
            return [dict(user_id=r['user_id'],display_name=r['display_name'],aliases=json.loads(r['aliases_json']),
                         active=bool(r['active']),version=r['version']) for r in rows if include_inactive or r['active']]

    def resolve_registered_user(self, user_id, agent_id):
        require_id(user_id)
        if not self.isolation_enabled():
            return user_id
        with self._connect() as conn:
            row = conn.execute('SELECT active FROM registered_users WHERE agent_id=? AND user_id=?',(agent_id,user_id)).fetchone()
            return user_id if row and row[0] else GUEST_USER_ID

    def isolation_switch(self, user_id, agent_id):
        if not self.isolation_enabled():
            return False
        with self._connect(write=True) as conn:
            row = conn.execute('SELECT user_id FROM isolation_actor WHERE agent_id=?',(agent_id,)).fetchone()
            changed = row is None or row[0]!=user_id
            if changed:
                conn.execute("UPDATE shared_memory_refs SET status='deleted' WHERE agent_id=? AND status='active'",(agent_id,))
                conn.execute('INSERT OR REPLACE INTO isolation_actor VALUES(?,?)',(agent_id,user_id))
                self._bump_context_epoch(conn,agent_id)
            return changed
