> 接口版本更新：单 HTTP 接口新增事务请求记录，最新主库 schema 为 v9。本文 v3 段落描述共享短期索引的迁移步骤，完整部署请参见 `../agent_api/README.md`。

# 共享短期上下文，隔离长期记忆

## 适用边界

同一个 SQLite 数据库、同一个 agent_id 代表一处共享现场。不同 user_id 进入同一现场时，看到同一短期消息序列。不同 agent_id 不共享；不同进程/设备要共享同一现场不能简单指向同一个 Qdrant 本地目录。当前仍是单应用进程、单 engine 持有 SDK 的部署方式，不提供跨进程对话锁或分布式会话。

未提供 user_id 的请求仍拒绝；可信上游可使用保留 ID `__guest__` 表示所有未确认说话人。该类别只访问共享 L1，不访问或生成个人 L3。启动预热仍与真实用户无关。

## 数据结构

新增 `shared_memory_refs` 绑定真实用户消息，保存门控后的 L3 引用；引用不作为发言、不进入抽取快照，随锚点退出并支持更正/删除失效。完整 v0.3 契约见 `../companion_memory/README.md`。

- `messages`：保存原始正文、role、user_id、agent_id 等。role=assistant 时 user_id 为空字符串，Prompt 不输出该字段，说话者只有 agent_id；后台依靠 session 关联了解整理来源。
- `shared_l1`：按 agent_id 保存有序的 message_id 引用及 context_only。主键 position 为 SQLite 分配的顺序，不依赖时钟排序新消息。
- `sessions`：仍按 user_id + agent_id 保留个人存储/整理会话；它不再决定前台能看见哪些短期消息。因此两位用户的 session_id 可以不同，但短期视图相同。
- `memory_jobs` / `memory_chunks`：仍归属单一 user_id + agent_id，删除代际、租约、重试及写入回执规则保持不变。

没有把共享对话伪装成一个公共 user_id，也没有取消任何 L3 过滤条件。

## Prompt 示例

当前请求来自 B，历史中 A 刚和 bot 对话：

```text
system: 当前身份是 B / bot；核心状态只属于 B；已通过所有者门控的 L3 引用按所有者标签共享。
user: [speaker {"user_id": "A", "agent_id": "bot", "speaker_role": "user"}]
      我刚才说周末想去公园。
assistant: [speaker {"agent_id": "bot", "speaker_role": "assistant"}]
           好，我们刚才在讨论周末去公园。
user: [speaker {"user_id": "B", "agent_id": "bot", "speaker_role": "user"}]
      你刚刚说什么？
```

role 保留为模型 API 的 user / assistant，归属使用 JSON 标头，消息正文原样保存。当前用户身份另在 system 中明确，不能将其他说话人的偏好当作当前用户偏好。新写消息的 token_count 包含身份标头；旧记录保留旧版近似 token 计数。

## 窗口和轮转

默认合计 20 条新消息，约 10 个问答轮次，不是每位用户各 10 轮。达到消息/token 边界或 `/new` 时：

1. 获取当前 agent 的进程内锁，串行化该 agent 下不同用户的完整问答。
2. SQLite 单事务冻结该 agent 下所有当前个人会话，并按用户各自的非 context-only、非 memory_excluded 消息生成独立任务。
3. 建立新的个人会话；共享窗口保留全局最近 `carryover_messages` 条有效消息引用（默认 20 条，约 10 轮），标为 context_only，构成跨会话滑动窗口。
4. 提交事务后唤醒后台 worker；后台慢写期间，新用户仍能读到共享 carry-over。
5. carry-over 仅供前台阅读，不重新整理、不写入后来的用户的长期快照。

L2 摘要也按个人快照生成，不能把混合现场摘要当作某个人的长期事实。执行任务前校验快照中的归属字段（兼容 v2 没有该字段的历史任务）。提取提示也明确禁止把转述其他人的事实当成目标用户事实；LLM 事实提取质量仍需真实场景验证，不宣称语义上绝不误提取。

`end_session()` / 自动轮转回执仍返回当前调用者的任务，其他参与者的任务可通过各自的 `memory_jobs()` 查看。空个人快照的任务直接标为 succeeded，不调用 LLM。

## 操作语义

| 操作 | 行为 |
|---|---|
| `/user B` | 改当前身份，不清空共享 L1；下一轮只检索 B 的 L3 |
| `/logout` | 清除当前身份，不删除共享短期历史 |
| `/session` | 显示共享 agent 的有效消息数、token 数、carry-over 数；owner-session 仍是当前用户的整理会话 |
| `/new` | 整个 agent 的窗口轮转，按用户分别提交整理 |
| `/discard` | 明确确认后清空整个 agent 的当前共享窗口，丢弃所有当前个人会话；既有后台任务不撤销 |
| `/quit` 的 save / discard | 分别采用上述整个 agent 的轮转 / 清空语义 |
| `/forget-all` | 仅当前用户 L3；不清空共享现场历史，同时旧消息标为不可重新提取 |
| `/remember` / `/read` / `/memories` / `/retry` | 仍为当前用户的个人操作，不纳入共享窗口 |

## 隐私含义

本设计有意共享真实说过的话：若 bot 在回答 A 时讲出了某条 A 的私有信息，这条已说出的回答也会成为 B 可见的短期记录。原始 L3 检索数组不会被存进共享表，但不能靠 L3 检索隔离保证“说出的秘密”不传播。如果产品需要私聊/房间隔离，应增加可信 room_id 或单独的私聊 agent，不要把不同房间复用同一个 agent_id。

## SQLite 升级

首次打开 v2 主库时自动升级到 v3：补 user_id / agent_id，从原 sessions 回填；仅将 active 会话的消息放入共享索引，保留 context_only，不重新引入已 discarded/冻结的历史。重开 v3 不重复导入。旧的消息、会话、任务和 L3 数据不删不改归属。

部署前停应用并用 SQLite backup API 或在数据库关闭后备份完整主库（含尚未合并的 WAL），再运行新版。升级后的库不能直接交回 v2 代码，旧版会拒绝更高 schema 版本；回滚应恢复备份。本次本地测试只使用临时数据库，没有连接或升级服务器业务库。

## 本地回归

服务器（Linux，旧框架对照基线）：

```bash
cd /data1/sunzhixing/Bumi_Mem
.venv/bin/python -m pytest -q
```

Windows 本地开发环境（历史）：

```powershell
cd D:\代码\Bumi_mem
.\.venv\Scripts\python.exe -m pytest -q
```

新增测试覆盖 A→B 共享、有说话人标头、L3 仍隔离、不同 agent 不共享、全局 20 条边界、carry-over 重启恢复和不重复提取、多人任务原子回滚、删除不影响他人任务、共享 discard、同 agent 并发串行化、v2 迁移和伪造任务归属拒绝。

本地测试使用 Fake provider / Fake L3 和真实 SQLite 验证逻辑，不等于真实模型提取准确率或 GPU 性能验收。
