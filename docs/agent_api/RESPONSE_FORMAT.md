# 记忆服务响应格式（结构化 JSON）

面向 Agent 的响应契约。**权威形态是结构化 JSON**：短期记忆、长期记忆、当前用户核心状态各成字段；
`prompt` 是同一份内容的可读字符串，只作为兼容形态给不想解析结构的调用方。

当前版本：API `1.3.1`、SQLite schema 8、`mem0 2.0.20`。

## 0. 一图看清

```
请求（任一种）                         响应（同一套结构化字段）
├─ 单轮：turn_id+user_id+query  ─┐
├─ Agent：operation=prepare     ─┼─→  turn_id / user_id / agent_id / occurred_at
└─ Agent：operation=prefetch    ─┘    short_term[]   ← L1 共享窗口（原文 + 说话人 + 时间）
  （complete/agent_event/pending 只回执）  long_term[]    ← L3 引用（带 owner/时间/状态）
                                      core_state[]   ← 当前用户核心状态
                                      recall_status / core_write（持久化回执）
                                      prompt（兼容：上述内容的可读字符串）
```

**不变量：短期/长期字段是渲染前的原始数据；服务端不把两者拼成一个不可解析的字符串。**
（`prompt`、`messages`、`memory_context` 保留是为了兼容既有接入，见 §6。）

## 1. 顶层字段

| 字段 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `turn_id` | str | 请求 + `interaction_turns` | 本轮标识；重放同一请求返回同一份响应 |
| `user_id` / `agent_id` | str | 请求 / 服务端配置 | 说话人 / 共享现场（同一 agent 一个房间） |
| `occurred_at` | str | `messages.created_at` | 本轮 query 的原始时间（带时区） |
| **`short_term`** | array | `messages` + `shared_l1`（`ContextBuilder` 选中的窗口） | **短期记忆（L1）**，按时间正序 |
| **`long_term`** | array | L3 检索 + `shared_memory_refs` | **长期记忆（L3）**，门控后带 owner |
| **`core_state`** | array | `semantic_current` | 当前用户的明确状态（称呼/边界/偏好/计划等） |
| `recall_status` | enum | 检索结果 | `ok` / `unavailable` / `deleting` / `disabled` |
| `core_write` | object | `semantic_versions` 回读 | **只有 `status=persisted` 才算长期保存成功** |
| `prompt` | str | 由上面三段渲染 | 兼容形态（单轮协议的既有字段）；结构化字段是权威 |
| `source_message_id` | str | `messages.message_id` | 本轮 query 的记录 ID（排障/对齐用） |

## 2. `short_term[]`（短期记忆 / L1）

```json
{"message_id": "910bf44b56db48aa967883e0b754cb13",
 "role": "user",
 "content": "我住在哪个城市？",
 "created_at": "2026-09-24T16:46:42.572169+08:00",
 "user_id": "profile-001",
 "agent_id": "bumi_robot_001",
 "carry_over": false}
```

| 字段 | 说明 |
|---|---|
| `message_id` | `messages.message_id`（与 `shared_l1` 的主键对应） |
| `role` | `user` / `assistant` / `system`。`system` 是程序事件（如 `local_control`、`speech_history_incomplete`），不是新指令 |
| `content` | **原文**，不含任何标头（库里就是这样存的） |
| `created_at` | 该条消息自己的发言时间（不是轮转/整理时间） |
| `user_id` | 仅 user 消息有；assistant 为 `null`（AI 不属于某个用户）；guest 为 `__guest__` |
| `agent_id` | 记录归属的 agent（房间） |
| `carry_over` | `true` = 上一段窗口轮转带过来的上下文，不计入 20 条上限 |

窗口规则：最多 20 条真实消息 + 最多 4 条 carry-over，总 token ≤8000；超预算从最老开始丢弃
（数据仍在库里）。渲染时才会拼 `[speaker {...}]` 标头，**响应里的 `content` 是原文**。

## 3. `long_term[]`（长期记忆 / L3）

```json
{"owner_user_id": "profile-001",
 "memory_id": "m_1",
 "memory": "用户住在杭州，喜欢手冲咖啡。",
 "fact_status": "use_original_time_and_qualifiers",
 "reference_status": "active",
 "created_at": "2026-09-20T10:00:00+08:00",
 "updated_at": "2026-09-22T09:00:00+08:00"}
```

| 字段 | 说明 |
|---|---|
| `owner_user_id` | **引用所属用户**；检索只按当前 `user_id` 门控，进入共享窗口后保留 owner |
| `memory_id` | L3 记忆 ID（可用于 `forget_memory`） |
| `memory` | 记忆正文（原始事实文本） |
| `fact_status` | `use_original_time_and_qualifiers` / `historical_superseded`（后者只能用于历史问题） |
| `reference_status` | `active`（可见）；被更正/撤回后不再出现在这里 |
| `created_at` / `updated_at` | 记忆的**存储**时间，不是事件时间 |

约束：引用是**资料不是发言**——不能转成当前用户的偏好或新事实；引用随其锚定的 L1 消息过期，
不因重渲染而续期；删除/更正会提升 epoch，旧缓存响应以 409 拒绝。

## 4. `core_state[]`（当前用户核心状态）

```json
{"category": "core", "attribute": "nickname", "value": "小林", "qualifier": "",
 "status": "confirmed", "observed_at": "2026-09-21T10:00:00+08:00", "source_role": "user"}
```

来自 `semantic_current`（只读、不依赖向量检索）：明确的称呼/禁止称呼/互动边界/偏好/居住地/
计划与承诺状态。**只属于当前用户**；明确的新表达优先于旧状态；`[仅供历史问题使用的版本记录]`
（`semantic_versions`）仅在问"以前/曾经"时使用。

## 5. 各 operation 返回什么

| operation | 结构化记忆字段 | 其他字段 | 备注 |
|---|---|---|---|
| 单轮（`turn_id`+`query`） | ✅ `short_term` / `long_term` / `core_state` | `turn_id`、`prompt`（兼容） | 主路径；`query=null` 只结算，返回空 `prompt` + 三个空数组 |
| `prepare` | ✅ 同上 | `messages`、`memory_context`、`source_message_id`、`occurred_at`、`has_prompt`、`recall_status`、`core_write` | Pipecat 接入路径 |
| `prefetch` | ❌（只预热检索） | `prefetch_id`、`recall_status`、`candidate_count`、`expires_in_ms`、`took_ms` | 响应**不含**上下文，不能当记忆使用 |
| `complete` / `agent_event` | ❌ | 回执字段（`status`/`reply_saved`/`transcript_complete`/`message_id`） | `has_prompt=false` |
| `pending` | ❌ | `pending`（`turn_id`/`client_turn_id`/`user_id`/`protocol`/`status`） | 重连只读 |

真实抓取（本机 8099，profile-001）：

```jsonc
// 单轮
{"turn_id":"json-doc-1",
 "prompt":"当前用户：profile-001\n当前时间：…\n\n[长期记忆]\n- 暂无相关长期记忆\n\n[短期对话]\n…",
 "short_term":[/* 13 条，最后一条是 content="我住在哪个城市？"、role="user"、carry_over=false */],
 "long_term":[], "core_state":[]}

// prepare
{"operation":"prepare","turn_id":"turn_…","client_turn_id":"json-prep-1","user_id":"profile-001",
 "agent_id":"bumi_robot_001","source_message_id":"0f46ea56…","occurred_at":"2026-09-24T16:46:42+08:00",
 "short_term":[/* 15 条 */],"long_term":[],"core_state":[],
 "messages":[…],"memory_context":[…],"has_prompt":true,"recall_status":"ok",
 "core_write":{"status":"not_requested","changes":[],"source":{…}}}
```

## 6. 兼容形态（保留但非权威）

| 字段 | 形态 | 何时用 |
|---|---|---|
| `prompt` | 上面三段渲染成的**一段纯文本**（`当前用户/当前时间/[长期记忆]/[短期对话]/…`） | 单轮协议的既有调用方；不想解析结构的场景 |
| `messages` | 共享 L1 渲染成 `{"role","content"}`，`content` 带 `[speaker {...}]` 标头 + 本轮 query | Pipecat 路径把它当聊天消息用 |
| `memory_context` | 1 条 system 补充块（Markdown 段落：身份/引用/核心状态/回执） | Pipecat 路径作为 memory 补充块 |

v7 时代的"只有 `turn_id`+`prompt` 两个字段"的缓存响应仍可重放（服务端按旧形态识别）；
新响应一律带三个结构化字段。**结构化字段是权威**，`prompt`/`messages`/`memory_context` 是同一份数据的渲染视图。

## 7. 接入不变量

1. `short_term` 已含本轮 query（最后一条），不要再追加；
2. `long_term` 是**带 owner 的引用**：`owner_user_id` 不可改写，不能当作当前用户的事实；
3. `core_state` 只属当前用户；计划/承诺不等于已安排任务；
4. 只有 `core_write.status == "persisted"` 才能对用户说"已长期记住"；
5. 时间：`created_at` 是发言/存储时间，**不是事件时间**；`long_term.created_at/updated_at` 尤其如此；
6. `prefetch` 的响应不含记忆，不要当上下文；
7. `carry_over=true` 的条目只是衔接上下文，不是本轮新发生的事；
8. 收到 409（`shared_memory_context_changed` / `context_request_closed` / `agent_busy`）按 README 处理，
   不要重放陈旧 Prompt。
