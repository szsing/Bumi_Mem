> v0.3 / schema 6：门控后的 L3 引用进入共享 L1，新增陌生身份、时间来源与 core_write。详细行为以 [陪伴记忆契约](../companion_memory/README.md) 为准。

# 单接口记忆服务：Agent 接入与部署

对外仅保留一个业务接口：`POST /v1/memory/context`。旧 `/v1/memory/prompt`、`/v1/memory/reply` 路由已移除。健康检查、在线文档不是额外的对话业务接口。

## 1. 请求：三个必需业务含义与两个可选时间字段

```http
POST /v1/memory/context
Authorization: Bearer <BUMI_SERVICE_API_KEY>
Idempotency-Key: logical-request-001
Content-Type: application/json

{
  "user_id": "user_A",
  "previous_ai_reply": null,
  "query": "周末我想去公园。"
}
```

- user_id：当前 query 的说话人，由受信任 Agent 后端提供。
- previous_ai_reply：上一轮 agent 实际输出的回答。第一轮为 null 或空字符串。
- query：本轮用户问题；传 null 表示只补存最后一条 AI 回复，不开始新问题。
- agent_id：固定在服务端 BUMI_AGENT_ID，不要求每次传，也不允许请求体覆盖。
- Idempotency-Key：可选请求头，但可靠接入应使用稳定请求 ID。网络重试必须保持相同 key 和完整请求字段（包括可选时间）。不要用 query 文本作 key，也不要为一次重试生成新 key。

服务凭证只给受信任 Agent 后端，不是终端用户认证；我们不提供声纹验证。一个 agent_id 表示一个共享现场。不同独立测试现场使用不同 agent_id 和不同数据目录。

## 2. 处理顺序与说话人身份

```text
补存 previous_ai_reply（说话者是 agent_id）
    → 检查上一轮补齐后是否到达整理边界，必要时轮转共享 L1、按用户保存后台任务
    → 保存当前 user_id + query
    → 检索当前 user_id + agent_id 的个人 L3
    → 拼接共享短期窗口、个人长期记忆和当前输入
    → 返回 messages，外部 Agent 调用回答模型
```

A→B 的实际记录示例：

```text
user_A：周末我想去公园。
agent_001：去公园不错，可以提前看看天气。
user_B：你刚刚说什么？
```

A 的 user_id 已经在上次请求保存，本次不重复写 A 的问题，也不把 AI 回答标成 A/B 的用户发言。

**用户消息保存 user_id；AI 消息仅以 agent_id 标识说话者。** 原 messages 表因旧列的 NOT NULL 约束，非用户消息 user_id 保存为空字符串；Prompt 中 AI 标头完全不含 user_id 或 reply_to。

内部 session_id/轮次关联仍保留，用来判断后台整理的上下文来源。它不代表 AI 的说话人是某个用户。例如 A 说喜欢绿茶、B 说喜欢咖啡，仍分别提取到 A/B 的 L3；不会因为当前请求是 B 就把整段现场内容写到 B。

### 可选时间与陌生身份

- `query_occurred_at` / `previous_reply_occurred_at`：带偏移的 ISO 8601 原始发言时间；缺省使用服务器接收时间，不声称知道真实播放时刻。不要在重试时生成新时间。
- `user_id="__guest__"`：陌生/未确认身份，只参与共享 L1，不调用个人 L3，也不生成该类别的长期记忆。
- `core_write.status`：`persisted` 才可确认对应变更长期保存；`not_requested`/`failed` 不能说“已长期记住”。旧通用 L3 的 queued 回执不变。
- 409 `shared_memory_context_changed`：缓存已被删除/更正撤销，禁止继续使用旧 Prompt。

## 3. 响应示例

B 的请求：

```json
{
  "user_id": "user_B",
  "previous_ai_reply": "去公园不错，可以提前看看天气。",
  "query": "你刚刚说什么？"
}
```

返回：

```json
{
  "request_id": "ctx_<server-generated-id>",
  "user_id": "user_B",
  "agent_id": "agent_001",
  "messages": [
    {"role":"system","content":"记忆使用规则、当前身份 user_B、当前 user_B 的核心状态、带所有者标签的共享 L3 引用……"},
    {"role":"user","content":"[speaker {\"agent_id\": \"agent_001\", \"speaker_role\": \"user\", \"user_id\": \"user_A\"}]\n周末我想去公园。"},
    {"role":"assistant","content":"[speaker {\"agent_id\": \"agent_001\", \"speaker_role\": \"assistant\"}]\n去公园不错，可以提前看看天气。"},
    {"role":"user","content":"[speaker {\"agent_id\": \"agent_001\", \"speaker_role\": \"user\", \"user_id\": \"user_B\"}]\n你刚刚说什么？"}
  ],
  "has_prompt": true,
  "previous_reply_saved": true,
  "recall_status": "ok",
  "consolidation_status": "not_triggered"
}
```

messages 已经包含当前 query，不要再次追加或叠加一套重复聊天历史。记忆 system 不包含 CLI 的 Bumi 人设；对方保留自己的人设、工具规则和正在执行的工具上下文。每轮替换服务返回的记忆上下文，不要自行保留已过期的引用。当前用户核心块按用户隔离；已发布的 L3 引用是共享现场资料，可保留其他所有者的明确标签。

- recall_status：ok/unavailable/deleting/disabled；只补存时为 not_requested。
- consolidation_status：not_triggered/queued/succeeded；是接收时的整理提交状态，不是后台完成通知。queued 不等于 L3 已保存成功。
- request_id：服务端诊断标识，不要求下次带回。

## 4. 第一轮、暂停、最后一轮与无输出

- 第一轮 previous_ai_reply=null 或空字符串：只写当前用户问题，不伪造 AI 回复。
- 已有上一轮时传空回复：明确表示上一轮没有 AI 输出，结束其待补存状态，保留用户原话，继续当前 query。不要用空值跳过仍在生成中的真实回复。
- 语音打断时可以传实际说出的前缀，不传未播放的生成尾部。三个字段没有完整播放状态信息，本服务不能代替对方判断实际播放进度。
- 用户暂停说话不会使上一轮关联在五分钟后自动过期；单接口要等下一次问题才收到回复，重启后也保留该关联。
- 最后一轮若没有后续 query，回复不会自动到达服务端。对方应在结束/断开前用同一接口补存：

```json
{"user_id":"user_B","previous_ai_reply":"这是最后实际说出的回答。","query":null}
```

返回 has_prompt=false、messages=[]，不检索 L3、不产生新用户消息。补存不是强制整理：只有达到既有消息/token 边界时才创建整理任务。

如果不做结束补存，最后一条 AI 回复可能只在对方内存中，不能宣称已持久化。没有待补存轮次却提交非空 previous_ai_reply 会返回 409 no_previous_turn，不猜测其来源，也不顺便写入当前 query。

## 5. 顺序、幂等和故障恢复

**同一 agent 的调用必须由一个协调器按顺序发送。** 三个字段不能识别并行生成的回复或乱序的多个独立会话。服务内部锁会串行处理数据库操作，但不能辨认客户端误把旧答案作为下一轮答案提交的情况；需要并行时必须扩展关联协议，而不是依赖内容猜测。

- 补存上一轮回答、上一轮边界整理、写入当前 query、建立本次请求记录，在同一 SQLite 事务中完成。
- Embedding/L3 检索在事务外执行，结果快照随后持久化。
- 同 key 同内容的当前请求重试返回同一结果，不重复补存 AI 回复、不重复插入 query、不重复创建整理任务。
- 如果事务后、Prompt 返回前崩溃，重启后用原 key 和原始三个字段重试，继续生成/读取缓存结果。
- 当前请求已被后续请求推进完成时，重放旧的取 Prompt 请求返回 409 context_request_closed，避免重新运行陈旧 Prompt。
- query=null 的补存回执可用原 key 重取，不重复保存最后一句话。
- 无 key 时无法承诺跨网络失败的精确去重。相同文本可能是真实的重复提问，不能按文本自动去重。
- 老双阶段内部调用的待完成轮次不能混入单接口，返回 incompatible_pending_turn。升级/切换入口前应结束旧流程；不要让 CLI 与 HTTP 同时使用同一运行库。

## 6. 长期记忆与删除边界

我们的服务不调用回答模型；外部 Agent 自行生成回答。现有摘要/提取 API 仍由我们调用，保留 Qwen3 Embedding、Mem0、Qdrant 和后台 Worker。

共享 L1 默认合计 20 条有效消息（约 10 轮），轮转后最多保留 4 条 carry-over（约 2 轮）。在补齐上一轮 AI 回复后先检查边界，再保存当前 query。后台按个人原始发言及对应对话片段分开提取，不重复整理 carry-over。

AI 已说出的内容属于共享现场：如果它说出了 A 的私有信息，B 也可以在 L1 看到；通过所属用户门控的 L3 结果存为绑定当前轮的 shared_memory_refs，并随共享 L1 一起呈现。不能同时承诺共享现场回答和这些回答的跨用户保密。

forget-all 清理个人 L3 和新增的核心/状态记忆，撤销该用户发布的引用，并使同 agent 下可能含该引用的其他用户缓存 Prompt 失效。对于单接口，等待补存的 AI 仍可写入公共短期记录，但删除前轮次的迟到 AI 消息标为不可再次提取，避免复活已删除的长期记忆。该行为不新增对外删除接口，管理操作继续通过内部接口进行。

## 7. 配置及服务器启动

本地开发只改代码并运行模拟测试，不下载模型、不调用真实 API、不操作服务器数据库。

服务器准备好实际模型目录和适配的 PyTorch/CUDA 环境后：

```bash
cd /data1/sunzhixing/Bumi_Mem
# 激活你准备好的服务器 Python 环境
python -m pip install -e ./src/mem0-new
python -m pip install -e '.[server,runtime]'
mkdir -p /data1/sunzhixing/mem0-runtime/secrets
cp configs/memory_service.env.example /data1/sunzhixing/mem0-runtime/secrets/memory_service.env
chmod 600 /data1/sunzhixing/mem0-runtime/secrets/memory_service.env
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

上述是示例路径，不代表机器上已经存在。填写外部环境文件：

| 配置 | 用途 |
|---|---|
| BUMI_DATA_DIR | 独立联调数据目录，首次不要接真实用户库 |
| BUMI_EMBED_MODEL_PATH | 实际存在的本地 Qwen3-Embedding-0.6B 目录 |
| MEM0_EMBED_DEVICE | cuda 强制 GPU，auto 允许 CPU 回退 |
| BUMI_AGENT_ID | 当前部署的共享现场 ID |
| BUMI_SERVICE_API_KEY | 给 Agent 后端的服务密钥，至少 32 字符 |
| BUMI_MEMORY_API_KEY | 我们自己的记忆摘要/提取 API Key，不分发给对方 |
| BUMI_MEMORY_MODEL / BASE_URL | 我们的记忆模型与完整兼容 API 根路径 |

```bash
export BUMI_SERVICE_ENV_FILE=/data1/sunzhixing/mem0-runtime/secrets/memory_service.env
bash scripts/run_memory_service.sh
curl -f http://127.0.0.1:8088/health/ready
```

也可导出环境变量后执行 python -m bumi_mem.api。启动先验证模型路径并预热，失败则不接受业务请求；没有本地模型时不会悄悄下载。启动脚本不改写 HOME，不绑定旧服务器目录。

默认绑定 127.0.0.1。远程接入建议 HTTPS 反向代理、内网访问控制、限流与超时；测试改为 0.0.0.0 时配合防火墙，不能直接暴露公网。Uvicorn 固定 workers=1；文件锁阻止第二个 HTTP 服务打开同一目录，但不能约束旧 CLI/LoCoMo 入口。同一 Qdrant 本地目录不要多进程混用。

spaCy/BM25 是上游可选资源，内部降级未必抛错；warmup 成功不等于所有可选检索增强都启用，服务器部署时需检查日志并做真实检索验收。

## 8. 同步/异步客户端

src/bumi_mem/client.py 只依赖 httpx，提供 BumiMemoryClient 与 AsyncBumiMemoryClient，不要求 Agent 部门安装模型或 Mem0。

```python
from uuid import uuid4
from bumi_mem.client import AsyncBumiMemoryClient

previous_reply = None
async with AsyncBumiMemoryClient(service_url, service_key) as memory:
    async for user_id, query in your_user_input_stream:
        key = uuid4().hex  # 发送前保存 key 和请求体；失败重试使用原值
        data = await memory.get_context(user_id, query,
                                        previous_ai_reply=previous_reply,
                                        idempotency_key=key)
        previous_reply = None  # 上一回复已成功交付，勿在本轮模型失败后再次误交
        previous_reply = await your_agent_generate(your_agent_system_messages + data['messages'])
    # 若输入流正常结束且此前至少有一轮，补存最后输出。
    await memory.get_context(user_id, None, previous_ai_reply=previous_reply,
                             idempotency_key=uuid4().hex)
```

以上是流程示例；对方需处理空输入流、模型异常、播放进度和断线重连，并持久化尚未提交的 previous_reply。不能在模型还在并行生成时提前处理下一请求。

同步版同名 get_context，无 await。命令行单次调用：

```bash
export BUMI_SERVICE_URL=http://127.0.0.1:8088
# 安全设置 BUMI_SERVICE_API_KEY，不要放入源码。
python -m bumi_mem.client --user-id A --query '你好' --idempotency-key first
python -m bumi_mem.client --user-id B --previous-ai-reply '你好，我在这里。' --query '你刚刚说什么' --idempotency-key second
python -m bumi_mem.client --user-id B --previous-ai-reply '我刚才说我在这里。' --flush --idempotency-key end
```

## 9. 错误码和验证

| HTTP | code | 含义 |
|---|---|---|
| 401 | unauthorized | 服务凭证缺失或错误 |
| 409 | idempotency_conflict | 同 key 的三个业务字段发生改变 |
| 409 | context_request_closed | 请求已被后续轮次推进，不能重跑旧 Prompt |
| 409 | no_previous_turn | 没有已保存的上一轮，却提交非空 AI 回复 |
| 409 | previous_prompt_not_ready | 前次 Prompt 尚未准备好却提交了回答 |
| 409 | incompatible_pending_turn | 与旧双阶段/CLI 进行中的流程混用 |
| 409 | memory_scope_changed | 私有记忆已删除/作用域改变，旧 Prompt 不能重放 |
| 413 | request_too_large | 请求体超过 1 MiB，含 chunked 传输 |
| 422 | invalid_request / invalid_identity | 字段或身份不合法 |
| 503 | not_ready / memory_service_error | 未就绪或内部失败，使用原 key/原请求体重试 |

query 最多 16000 字符；previous_ai_reply 最多 64000 字符；身份/key 最多 128 字符。query 空字符串不等于补存，补存必须明确传 null。第一轮的 previous_ai_reply 可以省略，默认 null。

## 10. 主库升级与验证范围

主库升级为 v6：在原 v5 的 context_requests、protocol 和机器人身份迁移基础上，新增共享记忆引用、来源撤回、核心/状态版本、上下文 epoch，以及原消息关联和可选时间字段。原会话、发言、L3 所有者与任务快照不改写。v2/v3/v4/v5 通过增量迁移升级；旧缓存 Prompt 失效后按新契约重建。

升级前停服务并备份完整 SQLite 数据（注意 WAL），回滚应恢复备份，不能用旧代码直接打开 v6。对话和幂等审计记录暂未自动清理；正在使用的 Prompt 缓存含个人记忆，需要保护数据库并制定保留期限。

本地测试用真实 SQLite、FastAPI 测试客户端和模拟模型后端；不等于真实 GPU、模型准确率、Pipecat 语音或断电验收。运行：

```powershell
cd D:\代码\Bumi_mem
.\.venv\Scripts\python.exe -m pytest -q
```
