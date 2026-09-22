# bumi-agent 实际代码接入说明（Bumi_Mem v0.4 / schema 7）

核查对象：`Noetix-AI/bumi-agent` 提交 `212900a`，`requirements.txt` 固定 `pipecat-ai[silero]==1.8.0`。这份说明按该仓库的真实代码编写，不是官方 Mem0 插件的替代配置。

本次只修改记忆服务仓库。**Agent 仓库没有安装适配器或修改启动代码；仍需 Agent 同事按下述位置接线。**

## 1. 实际架构与接入位置

他们的默认管线是：

```text
BumiMicSource → SileroUserTurnProcessor → [WakeWordProcessor]
             → AgentTurnProcessor → 自定义 TTS service → BumiSpeakerSink
```

- `bumi_agent/pipecat_pipeline.py::AgentTurnProcessor._stream_llm_text()` 在 `asyncio.to_thread()` 的生产线程中调用 `ConversationAgent.iter_text_turn_text()` / `iter_audio_turn_text()`。
- `bumi_agent/agent.py::ConversationAgent._stream_current_turn()` 内部自己循环工具调用，每轮通过 `_messages_for_llm()` 构造消息，再调用自有 `LLMService.stream()`。
- 因此这里不是官方 `LLMContextAggregator → LLMService` 管线；在音频/TTS节点中间挂官方 `Mem0MemoryService` 不会拦截它内部实际发出的 LLM 请求。
- `ConversationState.turn()` 串行拥有对话历史，工具实际结果结算后释放生成所有权，`SpeechProgress` 完成后才释放待结算语音。
- `SpeechProgress.commit()` 只提交对齐到 DDS 成功发布游标的文本；`committed_text` 与生成文本不是同一个概念。`speech_history_incomplete` 保留字幕不足时的未知尾部。
- `ConversationAgent.history` 还包含 `tool_calls`、`tool`、图片、本地 stop/sleep 结果，不能用 Bumi 的纯对话消息整体覆盖。

## 2. 同一 URL，显式分阶段操作

仍然使用 `POST /v1/memory/context` 和 `Authorization: Bearer <BUMI_SERVICE_API_KEY>`，没有新增 Mem0 SDK 端点，也不调用回答模型。

| operation | 用途 | 稳定身份/幂等依据 |
|---|---|---|
| `prepare` | 保存一个完整新用户问题，返回本轮记忆上下文 | `client_turn_id`，全 agent 范围唯一 |
| `complete` | 提交该轮确认输出，显式结束它 | 服务返回的 `turn_id` + 不变的结束内容/证据 |
| `agent_event` | 无需用户问题的机器人独立发言或本地控制事实 | `event_id`，全 agent 范围唯一 |
| `pending` | 重连时只读取待结算轮次元数据 | 只读，不生成回答或检索个人记忆 |

不带 `operation` 的原 `user_id/previous_ai_reply/query` 请求继续保持旧 exchange 行为，仅供已有客户端兼容。**同一个未结束轮次不能混用新旧协议。**

新操作的幂等身份在 body 中，不依赖每次自动生成的 `Idempotency-Key`。旧 exchange 仍使用该 header。重试保存原 body，包括原时间，不重新生成 key/时间。HTTP 409 不能用换 ID 的方式绕过。

### 2.1 prepare：完整转写就绪后，每个用户轮次调用一次

```json
{
  "operation": "prepare",
  "client_turn_id": "run-c7d3:turn-000001",
  "speaker": {"status": "matched", "profile_id": "profile-001"},
  "query": "昨天我去了一趟游乐场",
  "occurred_at": "2026-09-21T14:30:00+08:00"
}
```

也可以用可信的 `user_id` 替代 `speaker`。同时提供时二者必须一致。`matched` 使用稳定 `profile_id`，**不使用可修改的显示 label**；unknown/timeout/unavailable/insufficient_audio/disabled 统一映射为 `__guest__`。不保存分数、声纹模型诊断等额外字段。

返回结构：

```json
{
  "operation": "prepare",
  "turn_id": "turn_<server-id>",
  "client_turn_id": "run-c7d3:turn-000001",
  "user_id": "profile-001",
  "agent_id": "bot",
  "source_message_id": "<original-message-id>",
  "occurred_at": "2026-09-21T14:30:00+08:00",
  "messages": ["完整的带时间共享对话，仅为示意，实际每项是role/content对象"],
  "memory_context": [{"role": "system", "content": "记忆规则、共享引用、当前用户核心状态……"}],
  "has_prompt": true,
  "recall_status": "ok",
  "core_write": {"status": "not_requested"}
}
```

- Agent 正常运行时使用 **`memory_context`**，不要将 `messages` 拼到它已有的全部 history 后面。
- Agent 原有固定 SYSTEM_PROMPT 保留；补充块作为后续 system 消息加入请求视图。该仓库的自有 OpenAI-compatible adapter 直接传 `messages`，本次不套用其他 Pipecat 版本的 system_instruction/developer 迁移。
- 本轮 query 已保存在 Bumi 中，但 Agent 仍应只按其原流程在本地 history 中保存一次。工具续推理复用这次 prepare，不重复提交问题。
- `source_message_id/occurred_at` 可供 Agent 给自己的请求视图/历史建立时间旁注；不要将自定义字段直接作为 OpenAI消息顶层字段发送。Bumi的完整 `messages` 已有时间，Agent自有历史并不会因调用服务自动获得时间标头。
- `occurred_at` 的偏移应对应对话发生地的日历时区。`timeline_occurred_at()` 默认保留设备本地偏移；若机器人系统时区不同，传入 `timezone_info`（例如配置后的 ZoneInfo）。不要为了统一日志先转 UTC，否则凌晨的“昨天”可能偏一天。
- 原始 `TurnTimeline.turn_id` 是进程内计数，重启后会重复。必须加持久化的 run/session 命名空间，原未完成请求的 ID 在重连后保持不变。
- 服务不会通过新 prepare 暗中关闭旧轮次；旧轮未结算时返回 `409 agent_busy`。

### 2.2 complete：只提交结算后的文本，不提交生成草稿

```json
{
  "operation": "complete",
  "turn_id": "turn_<prepare返回的server-id>",
  "status": "interrupted",
  "reply": "确认已经发布的前缀",
  "occurred_at": "2026-09-21T14:30:01+08:00",
  "speech": {
    "delivery_basis": "dds_successful_publish",
    "alignment_covers_output": false,
    "alignment_status": "interrupted_missing_subtitles",
    "output_samples": 16000,
    "sample_rate": 16000
  }
}
```

返回 `has_prompt=false`，不触发新的 LLM 请求。显式状态为 completed/interrupted/cancelled/failed；failed 可以携带已确认前缀，cancelled 不得携带非空回复。

- 正确来源是 **`SpeechProgress.on_complete` 之后的 `committed_text`**，不是 LLMFullResponseEndFrame、原始 LLMTextFrame、生成完毕事件、`SpeechRequest.done` 或 `speech._text`。
- 这个源码在 commit **之前**设置 `settled=True`；仅轮询 settled 不够，使用完成回调作为同步点。
- `alignment_covers_output=false` 时，即使 reply 为空，也保存 `speech_history_incomplete`，不把“字幕未知”当成“没有说话”。该程序标记不进入长期抽取。
- 被打断、失败或字幕不完整的助手片段保留在 L1，但不作为新的长期事实或承诺提取证据。用户的原始问题仍按正常规则整理。
- `transcript_complete` 表示声明的对齐数据是否覆盖输出游标，**不表示生成全文已经播完**。DDS publish 也不是物理扬声器播放确认。
- 纯文字调试可省略 speech，明确按 `text_only` 处理；真实语音接入必须发送 SpeechProgress 的证据。
- 原轮次已完成后，同样的 complete 重试返回同样回执；更换文字/状态/证据返回 `409 reply_conflict`。迟到的 A 回复不会被挂到 B。

### 2.3 agent_event：独立欢迎语与本地控制

无用户问题的已知欢迎语：

```json
{
  "operation": "agent_event",
  "event_id": "run-c7d3:welcome-1",
  "kind": "speech",
  "text": "你好，我是Bumi。",
  "speech": {
    "delivery_basis": "dds_successful_publish",
    "alignment_covers_output": true,
    "alignment_status": "completed",
    "output_samples": 32000,
    "sample_rate": 16000
  }
}
```

本地 stop 结果，来自 `ConversationState.finish_control()` 已写入的程序事件：

```json
{
  "operation": "agent_event",
  "event_id": "run-c7d3:stop-1",
  "kind": "local_control",
  "control": {
    "command": "stop",
    "user_text": "停下来",
    "reason": "user_requested_stop",
    "result": {"status": "interrupted", "started": true},
    "feedback_status": "completed",
    "llm_followup": false
  }
}
```

这些接口**只记录事实，不执行动作、不调用回答模型、不自动关闭当前用户轮次**。先让本地停止/休眠生效，再异步上报。控制事件会撤销旧的缓存 Prompt，但原轮仍需根据实际结果 complete。

独立机器人发言属于 agent，不冒充用户发言；程序事件和这些独立发言不生成个人 L3。控制结果须是紧凑结构化摘要（result JSON 最多8,000字符），不能塞入音频、图像或整个工具历史。

本地 WAV 若没有可信文字，不要编造一段欢迎语入库。stop/sleep 的提示音状态在 feedback_status 中记录；其他纯音效可继续由 Agent 本地管理。

### 2.4 pending：重连恢复，不猜测旧轮次输出

```json
{"operation": "pending"}
```

只返回 pending 的 `turn_id/client_turn_id/user_id/protocol/status`，或者 null，不返回原问题或个人记忆。Agent协议的待结算轮不会在5分钟后自动过期。已结束的旧输入不能被重新 prepare 触发推理。

恢复顺序：读取本地持久化 outbox → 重试原 prepare/complete → 再处理新问题。若 outbox 已丢失，先确认旧播放/任务已停止，再用 pending 的明确 turn_id 提交 failed，并声明 `delivery_basis=unknown, alignment_covers_output=false, alignment_status=restart_unknown`，保留未知事实。不要用空回复伪造“原来没有说话”。服务不会在 pending 调用中自动完成这一步。

## 3. Agent 需要接线的四处代码

### A. `ConversationAgent.iter_text_turn_text()`

在 `conversation.turn(...): admitted` 之后、第一次 `_stream_current_turn()` 请求 LLM 之前：

1. 为当前真实用户轮次冻结 client_turn_id、身份和发言时间，将 prepare 请求放入持久化 outbox。
2. 在已有的 LLM 生产线程里调用同步 `BumiMemoryClient.agent_request()`。不要把阻塞 HTTP 放在实时 FrameProcessor/音频回调中。
3. 将返回的 server turn_id 绑定到**该轮专属 handle**。内存响应到达后再次检查 cancel_event；被取消的轮次不能进入 LLM，仍需有序结算服务器轮次。
4. 用户原话在 Agent 自有 history 中只保存一次。记忆故障时可按产品策略退回本地上下文，但不能确认长期保存成功，也不能丢弃未知结果的原请求。

`iter_audio_turn_text()` 的独立ASR路径最终进入上述方法。`asr.bypass` 的原始PCM路径没有可用 query，不能把占位提示“请理解这段语音”存成用户原话；要增加可靠最终转写，或明确暂不启用该路径的记忆集成。仓库示例配置使用 dashscope ASR，但实际部署配置需现场确认。

### B. `ConversationAgent._messages_for_llm()`

在构造完原请求消息、return 之前调用 `agent_bridge.merge_agent_context(...)`，它只插入记忆补充块，不改变原列表，不删改 tool_calls/tool/image/local_control/speech_history_incomplete，也不再次加入 query。每次工具循环复用该用户轮的同一份响应，检查 expected_user_id 和 expected_client_turn_id。

该 response 只属于当前轮的请求视图，不应写入永久 Agent.history，否则会残留过期核心块和共享引用。

冷启动时若 Agent.history 确实为空，可使用 `cold_start_spoken_history()` 从完整 messages 恢复过去的普通对话（自动排除本次query），然后按原流程添加本次用户输入。**不能从纯对话恢复工具执行状态，不能覆盖任何已有本地历史。** 该辅助方法不会自动修改 Agent 对象。

### C. `AgentTurnProcessor._process_turn()` 创建的 `SpeechProgress`

为每个 SpeechProgress 安装 on_complete 回调。回调只调用 `settled_speech_snapshot()` 冻结证据并入队，**不做HTTP、不等网络**；由独立发送线程/outbox将其与原prepare的server turn_id、响应任务的明确终态关联后发送complete。语音结算并不证明LLM/工具任务成功，因此快照不猜status，协调器须等待实际completed/interrupted/cancelled/failed结果。

若prepare尚未返回，先把快照绑定到该轮client_turn_id，等原prepare重试拿到server ID后发送，不读取“当前全局turn_id”。不要等下一位用户说话才第一次尝试保存最后回复。

### D. `ConversationState.finish_control()` / 独立机器人发言结束

已有的local_control消息可用 `local_control_payload()` 转成agent_event。同样异步入队，不阻塞停止指令或持有ConversationState的condition做网络I/O。欢迎语使用独立event_id；不伪造用户问题。

### 可复用的薄客户端与纯函数

本仓库：

- `src/bumi_mem/client.py`：同步/异步 HTTP 客户端；prepare_turn/complete_turn/agent_request；只依赖 httpx。
- `src/bumi_mem/agent_bridge.py`：上述身份、ID、时间、语音快照、上下文合并纯函数，只依赖标准库。

Agent团队可以只复用/复制这两个薄文件；**不需要为了调用HTTP在机器人进程安装或初始化 Mem0/Qdrant/Embedding**。httpx 已在其 requirements 中。此说明不意味着Agent仓库已经自动加载这两个文件。

示意（由Agent的轮次handle/outbox提供这些变量，不是可以直接粘贴进实时回调的完整补丁）：

```python
payload = prepare_payload(
    query=final_transcript,
    request_turn_id=client_turn_id(run_id, timeline.turn_id),
    identity=speaker_identity,
    identity_enabled=settings.speaker_id_enabled,
    identity_mode=settings.speaker_id_mode,
    occurred_at=timeline_occurred_at(timeline),
)
# 在对话生产线程中；先持久化payload，重试不改它。
response = memory_client.agent_request(payload)
# 每轮绑定，不使用可能已切换的全局用户/轮次ID。
turn_handle.bind(response)
messages = merge_agent_context(
    original_agent_messages, response,
    expected_user_id=turn_handle.user_id,
    expected_client_turn_id=turn_handle.client_turn_id,
)
# SpeechProgress.on_complete中只抓取快照，发送交给outbox工作线程：
snapshot = settled_speech_snapshot(speech)
outbox.enqueue_speech_snapshot(turn_handle.client_turn_id, snapshot)
# 协调器另外取得响应任务终态后，将status和server turn_id合入发送体。
```

`outbox/turn_handle` 是需要Agent侧实现的生命周期组件，不是本次已经安装的对象。仅有HTTP锁不够：要按逻辑轮次排序；中断/本地控制仍须即时响应。

## 4. 身份与配置特别注意

Agent `.env.example` 默认 `BUMI_SPEAKER_ID_ENABLED=0`、`BUMI_SPEAKER_ID_MODE=shadow`。适配helper在关闭或shadow模式下明确使用陌生身份，不会偷偷把仅用于观察的匹配结果变成个人记忆权限。启用可靠身份后再进入label模式，或由其他可信机制显式提供稳定user_id；不能使用显示名当主键。

每台独立机器人使用独立的服务agent_id/数据目录。服务端固定 `BUMI_AGENT_ID`，请求体不能任意覆盖它。新旧协议都要求服务密钥只放在可信Agent后端。

可用连接参数：服务URL（不是DeepSeek模型URL）、BUMI_SERVICE_API_KEY、HTTP超时。示例见 `configs/bumi_agent.env.example`；Agent原Settings尚未读取这些新配置，须接线时添加。

检索60ms目标没有在本次重新测量；HTTP往返、状态保存、DDS结算不是同一个指标。本次没有启动Agent、声纹服务、音频设备、真实模型或GPU。2026-09-22 的 CPU/本地 Qdrant 压测与串行拼装分解见 [时延压测报告](../latency/REPORT.md)。

## 5. 升级与验证

Bumi_Mem 0.4.0 / OpenAPI 1.2.0 / SQLite schema 7。v6增量迁移只增加explicit-turn/event幂等表，保留旧pending轮次；升级前结束旧入口或按旧协议结算，不能让新complete操作处理旧protocol轮次。部署前备份，旧二进制拒绝更高版本库，回滚恢复备份。

```powershell
# 实际Agent源码的纯逻辑契约测试（不加载Pipecat或硬件）
$env:BUMI_AGENT_REPO = 'D:\代码\Bumi\_agent'
.\.venv\Scripts\python.exe -m pytest -q tests/test_agent_protocol.py tests/test_agent_bridge.py
# 全部离线测试
.\.venv\Scripts\python.exe -m pytest -q
```

跨仓库测试直接加载其SpeakerIdentity、SpeechProgress、ConversationState，并执行其真实消息构造方法的AST，不复制这些实现，不改Agent文件。CI没有Agent checkout时这5项实际源码测试显式skip，其余HTTP/桥接测试照常运行。

上线还需验证：实际Agent接线、字幕与DDS对齐、打断/失败、ASR最终轮次、工具执行与本地stop/sleep、outbox恢复、身份模式、真实检索延迟及模型回答质量。离线契约通过不是整机或已部署Pipecat链路通过。
