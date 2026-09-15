# Bumi_Mem

## 当前收敛验收：用户ID隔离与已确认记忆恢复

- 未提供user_id时不自动使用demo_user；开发CLI用`/user <id>`绑定、`/logout`清除当前身份。
- `/remember <text>`明确写入并返回核验后的长期记忆ID；普通后台任务的queued/accepted不等于LTM已保存。
- 详细边界与专项验收见 [用户记忆说明](docs/user_memory/README.md)。
- 只验证自己的隔离进程正常重启/SIGKILL及模拟前端重连；整机休眠/断电另行安排，不冒充已验收。
- 原模型、检索参数及同事的性能配置不在本轮调整；之前的typed模块保留但不作为本轮入口。


Bumi_Mem 是一个基于 Mem0 v2.0.20 的陪伴机器人分层记忆框架原型。v0.2 在原 MVP 基础上增加持久化后台整理，当前实现：

- **L1 会话记忆**：SQLite 持久化，最多 20 条有效消息、8,000 token；
- **L2 会话摘要**：会话边界生成，仅用于存档、调试和审计；
- **L3 长期记忆**：通过 Mem0 提取并检索（固定 2.0.20 的普通抽取路径是 ADD-only，不自动更新旧事实）；
- **后台整理任务**：SQLite 持久化、立即切换会话、有限重试、分块检查点、租约恢复；
- DeepSeek/OpenAI-compatible API 与本地 Qwen Provider 独立可选；
- 达到窗口上限或输入 `/new` 后才写入 L3，普通消息只检索 L3。

当前版本暂不包含长期用户画像、关系状态或情感状态机。

## Architecture

```text
用户输入 → 保存 L1 → 有截止时间的 L3 检索 → 原有 L1/L3 Prompt → Chat Provider → 回答
                                                                    │
                                                             到达会话边界
                                                                    ▼
                           同一 SQLite 事务：冻结旧 L1 + 保存任务快照 + 新建会话
                                                                    │
                                       新会话立即可继续（仍保留最后 4 条只读衔接）

后台 Worker → 领取任务/续租 → L2 摘要 → 分块 L3 写入与核对 → 成功 / 延迟重试 / 待人工核查
```

普通对话不调用记忆提取模型，不等待上一段整理结束。任务在当前进程运行期间执行；关闭程序后不继续运行，下次启动恢复。

**本地 SDK 的安全边界**：一个固定线程创建并串行调用同一个 Mem0/Qdrant 实例。前台检索不能无限等待后台写入，默认等待上限为 0.2 秒；超时或检索失败时，本轮仅使用原有 L1/衔接上下文，`TurnResult.recall_status` 为 `unavailable`。这不改变查询算法、Top-5、20 message / 8,000 token 或 4 条 carry-over，但不承诺后台忙时每轮都能取到 L3。模型首次加载也可能触发该降级。可用 `--memory-read-timeout` 调整等待上限。

同一本地 Qdrant 目录只运行一个应用进程；本版不将其当作支持多进程共享的服务端数据库。

## Repository layout

```text
Bumi_Mem/
├── apps/
│   └── mem0_chat.py           # 薄 CLI 启动器
├── configs/
│   ├── deepseek.env.example
│   └── locomo-v2.0.20-manifest.json
├── scripts/
│   ├── runtime_env.sh
│   ├── run_mem0_chat.sh
│   ├── start_vllm_qwen.sh
│   └── ...
├── src/
│   ├── bumi_mem/              # 自研框架代码
│   ├── mem0-new/              # Mem0 v2.0.20 submodule
│   └── memory-benchmarks/     # Benchmark submodule
├── tests/
└── pyproject.toml
```

`src/mem0-new` 保持为未修改的上游 submodule；Bumi_Mem 通过 Adapter 使用 Mem0。

## External runtime layout

模型、数据库、密钥、日志和缓存不进入 Git 仓库：

```text
/home/legendre/data/ZhixingSun/
├── Bumi_Mem/
├── models/
│   ├── Qwen2.5-7B-Instruct/
│   └── Qwen3-Embedding-0.6B/
└── mem0-runtime/
    ├── conda/envs/mem0/
    ├── conda/envs/vllm/
    ├── cache/
    ├── data/
    │   ├── bumi_memory.sqlite3
    │   └── bumi_mem_l3/
    ├── logs/
    ├── results/
    ├── secrets/
    │   └── deepseek.env
    └── tmp/
```

可通过以下环境变量覆盖：

```bash
MEM0_RUNTIME_ROOT=/path/to/runtime
MEM0_MODEL_ROOT=/path/to/models
```

## Clone

上游项目使用 Git submodule：

```bash
git clone --recurse-submodules <repository-url>
```

已有 clone：

```bash
git submodule update --init --recursive
```

固定版本：

- Mem0 `v2.0.20`, commit `9a7924befd7026e41e445ba809370009e5e985a6`；
- memory-benchmarks commit `4b61c5d31b9c668a12b4f5e78064248a02c82d2b`。

## Install

在已有服务器环境中：

```bash
/home/legendre/data/ZhixingSun/mem0-runtime/conda/envs/mem0/bin/python \
  -m pip install --no-deps -e /home/legendre/data/ZhixingSun/Bumi_Mem
```

新环境可依次安装上游 Mem0 和 Bumi_Mem：

```bash
python -m pip install -e src/mem0-new
python -m pip install -e ".[dev]"
```

## DeepSeek quick start

真实密钥保存在仓库外。首次配置：

```bash
mkdir -p /home/legendre/data/ZhixingSun/mem0-runtime/secrets
cp configs/deepseek.env.example \
  /home/legendre/data/ZhixingSun/mem0-runtime/secrets/deepseek.env
chmod 600 /home/legendre/data/ZhixingSun/mem0-runtime/secrets/deepseek.env
```

编辑外部 `deepseek.env` 后运行：

```bash
cd /home/legendre/data/ZhixingSun/Bumi_Mem
scripts/run_mem0_chat.sh --user-id demo_user --agent-id bumi_companion
```

脚本默认：

```bash
CHAT_PROVIDER=deepseek
MEMORY_PROVIDER=deepseek
```

回答模型和记忆模型可以独立选择：

```bash
scripts/run_mem0_chat.sh \
  --chat-provider deepseek \
  --memory-provider local
```

CLI 参数覆盖环境变量；API Key 只能通过环境变量或外部 Secret 提供。

## Local Qwen mode

显存空闲时启动本地 OpenAI-compatible 服务：

```bash
scripts/start_vllm_qwen.sh
```

全本地运行：

```bash
scripts/run_mem0_chat.sh \
  --user-id demo_user \
  --chat-provider local \
  --memory-provider local
```

停止：

```bash
scripts/stop_vllm_qwen.sh
```

本地 Memory Provider 会按完整对话轮切块，每块默认不超过 3,500 token；不会从单条消息中间切断。

## CLI commands

```text
/help                            显示帮助
/memories                        查看当前 user/agent 的 L3
/search <query>                  手动检索 L3
/session                         查看 L1 以及后台任务
/jobs                            查看最近 50 个任务的状态、尝试次数与错误
/retry <job-id>                  手动重试失败任务
/retry <job-id> --allow-replay   核查 L3 后，显式允许重放结果不确定的分块；需再次输入 REPLAY
/recall on|off                   控制回答前的 L3 检索
/debug on|off                    调试输出，包括 recall_status 和任务回执
/new                            保存整理任务并立即开始新会话
/discard                        不写 L3，丢弃当前会话；不撤销之前已提交的后台任务
/forget-all                     屏蔽当前 user/agent 的 L3、取消旧写入并后台清理；需输入 user-id
/quit                           save / discard / cancel
```

`/new` 和 `save` 返回的是**任务已持久化的回执**，不是“记忆整理完成”。`/quit` 不等待新的整理任务；EOF/Ctrl+C 保留当前 L1，不自动打包。关闭时正在进行的外部调用无法安全强杀；进程终止后不会继续工作。

## Session and task semantics

触发条件保持不变：20 条有效消息、8,000 L1 token、`/new`、`/quit` 选择 `save`。

1. 一个 SQLite 事务冻结完整有效 L1、持久化源消息快照和不含密钥的任务配置、关闭旧窗口、新建 active 会话并复制最后 4 条 `context_only` 衔接消息。
2. 同一用户/agent 只能有一个 active 会话，但可以有多个待整理的历史会话。聊天状态与任务状态分离。
3. 一个进程内 Worker 从 SQLite 队列领取任务，使用租约避免第二个 Worker 重复领取。后台模型/向量调用不持有 SQLite 事务。
4. L2 成功后持久化并在后续重试复用；L2 失败时保存确定性 fallback 摘要并继续 L3，不改变原摘要/抽取 Prompt。
5. 分块 ID 由 job 和 chunk index 决定，不包含 attempt。已确认成功的块不重放；未完成块保留检查点。`context_only` 不作为新证据再次写入。
6. 已确认发生在写入前的失败（如固定 SDK 的 LLMError）自动退避重试，默认最多 3 次任务尝试。间隔默认 5、10 秒，可配置。失败任务不阻塞新聊天，`/retry` 可给它新的尝试预算。
7. 普通后台任务状态：`queued → running → succeeded`，或 `retry_wait / failed / cancelled`。

可配置参数：

```text
--memory-read-timeout 0.2
--memory-max-attempts 3
--memory-retry-base-seconds 5
```

### Crash recovery and write uncertainty

- 排队任务在下次启动恢复。中断任务先等待旧租约过期（默认最多约 30 秒），不能在启动时把所有 running 任务直接重置。
- 摘要/准备阶段中断可以自动恢复；已经确认成功的分块保持成功。
- **如果外部写入可能已成功，但 SQLite 还没收到成功确认，不能同时保证自动重试与绝不重复。** 本版将该块标为 `uncertain`、任务标为 `failed`，停止盲目重试。先核查 L3，再用 `/retry <job-id> --allow-replay` 显式允许重放；它可能产生重复记忆。
- 正常返回的 Mem0 ADD 记录会按 ID 读回核对。固定 SDK 仍可能将某些抽取解析失败表示成空结果；本层不能把这个接口变成端到端 exactly-once 保证。
- 新版任务表会增量创建，原 L1/L2 与历史审计保留。v0.1 的遗留 processing/failed 会话会转为任务；若旧审计显示已经尝试过 L3 写入（成功、失败或进行中），无法从旧日志证明是否产生副作用，先标记为待人工核查，不自动重放。首次升级前应关闭旧程序并备份运行数据库；不要直接用 v0.1 回滚已经迁移的运行数据库。
- 任务记录 provider/model 标识；配置发生变化时停止自动处理，恢复原配置后重试，避免悄悄用不同模型重放。

### Forget and background writes

`/forget-all` 保留“清理当前 user/agent 的 L3”的范围，不等同于删除全部原始聊天、L2、审计日志或备份。

- 请求先落盘，立即屏蔽该 scope 的 L3 读取并取消旧任务。
- 已经在执行的旧写入可能完成，但后台删除排在它后面；清理并核对成功之前，不解除读取屏蔽。
- 请求之前的 L1 源消息被标记为禁止再进入整理，防止以后再次提交旧 L1 造成复活。它们仍可作为原有聊天上下文，用户有明确的新表达时可形成新的记忆。
- 删除代际持久化；第一次删除之后用新的 Mem0 `run_id` 隔离其内部最近消息抽取缓存，避免旧缓存成为新的抽取来源。删除前的 generation 0 保持原 SDK 默认 scope。
- 删除失败时继续屏蔽并记录任务错误；原用户可以 `/retry`。新代际写入等待删除成功后再执行。

### Python API change

`BumiMemory.chat()` 的 `auto_consolidation` 与 `end_session()` 现在返回 `ConsolidationSubmission`（`accepted/job_id/status/new_session_id`），不再同步返回整合结果。不能把 `accepted=True` 当成 L3 成功。

使用 `memory_jobs()` / `store.get_job(job_id)` 查看结果，`retry_memory_job()` 重试。CLI 自动管理 Worker 生命周期；嵌入其他程序时应调用 `engine.close()` 或使用上下文管理器。真实 SDK 必须通过 `SerialMemoryAccess(factory=...)` 在拥有它的线程中创建，不要自行并发调用裸 Mem0。`auto_start=False` 和 `worker.run_once()` 供受控测试使用，不应与正在运行的 Worker 混用。

## Long-term memory policy

Mem0 提取 Prompt 要求使用以下前缀：

```text
[用户事实]
[用户偏好]
[用户边界]
[用户计划]
[重要事件]
[机器人承诺]
[共同事件]
```

问句、寒暄、临时信息、普通助手回答、通用知识和未经用户确认的推测不应进入 L3。

## Tests

单元测试使用 Fake Provider 和 Fake Mem0，不调用外部 API：

```bash
/home/legendre/data/ZhixingSun/mem0-runtime/conda/envs/mem0/bin/python \
  -m pytest -q
```

测试覆盖自动边界、任务回执、L1/carry-over/Prompt 不变性、有限重试、已成功分块不重放、租约和重启恢复、后台慢写不阻塞下一轮、删除竞态、SDK 线程归属、写后核对、旧 SQLite 迁移和 CLI 退出行为。

这些是使用 Fake/受控替身的离线测试，不代表真实 Qdrant、GPU、DeepSeek 或多设备部署已经验收。真实运行仍需初始化 submodule、安装依赖，并在目标环境用 `/new`、连续对话、`/jobs`、退出重启做冒烟测试。旧的 Mem0 benchmark 入口不等于 Bumi 端到端测试。

## Existing evaluation tools

```bash
scripts/run_mem0_long_test.sh --quick --run-id local-quick
scripts/run_mem0_deepseek_v4_test.sh --quick --run-id deepseek-quick
scripts/run_mem0_locomo_deepseek.sh locomo-pilot \
  --conversations 0 --max-questions 5
```

所有运行数据写入 `/home/legendre/data/ZhixingSun/mem0-runtime`。
