# Bumi_Mem

Bumi_Mem 是一个基于 Mem0 v2.0.20 的陪伴机器人分层记忆框架原型。当前版本实现：

- **L1 会话记忆**：SQLite 持久化，最多 20 条有效消息、8,000 token；
- **L2 会话摘要**：会话边界生成，仅用于存档、调试和审计；
- **L3 长期记忆**：通过 Mem0 提取、更新并使用本地向量检索；
- DeepSeek/OpenAI-compatible API 与本地 Qwen Provider 独立可选；
- 达到窗口上限或输入 `/new` 后才写入 L3，普通消息只检索 L3。

当前版本暂不包含长期用户画像、关系状态或情感状态机。

## Architecture

```text
                           ┌──────────────────────┐
用户输入 ─────────────────►│ L1 SQLite Session   │
                           │ 20 messages / 8k tok │
                           └──────────┬───────────┘
                                      │
                    当前输入           │ 最近 L1
                       │              │
                       ▼              ▼
                 Mem0.search()   Context Builder
                       │              │
                       └──── L3 ──────┤
                                      ▼
                              Chat Provider
                                      │
                                  先展示回答
                                      │
                 /new 或窗口上限       ▼
                           ┌──────────────────────┐
                           │ Session Consolidator│
                           ├──────────────────────┤
                           │ L2 summary           │
                           │ Mem0.add() -> L3     │
                           └──────────┬───────────┘
                                      ▼
                         新 session + 4 条只读衔接上下文
```

普通对话的关键路径是：

```text
SQLite append → Mem0.search → L1/L3 Prompt → Chat LLM
```

不会在每条消息后调用 `Mem0.add()`。长期记忆整合只发生在明确的会话边界。

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
/help                  显示帮助
/memories              查看当前用户全部 L3
/search <query>        手动检索 L3
/session               查看 L1 session 状态
/recall on|off         控制回答前的 L3 检索
/debug on|off          控制调试输出
/new                   整合当前 L1 并开始新 session
/discard               不写 L3，丢弃当前 session
/forget-all            删除当前用户全部 L3，需输入完整 user-id
/quit                  save / discard / cancel
```

正常模式不会向用户展示 Mem0、向量库或自动 session 边界。Debug 模式会显示 L2、L3 事件和 session ID。

## Session semantics

触发会话整合的条件：

- 达到 20 条有效 message；
- 达到 8,000 L1 token；
- 用户输入 `/new`；
- `/quit` 时选择 `save`。

整合顺序：

1. 冻结完整 L1 snapshot；
2. Memory Provider 生成 L2；
3. Mem0 `infer=True` 处理原始会话并写入 L3；
4. 成功后关闭旧 session；
5. 新 session 获得最后 4 条 `context_only` 衔接消息。

L3 失败时不会删除 L1。若 L2 已成功但 L3 失败，L2 会被清除，重试时重新生成 L2 和 L3。若 L2 失败但 L3 成功，系统使用确定性 fallback 摘要。

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

覆盖自动边界、失败恢复、幂等重试、carry-over、多用户隔离、Provider 独立配置、Prompt 顺序和本地切块。

## Existing evaluation tools

```bash
scripts/run_mem0_long_test.sh --quick --run-id local-quick
scripts/run_mem0_deepseek_v4_test.sh --quick --run-id deepseek-quick
scripts/run_mem0_locomo_deepseek.sh locomo-pilot \
  --conversations 0 --max-questions 5
```

所有运行数据写入 `/home/legendre/data/ZhixingSun/mem0-runtime`。
