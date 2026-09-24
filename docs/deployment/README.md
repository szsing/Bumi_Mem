# 服务器启动与 Agent 连接（2026-09-22）

本次目标路径：

- 代码：`/data1/sunzhixing/Bumi_Mem`
- Embedding：`/data1/sunzhixing/model/Qwen3-Embedding-0.6B`（注意是 model，不是 models）
- 新联调数据：`/data1/sunzhixing/bumi-memory-service/data-test`
- 私有配置：`/data1/sunzhixing/bumi-memory-service/service.env`
- 可给 Agent 的连接配置：`/data1/sunzhixing/bumi-memory-service/agent-connection.env`

这些是供你在服务器执行的步骤，本地开发助手没有登录服务器、没有启动远程服务，也没有配置远程防火墙。

## 1. 先激活已经跑通 Embedding 的 Python 环境

不要在 Agent 环境或已有 GPU 环境中盲目升级 Torch/CUDA。下面脚本使用当前 python，首次配置会保存其绝对路径。若换了Python环境，要同步修改service.env中的BUMI_PYTHON，不要重建配置来随意轮换服务密钥。缺依赖先按预检提示补齐；首次安装的完整依赖见项目 pyproject.toml，若原环境正在生产使用，应在受控服务环境完成安装与验证。

更新代码前确保工作区干净、没有旧进程正在使用这个 checkout；已有真实数据先备份。不要用 reset --hard 覆盖本地修改。

```bash
cd /data1/sunzhixing/Bumi_Mem
git status --short
git fetch origin
git switch feature/agent-memory-api
git pull --ff-only
git submodule update --init src/mem0-new
```

若没有 server 侧依赖，至少需要兼容版本的 fastapi/uvicorn/filelock；实际模型还需要原来的 mem0、openai、torch、sentence-transformers 等环境。预检不会替你升级它们。

## 2. 首次生成配置，密钥只在服务器输入

```bash
bash scripts/setup_memory_service.sh
```

脚本在终端隐藏输入 DeepSeek Key，并生成另一个独立的服务 Key。两个配置文件权限为600，位于Git目录之外；已有配置时拒绝覆盖，避免重启时意外轮换密钥。

重要：

- `service.env` 含 DeepSeek Key，**不要整份发给 Agent 部门**。
- `agent-connection.env` 不含 DeepSeek Key，只含服务凭据、待填写的服务地址和预期 agent_id；通过私密渠道交付。
- 默认 agent_id 为 `bumi_robot_001`，默认端口8088，默认仅监听127.0.0.1。
  **端口是主机级资源、不按用户隔离**：同机多实例必须显式分配不同端口，不能假设 8088 空闲。
  本机约定（2026-09-24）：8088 归 sunzhixing 的实例，我们的验收实例用 8099。
- 数据使用新的联调目录，不自动继承你以前的记忆。要复用旧数据，先明确原 BUMI_DATA_DIR 和一致备份，不能只凭代码路径猜数据位置。
- 默认设备auto，CUDA可用时选GPU，否则CPU；要禁止CPU回退，可在私有配置中改为 `MEM0_EMBED_DEVICE=cuda`。CPU不代表已经满足之前的60ms目标。

## 3. 验证环境与 DeepSeek

```bash
export BUMI_SERVICE_ENV_FILE=/data1/sunzhixing/bumi-memory-service/service.env
bash scripts/check_memory_service.sh
```

会检查依赖、目录、设备，并发送一次很小的 JSON 请求（有少量 API token 费用）。不加载完整 Embedding、不创建记忆数据库、不启动监听端口。只检查环境、不调用API可加 `--skip-llm`，但这不能证明Key或模型可用。

0.4.1 将记忆服务默认模型改为 `deepseek-flash`，摘要和 Mem0 抽取都显式发送 `thinking.type=disabled`。可按需要改为当前官方支持的 `deepseek-v4-pro`，但应重新做真实整理/召回验证。已有配置文件不会被自动改写。

依据：2026-09-22 核对的 DeepSeek 官方 Models & Pricing / Thinking Mode 页面：

```text
https://api-docs.deepseek.com/quick_start/pricing
https://api-docs.deepseek.com/guides/thinking_mode
```

若用第三方代理，不要假定支持同样的模型名和参数；需要按代理契约配置 provider/base URL/model。

若提示 OPENROUTER_API_KEY，请在记忆服务专用 shell 中取消该环境变量后再试。固定 Mem0 SDK 会用它覆盖显式路由，本服务现在拒绝这种隐式改道。

预检成功不等于完整模型/数据库启动成功；Embedding和Qdrant以服务启动预热及健康检查为准。API调用只证明通信/JSON输出，不证明真实抽取质量。

## 4. 前台启动与本机验证

```bash
bash scripts/run_memory_service.sh
```

保持这个进程运行，在另一个终端检查：

```bash
# 端口以实际分配为准（8088 只是默认值）
curl --fail-with-body http://127.0.0.1:8088/health/ready
```

预期200，包含 `status=ready`、`warmed_up=true`。`/health/ready` 无法区分端口后面是谁的服务，
需再用自己的 `BUMI_SERVICE_API_KEY` 打一次 prepare 核对返回的 `agent_id`。
不应同时启动第二个进程共用同一数据目录。

第一次先前台看启动报错。常驻测试可在确认成功后用tmux或nohup；正式服务建议由运维使用systemd等进程管理，并保证使用正确Python环境。不要把SSH窗口关闭后仍会运行当作默认保证。

如果端口已被占用，核对已有进程或换 BUMI_PORT，不要随意杀掉不明服务。

## 5. 让另一台 Agent 机器访问

在受控网络中，停止测试服务后编辑私有service.env：

```dotenv
BUMI_HOST=0.0.0.0
# 改成实际分配的端口；同机多实例必须不同
BUMI_PORT=8088
```

重新启动。防火墙/云安全组只允许Agent来源访问这个TCP端口；公网不要裸露HTTP凭据与记忆数据，应使用HTTPS反向代理或受保护的网络。

Agent访问地址是服务器实际可达IP/域名，不是0.0.0.0；另一台机器也不能使用127.0.0.1指向你这台服务器。Base URL形状：

```text
http://<Agent可访问的服务器IP>:<实际分配的端口>
```

把实际地址填入agent-connection.env。当前还没有提供可达IP，因此脚本故意不猜地址，也不自动修改防火墙。

## 6. 给 Agent 部门的信息

只需提供：

1. Base URL；
2. BUMI_SERVICE_API_KEY（只分享这把服务密钥）；
3. 对应的 agent_id，例如bumi_robot_001；
4. API版本1.4.0和接口文档（主路径：每轮一次请求，响应为四个结构化字段）；
5. 该地址对应哪台机器人/共享现场。

不需要给他们 DeepSeek Key、Embedding模型路径或数据库访问权限。

每轮流程：

```text
prepare(client_turn_id + 可信user_id或speaker + 完整query + 原时间)
    → 保存服务返回的turn_id，把memory_context合入自己的LLM请求
complete(turn_id + 实际终态 + 确认回复 + speech证据)
```

欢迎语和stop/sleep走agent_event，重连用pending。先从Agent机器调用带Bearer的pending，核对返回agent_id，再做prepare/complete联调。headers仅健康检查不用Key，业务请求必须有Key。

Agent仍需按实际接线文档完成改造；单纯填写URL不代表自动接通。其示例声纹默认关闭/shadow时属于陌生身份，不会读写个人L3。

## 7. 最后验证一次真实整理

ready只表明启动预热，不代表DeepSeek已经处理真实会话。用合成、已确认身份测试一段会话，触发窗口整理后核验长期记录与召回；queued不能当作已保存。不要用空数据库的空结果证明身份隔离成功。

当前0.6.0保持六字段输入和每轮一次调用，但响应改为结构化数组，主库升级为schema 9以撤销旧字符串缓存；升级见 ../structured_turn_response/README.md。离线测试没有使用你的真实Key，也没有在你的服务器验证模型或网络。
