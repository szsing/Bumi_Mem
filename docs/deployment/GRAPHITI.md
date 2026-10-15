# Graphiti 长期记忆部署与验收

本次服务运行链路为 Graphiti，不再使用 Mem0。旧 Mem0 实现和历史测试暂留仓库作为对照，不是运行时依赖。主 API 仍是 `/v1/memory/context`，四字段响应不变。

## 已实现的行为

- L1 仍使用 SQLite 原文与共享近期上下文。
- 抽取与 L1 独立：`BUMI_EXTRACTION_TURNS=10` 个完成轮次，或 `BUMI_EXTRACTION_TOKENS=8000` 提前触发。token 计数为保守近似值，后台还会按来源头大小分段；单轮过长不会静默截断。
- 原始消息、未满批次和已排队任务持久化。没有空闲 flush；关闭服务不强制提交未满批次。
- 通过上游 `add_episode_bulk([RawEpisode(...)])` 写入一批 message episode；超长批次分成多个，按顺序执行。每批不跨不同所有者合并。
- 使用固定上游代码的默认 C2；不传 custom extraction instructions，不限定旧的七类记忆，不先生成 L2 摘要。默认语言行为同上游，中文输入主要产生中文事实，关系标签可能是英文。
- 原生 `search_()` 综合默认配方，涵盖边、节点、episode 和 community；本地 BGE 重排。原 Mem0 top-k/score 阈值不映射。上游每类默认上限及 BGE 最低分配置照用，BGE 分数不是概率。
- 原生 Graphiti 保留文本、向量与图结构，Neo4j 同时存储相关索引。无需 Qdrant。
- 当前和历史结果都可能召回，内容标明图对象类型、有效/失效/记录时间；记录时间不能当事件发生时间。外部响应仍受上下文长度预算限制，不保证所有图结果都进入最终上下文。
- 旧 semantic_state 的即时更新、机器人承诺写入、核心/历史注入和冲突过滤在 Graphiti 服务关闭；显式 `remember()` 停用。
- Graphiti 在长期运行的事件循环上使用 async SDK；前台查询不在后台写入后排队。CPU/GPU/数据库资源仍会竞争，需要服务器实测。
- 每个分组完成 5 批后执行一次后台 community 重建；多个待更新批次合并，失败至少 60 秒后重试。分组重建会清除旧 community，期间可暂时缺少 community 结果。抽取不因 community 失败被重新执行。
- 写入中断标记 uncertain，不盲目自动重放。不承诺跨 SQLite 和 Neo4j 的 exactly-once。

## 1. 上传与 Python 环境

上传整个本次代码目录到服务器的新 checkout，保留已有版本以便对照；不上传 Windows `.venv`。不需要初始化 `src/mem0-new` 子模块。

使用 Python 3.10+；建议保留当前已验证的 Torch/CUDA 组合，在专用服务环境安装。本地验证使用 Python 3.12。执行：

```bash
cd /实际上传的/Bumi_mem
python -m pip install -e '.[server,runtime,test]'
```

`pyproject.toml` 固定 GitHub 的 `v0.31.0` 源码发布标签，而不是索引中更旧的包。该标签源码自身的包 metadata 标成 `0.32.0`，因此 `pip show graphiti-core` 显示 0.32.0 并不表示装了另外的 main；可检查安装来源：

```bash
python -c "from importlib.metadata import distribution; print(distribution('graphiti-core').read_text('direct_url.json'))"
```

若服务器访问 GitHub 下载失败，需要先在可访问网络下载同一发布标签源码，再从本地安装；不要静默改装更旧版本，否则 C2 与接口可能不同。

## 2. 本地模型

已有 embedding：`/data1/sunzhixing/model/Qwen3-Embedding-0.6B`。

新增 reranker：`/data1/sunzhixing/model/bge-reranker-v2-m3`。

```bash
mkdir -p /data1/sunzhixing/model
python3 -m venv /data1/sunzhixing/model/.hf-download-env
/data1/sunzhixing/model/.hf-download-env/bin/python -m pip install -U huggingface_hub
HF_HUB_OFFLINE=0 /data1/sunzhixing/model/.hf-download-env/bin/hf download \
  BAAI/bge-reranker-v2-m3 \
  --local-dir /data1/sunzhixing/model/bge-reranker-v2-m3
```

服务只从本地模型目录加载，不自动下载。模型层适配仅替换上游 BGE 的硬编码仓库名为目录；继承原生 rank 方法。Qwen 使用本地 SentenceTransformer 的 encode，向量维度必须为配置的 1024。`auto` 的 GPU 使用取决于实际 Torch 环境；强制 GPU 用 `BUMI_EMBED_DEVICE=cuda`。

## 3. 新建配置和 Neo4j

将 `configs/graphiti.env.example` 复制到 Git 目录外的私有 `graphiti.env`，填写服务 key、DeepSeek key 和 Neo4j 密码。保持新的 `BUMI_DATA_DIR` 和新的 `BUMI_NEO4J_DATA_DIR`，不继承旧 Mem0 联调数据。

```bash
export BUMI_SERVICE_ENV_FILE=/绝对路径/graphiti.env
set -a
source "$BUMI_SERVICE_ENV_FILE"
set +a

docker compose -p bumi-graphiti -f scripts/neo4j.compose.yaml up -d
```

Compose 使用 Neo4j 5.26.0 Community，数据目录持久化，HTTP/Bolt 仅绑定本机。默认端口为 7474/7687，如被占用，在配置中设置 `BUMI_NEO4J_HTTP_PORT`、`BUMI_NEO4J_BOLT_PORT` 并同步改 `BUMI_NEO4J_URI`。该脚本需要服务器已具备 Docker Compose；没有 Docker 时由部署环境提供兼容 Neo4j 并配置 URI，不自动安装系统服务。

不要共用已有真实图数据库。通过 SSH 隧道访问浏览器：

```bash
ssh -L 7474:127.0.0.1:7474 -L 7687:127.0.0.1:7687 用户@服务器
```

随后本地浏览器打开 `http://127.0.0.1:7474`，使用配置中的 Neo4j 账号密码。

## 4. 预检和启动

```bash
python -m bumi_mem.service_probe --skip-llm
# 下一个命令会调用一次真实 DeepSeek API：
python -m bumi_mem.service_probe
bash scripts/run_memory_service.sh
```

主服务启动预热会检查图索引、加载两个本地模型并跑一次向量/重排，不做抽取。`/health/ready` 成功不等于记忆质量通过。只运行一个 worker，不与验收探针同时打开同一数据目录。

读超时首版设为 30 秒，便于真实验证原生综合检索；不代表已达到原先 60ms 目标。模型加载预热预算 300 秒，可配置。重排/向量推理及 Neo4j 性能应在目标服务器测量。

## 5. 真实写入与跨重启召回

在主服务停止时，使用相同配置运行独立探针（会产生真实 API 费用和实验数据）：

```bash
python -m bumi_mem.graphiti_probe --user-id graphiti_acceptance_001
```

脚本经过真实轮次协议提交默认 10 轮，等待持久任务结束，随后直接从 Graphiti 召回，避开 L1 与旧语义状态。检查输出是否有来源、类型、时间且实际回答问题；非空不能单独证明正确。

重新执行下列命令验证进程重启后仍可召回：

```bash
python -m bumi_mem.graphiti_probe --recall-only --user-id graphiti_acceptance_001
```

该探针第一次运行最后使用 null query 结算最后一条 AI 回复，只为无人值守验收凑满完成轮次；生产不依赖显式结束，通常由下一次请求结算上次 AI 回复。

Community 默认要同一用户/agent 成功 5 批才构建。验收可临时设 `BUMI_COMMUNITY_EVERY=1` 后跑一次，或正常积累 5 批；后台最多约 5 秒后开始检查。不要以第一次未见 community 判断抽取失败。

## 6. 调试和恢复

主服务运行时可查询带 bearer 认证的 `GET /v1/memory/debug?user_id=...`，返回 namespace、每批/分段写入回执和 community 水位。主四字段上下文响应不暴露内部图结构。

Neo4j 查询示例（先从 debug 取应用生成的 group_id）：

```cypher
MATCH (n {group_id: '替换为debug返回值'}) RETURN n LIMIT 100;
```

SQLite 原始消息/任务在 `BUMI_DATA_DIR/bumi_memory.sqlite3`；Graphiti 写入日志及 community 状态在 `BUMI_DATA_DIR/graphiti/graphiti-journal.sqlite3`。日志仅用于执行状态和来源映射，不参与长期内容召回。

发生 uncertain 时保留现场，查看任务分段、日志和图中 episode。应用不会凭 episode 存在就宣称全部写入完成；未完成日志不会自动重放。实验可换新 namespace/数据目录重新验证，原目录留作排查。不要只删本地写入日志后对原图重试。

本次未实现通用图对象管理 API；旧的扁平 get/list 入口明确不可用，使用图 debug/search 与 Neo4j 浏览器。整组遗忘保留删除代际机制；完整隐私删除、备份清理和家庭细粒度授权不属于这次验收。

## 7. 验收范围

离线：`python -m pytest -q -p no:cacheprovider`。新增 Graphiti 专项测试覆盖批次独立、跨重启尾部、语义路径停用、异步并发与关闭、超长输入、原生调用参数、历史结果以及 community 重试。上传包不包含旧 Mem0 子模块时，一项读取其源码的历史测试会跳过。

服务器上还需确认：真实 C2 中文抽取正确、偏好更新历史可辨、同一 namespace 数据持久化、BGE 与 Qwen 设备/显存、综合检索输出、community 构建和故障恢复。未运行这些真实检查前，不声称已在机器人或服务器验收通过。
