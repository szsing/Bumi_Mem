# Bumi 记忆读写时延压测报告

| 项目 | 内容 |
|---|---|
| 任务 | Memory-B1-S4「定义记忆读写 API 与延迟预算」（链 A 并行分支） |
| 验收指标 | 检索 P90 ≤ 60ms（#7）；串行关键路径只计最后的上下文拼装，≤ 20ms |
| 测试日期 | 2026-09-22 ~ 09-23 |
| 执行人 | 张冉 |
| 被测版本 | Bumi_Mem `6315aaf`（main，schema 7 / v0.4 Agent API）、mem0 2.0.20 |
| 压测脚本 | `tools/latency/latency_bench.py`（等价于本机 `~/.local/bin/bumi-latency-bench`） |
| 结论 | **检索 P90 ≤ 60ms：仍未通过，但 GPU 复测后已接近**：300 条 P90 **64.9ms**（超预算部分全部来自实体加成）、5000 条 P90 **221.7ms**（瓶颈转为 Qdrant local client）；**串行拼装 ≤ 20ms：通过**（P90 14.5 / 15.5ms） |

> 一句话结论：目标部署形态是 **H20 + GPU 编码**，本报告先给 CPU 基线，再给 2026-09-23 的 GPU 复测
> （`~/.venvs/bumi-cuda`，torch 2.8.0+cu128，`CUDA_VISIBLE_DEVICES=4`，`--embed-device cuda`）。
> 完整检索 P90 从 CPU 的 156.8 / 393.2ms 降到 **64.9ms（300 条）** 与 **221.7ms（5000 条）**：
>
> - **300 条已经很接近达标**：不含实体加成的查询 P90 只有 **41.5ms**，超预算的全部来自**实体加成**
>   （含实体分组 P90 66.2ms；实体加成本身 GPU 下 24.2ms，是一次额外的批量编码）。
> - **5000 条仍差得远**：瓶颈已经不在编码，而是 **Qdrant local client 的精确扫描**（稠密路 P90 89.6ms +
>   BM25 路 82.5ms），与 GPU 无关，必须换 Qdrant Server/HNSW。
> - 串行关键路径 P90 **14.5ms（300 条）/ 15.5ms（5000 条）**，满足 ≤20ms；首轮冷启动 25–26ms 仍需预热或豁免。
>
> 结论：GPU 解决了“编码”这一项（P90 64.4→23.1ms），剩下两个互相独立的否决项——**实体加成**
> 和**本地 Qdrant**——要在验收前各自解决；提前发起只能把检索移出串行路径，不能替代这两个修复。
>
> CPU 基线（`~/.venvs/bumi`，torch 2.8.0+cpu）保留在 §3.1–3.6：完整检索 P90 156.8ms（300 条）/
> 393.2ms（5000 条），串行 P90 16.6 / 16.7ms。

---

## 1. 需求与计时口径

### 1.1 指标定义

本报告区分两个不能相加、也不能互相替代的指标：

| 指标 | 起止 | 预算 | 说明 |
|---|---|---|---|
| 检索时延 | query（或流式前缀）送入检索 → 排序后的候选/引用就绪 | **P90 ≤ 60ms** | 可在 ASR 说完之前提前发起，属于预取分支，不直接阻塞回答 |
| 串行关键路径 | ASR 终稿到达 → `memory_context` 返回 | **P90 ≤ 20ms** | 只包含最后的上下文拼装：等待预取结果、引用落库、L1 窗口读取与渲染、Prompt 持久化 |

提前发起后的串行时长模型（`t_lead` 为预取相对终稿的提前量）：

```text
t_serial = max(0, t_retrieval - t_lead) + t_assembly
要求 t_serial ≤ 20ms  ⇒  t_lead ≥ t_retrieval + t_assembly - 20ms
```

### 1.2 被测接口

对外只有 `POST /v1/memory/context` 一个业务接口，读写语义见
[Agent 接入文档](../agent_api/README.md) 与 [bumi-agent 接入说明](../bumi_agent_integration/README.md)：

| 路径 | 操作 | 本次是否测量 | 预算 |
|---|---|---|---|
| 读 | `operation=prepare`（保存本轮问题并返回记忆上下文，内部含 L3 检索） | 是 | 检索 ≤60ms P90（预取分支）；串行拼装 ≤20ms P90 |
| 写 | `operation=complete` / `agent_event`（结算已说出的回复/本地事件） | 部分（L1 落库计时） | L1 持久化 P90 ≤10ms；L3 整理异步，不进串行预算 |
| 读 | `operation=pending`（重连读取待结算轮次元数据） | 否 | 只读元数据，不检索 L3 |
| 读 | CLI `/search`、显式回忆（`BUMI_EXPLICIT_RECALL_TIMEOUT=1.0`） | 否 | 用户主动回忆，允许更长等待，不套 60ms |
| 管理 | `forget_all()` / `forget_memory()` | 否 | 非对话关键路径 |

`0.2s` 的 `BUMI_MEMORY_READ_TIMEOUT` 是**前台等待上限/降级阈值**，不是检索时延指标；本报告同时给出
“0.2s 截止下的前台行为”和“允许检索跑完的真实耗时”两组数据。

### 1.3 建议写入任务卡的延迟预算

| 编号 | 指标 | 预算 | 本次结果 |
|---|---|---|---|
| B1 | L3 检索（预取分支，目标部署/目标规模） | P90 ≤ 60ms | 未通过：CPU 156.8 / 393.2ms；GPU 64.9ms（300 条）/ 221.7ms（5000 条），见 §3.7 |
| B2 | 串行上下文拼装（ASR 终稿 → memory_context） | P90 ≤ 20ms | 通过：CPU 16.6ms、GPU 14.5ms（首轮冷启动 25–33ms 待处理） |
| B3 | L1 当前轮落库（`reserve_turn`） | P90 ≤ 10ms | 通过：2.3ms |
| B4 | Prompt 持久化（`save_turn_prompt`） | P90 ≤ 10ms | 通过：1.2ms |
| B5 | L3 整理/抽取 | 异步，不进串行预算 | 未测（后台任务） |
| B6 | 预取命中率 / 陈旧重算率 | 待定，需可观测 | 未测（接口未实现） |

预算口径固定为：本机内存服务进程内的时间，不含 HTTP/反向代理、ASR、DDS 播放、声纹和回答模型。

---

## 2. 环境与方法

### 2.1 环境

| 项目 | 值 |
|---|---|
| 机器 | Linux，180 逻辑 CPU，1929 GiB 内存 |
| GPU | 本机有 **8×NVIDIA H20**（97.8 GiB/卡，驱动 580.178.04，CUDA 13.0）。CPU 基线用 `~/.venvs/bumi`（`torch 2.8.0+cpu`，`auto` 回退 CPU）；**GPU 复测用 `~/.venvs/bumi-cuda`（`torch 2.8.0+cu128`，`CUDA_VISIBLE_DEVICES=4`，`--embed-device cuda`）**，见 §3.7 |
| Python | 3.10.12（`~/.venvs/bumi`） |
| 记忆栈 | mem0 2.0.20，Qwen3-Embedding-0.6B（1024 维，本地文件），spaCy 词形还原/实体抽取，fastembed `Qdrant/bm25` |
| 向量库 | **Qdrant local client**（`QdrantClient(path=...)`，单进程锁，非 Qdrant Server/HNSW） |
| 数据目录 | `mem0-runtime/latency-bench-final300`、`...-final5000`（隔离目录，不碰真实用户数据） |

参考：`docs/stage0/REPORT.md` 在另一台机器（RTX 4090，CUDA 12.4/FP16，依赖版本不同）测得
Qwen3-Embedding 单条编码 P95 约 **10.3ms**（32/128/256 tokens），同批 CPU P95 为
53.0 / 115.6 / 215.6ms。该数据只能作为量级参考，不是本机复测结果。**本机 H20 的 GPU 编码指标尚未
测量**：要判定 GPU 路径能否达标，需要在装有 CUDA 版 PyTorch 的环境用同一脚本复测，这是本次首要补测项。

#### 2.1.1 环境证据在哪里

脚本每次运行都会在最前面打印 `⓪ 环境证据`，并把同一份结构写进 JSON 的 `environment` 字段，
验收时不需要翻日志猜环境：

| 字段 | 含义 | 本次取值（示例） |
|---|---|---|
| `hostname` / `platform` / `logical_cpus` | 宿主 | `AIC-C01-H20-n01` / `Linux-5.15...` / 180 |
| `python_executable` | 实际解释器 | `/data1/zhangran/.venvs/bumi/bin/python` |
| `MEM0_EMBED_DEVICE_env` / `requested_embed_device` / `resolved_embed_device` | 环境变量、请求与实际生效的编码设备 | `auto` / `auto` / **`cpu`** |
| `torch_version` / `torch_cuda_build` / `torch_cuda_available` / `torch_cuda_device_count` / `torch_cuda_devices` | torch 与 CUDA 证据 | `2.8.0+cpu` / `None` / **`False`** / 0 / 缺省 |
| `embed_model` | 编码模型路径 | `.../Qwen3-Embedding-0.6B` |
| `vector_backend` / `vector_store_collection` | 向量库形态与集合 | `qdrant-local-path` / `bumi_latency_bench` |
| `config.corpus` / `config.unique_corpus` | 请求语料条数 / 实际唯一向量数 | 5000 / 5000 |
| `environment.warmup` | 冷启动是否走完 | `memory_construct_ms`、`bm25_available`、`first_search_ms`、`first_search_hits`，引擎路径另有 `engine_warmup_ms` |

判定 GPU 是否真的生效，必须同时看 `torch_cuda_available=true`、`resolved_embed_device=cuda`
和 `torch_cuda_devices` 非空；只写 `MEM0_EMBED_DEVICE=cuda` 不代表真的用了 GPU（`auto` 在 CPU 版
torch 下会静默回退）。

### 2.2 语料、查询与采样

- 合成语料：300 条与 5000 条唯一文本（`point_id` 按文本去重，实际向量条数=语料条数）。
- 查询：8 条固定查询（中英混合、短查询与含实体长查询），每条重复 20 次，共 160 个样本；`top_k=5`、
  `threshold=0.1`、`rerank=False`，与 CLI 默认一致。
- 计时点：直接包装真实调用链上的函数（`mem0.Memory.search`、`lemmatize_for_bm25`、
  `extract_entities`、`embedding_model.embed/embed_batch`、`vector_store.search/keyword_search`、
  `_compute_entity_boosts`），不重新实现检索逻辑。
- 前台路径：走真实 `bumi_mem.cli.build_engine` → `engine.prepare_prompt`，
  并额外用 5s 等待上限做一次“检索 vs 串行拼装”分解（见 §3.4）。
- 百分位：`P90` 取排序后 `round(0.9×(n-1))` 位置；`中位` 为 `statistics.median`。
  语料分布在“有实体加成/无实体加成”两组间双峰，因此**总中位数参考意义有限**，以分组数据和 P90 为准。

### 2.3 已修正的方法学问题

初版语料生成器 `tail = TAILS[(index * 7) % len(TAILS)]` 恒等于 `TAILS[0]`，导致语料超过约 980 条后
大量重复，5000 条输入实际只写入 982 条唯一向量，Qdrant 数字偏乐观。本报告已修正生成器并重跑
300/5000 两组；`tools/latency/latency_bench.py` 会打印“唯一文本/向量数”。§3.5 的规模探针与修正后的
5000 条结果互相吻合。

---

## 3. 测试结果

§3.1–3.6 是 **CPU 基线**（`~/.venvs/bumi`，torch 2.8.0+cpu）；**GPU（H20）复测在 §3.7**，
表结构与口径完全一致，可直接对照。

原始数据：

- `mem0-runtime/logs/latency-final-300.json` / `.txt`（300 条，8 查询 × 20 次 + 引擎前台）
- `mem0-runtime/logs/latency-final-300-assembly.json` / `.txt`（300 条，串行拼装分解）
- `mem0-runtime/logs/latency-final-5000.json` / `.txt`（5000 条，8 查询 × 20 次 + 引擎前台）
- `mem0-runtime/logs/latency-final-5000-assembly.json` / `.txt`（5000 条，串行拼装分解）
- `mem0-runtime/logs/latency-qdrant-scale.json` / `.txt`（Qdrant 本地检索规模伸缩）
- §3.7 GPU：`mem0-runtime/logs/latency-gpu-300*.json|txt`、`latency-gpu-5000*.json|txt`、
  `latency-qdrant-scale-gpu.json|txt`

### 3.1 总检索时延（`memory.search`，n=160）

| 语料（唯一向量） | 均值 | 中位 | **P90** | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
| 300 | 104.6 | 99.4 | **156.8** | 157.7 | 167.1 |
| 5000 | 344.8 | 342.4 | **393.2** | 395.2 | 601.8 |

单位 ms。两组都远超 60ms；且 5000 条相对 300 条，P90 劣化 2.5 倍。

按“是否触发实体加成”分组（双峰分布的真实形态）：

| 语料 | 分组 | 样本 | 均值 | 中位 | P90 |
|---|---|---:|---:|---:|---:|
| 300 | 无实体加成（4 条查询） | 80 | 74.2 | 73.5 | 76.7 |
| 300 | 有实体加成（4 条查询） | 80 | 135.0 | 133.0 | 157.7 |
| 5000 | 无实体加成 | 80 | 313.4 | 311.1 | 321.4 |
| 5000 | 有实体加成 | 80 | 376.3 | 368.6 | 395.2 |

300 条时，即使完全不含实体加成的查询分组，P90 也已经是 **76.7ms**，仍高于 60ms。

### 3.2 分阶段耗时（决定瓶颈在哪里）

300 条（单位 ms）：

| 阶段 | 样本 | 均值 | 中位 | P90 | 最大 |
|---|---:|---:|---:|---:|---:|
| spaCy 词形还原 | 160 | 2.5 | 2.5 | 2.7 | 3.4 |
| spaCy 实体抽取 | 160 | 4.7 | 4.7 | 5.2 | 5.7 |
| **稠密查询编码（Qwen3，CPU）** | 160 | 50.3 | 48.3 | **64.4** | 71.1 |
| 语义路 Qdrant 检索 | 160 | 11.2 | 11.2 | 11.4 | 13.2 |
| BM25 路检索（含稀疏编码） | 160 | 7.8 | 7.9 | 8.4 | 8.5 |
| **实体加成检索** | 80 | 55.2 | 54.1 | **66.5** | 68.7 |
| └ 其中实体批量编码 | 80 | 52.4 | 51.7 | 63.8 | 65.8 |
| **总计 `memory.search`** | 160 | 104.6 | 99.4 | **156.8** | 167.1 |

5000 条（单位 ms）：

| 阶段 | 样本 | 均值 | 中位 | P90 | 最大 |
|---|---:|---:|---:|---:|---:|
| 稠密查询编码（Qwen3，CPU） | 160 | 52.0 | 50.1 | 66.2 | 79.2 |
| 语义路 Qdrant 检索 | 160 | 133.2 | 131.7 | 132.9 | 376.6 |
| BM25 路检索 | 160 | 122.3 | 121.9 | 131.0 | 131.9 |
| 实体加成检索 | 80 | 57.6 | 56.3 | 69.8 | 74.9 |
| **总计 `memory.search`** | 160 | 344.8 | 342.4 | **393.2** | 601.8 |

要点：

1. **稠密查询编码单项 P90 64.4 / 66.2ms 已单独超过 60ms 预算**，且与语料规模无关；这是 CPU 形态的硬上限。
2. 实体加成是第二大可变成分：它对含实体查询再做一次 `embed_batch` + 实体库检索，把总时长从约 73ms
   推到约 157ms（300 条）。实体加成 P90 66.5ms。
3. Qdrant 检索在 300 条时只有约 8–11ms，但在 5000 条升到约 130ms，随规模线性增长（见 §3.5）。

### 3.3 引擎前台路径（0.2s 截止，真实 CLI/服务路径）

| 语料 | `prepare_prompt` 单轮耗时（ms） | 超过 0.2s 截止 | `recall_status` |
|---|---|---:|---|
| 300 | 109.0 / 93.0 / 93.6 / 182.6 | 0/4 | `ok`×4 |
| 5000 | 355.1 / 338.7 / 338.8 / 372.9 | 4/4 | `unavailable`×4 |

5000 条时前台 0.2s 截止必然触发降级：本轮回答只能使用 L1 衔接上下文，长期记忆检索结果不可用。
这正是“必须把检索提前发起”的动机；但降级只是不阻塞，并不满足 60ms 指标。

### 3.4 串行关键路径分解（5s 等待上限，n=8）

允许检索真正跑完，再把“检索”和“检索之后的串行部分”分开计时：

| 语料 | 检索 中位 / P90 | 串行（总−检索）中位 | **串行 P90** | 串行最大 | 首轮（冷启动） |
|---|---:|---:|---:|---:|---:|
| 300 | 115.7 / 339.0 | 15.4 | **16.6** | 33.3 | 33.3 |
| 5000 | 443.5 / 503.7 | 15.8 | **16.7** | 32.1 | 32.1 |

串行阶段明细（300 条，单位 ms）：

| 串行阶段 | 中位 | P90 | 最大 |
|---|---:|---:|---:|
| 排轮次/写当前问题（`reserve_turn`） | 1.4 | 2.3 | 3.3 |
| 核心状态读取（`semantic_context`） | 0.6 | 0.8 | 1.0 |
| 召回引用落库（`attach_memory_refs`） | 1.4 | 1.5 | 1.5 |
| 上下文拼装与渲染（`build_context`） | 2.6 | 3.0 | 3.0 |
| Prompt 持久化（`save_turn_prompt`） | 1.1 | 1.2 | 1.2 |
| 已插桩合计（中位） | 7.1 | — | — |
| **串行总计** | **15.4** | **16.6** | **33.3** |

要点：

1. 已插桩的 SQLite 读写与渲染合计约 7ms（中位），串行总计约 15ms，差额约 8ms 来自锁、作用域检查、
   候选过滤、引用可见性计算、JSON/响应组装等未单独插桩的部分。
2. 串行部分**与语料规模无关**：300 条与 5000 条的 P90 分别是 16.6 / 16.7ms，满足 ≤20ms。
3. 第一轮串行 32–33ms，明显高于后续 7 轮的 13.0–16.7ms，属于进程内冷启动（懒加载/空缓存）。
   部署时要么在 ready 之前预热，要么明确把首轮排除在 SLA 之外。
4. 该分解里的“检索”段走 `SerialMemoryAccess` 单工作线程，**包含排队等待**，所以 P90 339/504ms 高于
   直接调用 SDK 的 `memory.search`（156.8/393.2ms）。两者口径不同，报告同时保留；生产验收应以引擎路径为准。

### 3.5 Qdrant 本地 client 的规模伸缩（随机向量，30 次查询）

只测存储层稠密检索，不含编码；随机归一化向量，**不是语义质量证据**：

| 向量条数 | 中位 | P90 | 最大 | 写入吞吐（本地 client） |
|---:|---:|---:|---:|---:|
| 300 | 8.91 | 9.16 | 46.46 | 793 条/秒 |
| 1000 | 25.25 | 25.47 | 26.95 | 801 条/秒 |
| 3000 | 72.12 | 72.58 | 75.09 | 809 条/秒 |
| 10000 | 251.31 | 253.38 | 313.16 | 798 条/秒 |

增长近似线性（约 25ms / 1000 向量），说明 `QdrantClient(path=...)` 本地模式在做精确扫描，没有
HNSW 服务端索引。这条曲线意味着：**即使编码变成 0ms，本地 client 在 3000 条时单路检索（72.6ms P90）
就已经超 60ms**。同时也解释了修正后 5000 条语料 Qdrant 单路约 130ms 的结果。

### 3.6 冷启动与写入路径（300 条）

| 项目 | 耗时 |
|---|---:|
| `Memory.from_config` 构造（含 embedding 模型加载） | 4595.8ms |
| BM25 编码器首次加载 | 244.6ms |
| 首次 `search`（含模型/索引预热） | 1322.9ms |
| 引擎启动预热 `engine.start()` | 1212.4ms（300）/ 1704.3ms（5000） |
| 300 条批写入：稠密编码 2022.5ms（6.7ms/条）；Qdrant 写入 408.3ms | Qdrant 写入 734.7 条/秒 |
| 单条稠密编码：短(8字) / 中(60字) / 长(300字) | 中位 67.9 / 75.3 / 142.5 |
| 批量编码 batch=32 | 280.3ms → 8.8ms/条 |

写入侧结论：L1 当前轮事务（`reserve_turn`）P90 2.3ms、Prompt 持久化 P90 1.2ms，均远低于 10ms
预算；L3 批写入是离线/异步整理路径，不属于回答关键路径，当前吞吐约 735–800 条/秒（本地 client）。

### 3.7 GPU（H20）复测

环境：`~/.venvs/bumi-cuda`（`torch 2.8.0+cu128`）、`CUDA_VISIBLE_DEVICES=4`、`--embed-device cuda`；
其余（合成语料、8 条查询、`top_k=5`、Qdrant local client）与 §3.1–3.6 完全一致。原始 JSON 带
`environment` 块，可直接核对 `torch_cuda_available=true`、`resolved_embed_device=cuda`、
`torch_cuda_devices=['NVIDIA H20']`、`torch_version=2.8.0+cu128`。

原始数据：`latency-gpu-300.json|txt`、`latency-gpu-300-assembly.json|txt`、
`latency-gpu-5000.json|txt`、`latency-gpu-5000-assembly.json|txt`、`latency-qdrant-scale-gpu.json|txt`。

总检索时延（`memory.search`，n=160，单位 ms）：

| 语料 | 均值 | 中位 | **P90** | 最大 |
|---|---:|---:|---:|---:|
| 300 | 52.8 | 60.6 | **64.9** | 76.0 |
| 5000 | 210.8 | 215.2 | **221.7** | 419.8 |

分阶段（P90，单位 ms）：

| 阶段 | 300 条 | 5000 条 |
|---|---:|---:|
| spaCy 词形还原 | 1.9 | 2.3 |
| spaCy 实体抽取 | 4.2 | 4.7 |
| **稠密查询编码（Qwen3，GPU）** | **23.1** | **24.0** |
| 语义路 Qdrant 检索 | 7.1 | 89.6 |
| BM25 路检索 | 5.3 | 82.5 |
| **实体加成检索** | **24.2** | 25.0 |
| **总计** | **64.9** | **221.7** |

CPU → GPU 对照（P90，单位 ms）：

| 项目 | CPU（`bumi`） | GPU（`bumi-cuda`） | 变化 |
|---|---:|---:|---|
| 稠密查询编码，300 条 | 64.4 | 23.1 | −64% |
| 实体加成，300 条 | 66.5 | 24.2 | −64% |
| 总检索，300 条 | 156.8 | **64.9** | −59% |
| 总检索，5000 条 | 393.2 | **221.7** | −44% |
| 串行拼装 P90（300 / 5000） | 16.6 / 16.7 | **14.5 / 15.5** | 略降 |

300 条按是否触发实体加成分组：不含实体加成 P90 **41.5ms（已达标）**，含实体加成 P90 **66.2ms
（超预算）**。即 300 条这一档的超预算**完全来自实体加成的第二次编码**。

引擎前台路径（0.2s 截止）：

| 语料 | `prepare_prompt` 单轮（ms） | 超过截止 | `recall_status` |
|---|---|---:|---|
| 300 | 68.1 / 53.6 / 52.3 / 75.7 | 0/4 | `ok`×4 |
| 5000 | 234.7 / 219.6 / 220.5 / 224.1 | 4/4 | `unavailable`×4 |

串行分解（5s 等待上限，n=8）：检索 P50 56.5 / P90 73.2（300 条）、242.8 / 250.7（5000 条）；
**串行 P50 14.2 / P90 14.5（300 条）、14.7 / 15.5（5000 条）**；首轮冷启动 25.4 / 26.1ms。

GPU 环境下的 Qdrant local 规模探针（随机向量，30 次查询，P90）：300→5.9ms、1000→17.1ms、
3000→49.3ms、10000→171.8ms；同机早前一次运行是 9.2 / 25.5 / 72.6 / 253.4ms，说明这台共享机器本身
有约 ±30% 的 run-to-run 抖动，但**线性增长、“本地 client 无 HNSW”的结论不变**。

---

## 4. 分析

### 4.1 60ms 预算被哪些项吃掉

300 条、无实体加成的最简查询：词形还原 2.7 + 实体抽取 5.2 + 稠密编码 64.4 + Qdrant 11.4 +
BM25 8.4 ≈ **92ms（P90 口径，分项不可直接相加，仅示意）**，已经超过 60ms；含实体查询还要再加
一次实体批量编码约 64ms。也就是说：

- 在当前 CPU 形态下，**单次查询编码一项就足以判定不达标**；
- 本地 Qdrant 让“规模”成为第二个独立否决项；
- 实体加成让“含实体查询”成为第三个否决项。

三项互相独立，必须分别解决，不能靠“提前发起”一起绕过。GPU 复测（§3.7）后编码这一项已解决，
剩余的实体加成与本地 Qdrant 两项见 §4.3。

### 4.2 提前发起能买到什么、买不到什么

按 `t_serial = max(0, t_retrieval − t_lead) + t_assembly`，取 `t_assembly` P90 = 16.7ms，则串行路径
最多还能承受 3.3ms 的检索等待：

| 检索口径 | 检索 P90 | 需要的 `t_lead` |
|---|---:|---:|
| 引擎路径，300 条（本次） | 339.0ms | ≥ 336ms |
| 引擎路径，5000 条（本次） | 503.7ms | ≥ 500ms |
| 直接 `memory.search`，300 条（本次） | 156.8ms | ≥ 153ms |
| 直接 `memory.search`，5000 条（本次） | 393.2ms | ≥ 390ms |
| 目标形态估算：GPU 编码 + Qdrant server/HNSW + 实体加成移出热路径 | ~25–30ms（估算） | ≥ 22–27ms |

结论：如果 ASR 稳定前缀只能提前 20–30ms，**只有把检索压到 25–30ms 量级，提前发起才真正有效**；
在当前 150–500ms 的检索上，提前发起既买不到 60ms 达标，也买不到串行 ≤20ms（会退化成“等检索”）。
另外，前缀与终稿不一致时的缓存复用规则必须单独定义（见附录 B），否则会返回基于半句话的错误候选。

### 4.3 GPU 复测后的差距（实测，不再是估算）

GPU 复测把问题切成两个互不依赖的剩余项：

| 规模 | 剩余瓶颈 | 证据 | 修复方向 |
|---|---|---|---|
| 300 条 | **实体加成** | 无实体查询 P90 41.5ms（达标），含实体查询 66.2ms；实体加成自身 P90 24.2ms，是第二次编码 | 并入同一次前向 / 后置到下一轮 / 低置信才触发；或复用查询向量 |
| 5000 条 | **Qdrant local client** | 稠密路 P90 89.6ms + BM25 路 82.5ms，编码只占 24ms | 换 Qdrant Server + HNSW + payload 索引 |

另外，`docs/stage0/REPORT.md` 的独立探针（CUDA FP16 单条编码 P95 约 10.3ms）说明编码还有进一步压缩空间
（本次 23.1ms 含批处理与框架开销），但**它已经不是当前的限制项**，继续优化编码不会让 5000 条达标。

---

## 5. 结论与整改建议

### 5.1 结论

1. **GPU 编码解决了最大单项。** 稠密编码 P90 64.4→23.1ms；300 条完整检索 P90 156.8→**64.9ms**，
   5000 条 393.2→**221.7ms**。
2. **检索 P90 ≤60ms 仍未通过，且两个规模的瓶颈不同。**
   - 300 条差在**实体加成**：无实体 P90 41.5ms（达标），含实体 66.2ms，超预算部分全部来自那次额外编码。
   - 5000 条差在**Qdrant local client**：稠密 89.6 + BM25 82.5ms 的精确扫描，编码已不是主因。
3. **串行关键路径 ≤20ms 通过（有保留）。** GPU 下 P90 14.5（300 条）/15.5ms（5000 条），n=8；
   首轮冷启动 25–26ms 需预热或豁免；样本量小，需要扩到 ≥50 轮回归。
4. **5000 条前台 0.2s 截止仍 4/4 降级**（`recall_status=unavailable`），当前形态不应直接面对 60ms 验收。
5. **60ms 是部署形态指标。** GPU 已就位，剩余两项（实体加成、Qdrant Server/HNSW）各自解决后即可复测判定。

### 5.2 整改建议

| 优先级 | 事项 | 验收标准 |
|---|---|---|
| P0 | Qdrant Server + HNSW + payload 索引；本地 client 仅限开发/小数据 | ≥10k 向量、带 `user_id/agent_id` 过滤，单路 P90 ≤15ms（当前 89.6ms） |
| P0 | 实体加成移出首轮热路径：并入同一批前向、后置到下一轮，或低置信才触发 | 含实体与不含实体查询的检索 P90 差 ≤10ms（当前 41.5 vs 66.2ms） |
| P0 | GPU 部署固化：`--embed-device cuda` + fail-fast 不回退；观测实际 device | `resolved_embed_device=cuda`、编码 P90 ≤25ms（已达成 23.1ms） |
| P1 | 一次查询只编码一次，复用查询向量做实体库检索 | 实体类查询不再出现第二次 24ms（GPU）编码 |
| P1 | 实现流式前缀预取契约（附录 B）并打点 | 预取 P90 ≤60ms；命中率/陈旧重算率可观测；预取失败不影响 prepare |
| P1 | 串行拼装回归：把 `--assembly` 纳入部署验收/CI | 排除首轮后串行 P90 ≤20ms，样本 ≥50 轮 |
| P1 | 扩展预热：覆盖 BM25、spaCy、实体编码器与首轮 Qdrant | ready 之后第一轮不再出现 32ms 串行 / 1.3s 首查 |
| P2 | 服务端观测 `prefetch_wait_ms`、`prefetch_age_ms`、`assembly_ms`、`recall_status` | 线上可分解首轮/稳态时延，能定位是预取未命中还是拼装劣化 |

---

## 6. 限制与未验证项

GPU 复测已把结论推进到“编码不再是瓶颈”，但**仍不能证明目标部署已达标**：

- GPU 复测是**单卡（H20，`CUDA_VISIBLE_DEVICES=4`）、单进程、无并发**；多用户并发下的编码排队未测。
- CPU 基线与 GPU 复测用了两个 venv（`bumi` = torch+cpu，`bumi-cuda` = torch+cu128），
  依赖版本除 torch 外来自同一份 freeze；GPU 只测了 `--embed-device cuda` 一条路径，
  没有对比半精度/量化/`torch.compile` 等变体。
- 向量库在两组测试里都是 **Qdrant local client**，不是服务端 HNSW/payload 索引；5000 条的结论主要受它支配。
- 合成语料 + 8 条固定查询，不是真实记忆分布；300/5000 条远小于生产规模。
- 串行分解 n=8，样本偏小；首轮冷启动与稳态混在一起，需要用 ≥50 轮、排除首轮的统计复测。
- 不含 HTTP/Pydantic/JSON 序列化/反向代理、ASR、声纹、DDS 播放和回答模型；这些是另外的指标。
- 未做并发、长稳、内存增长、进程重启恢复、多用户同时检索的压测。
- 未测真实写入端到端（`remember`/后台抽取/`forget`）及其对检索线程的占用。
- 未测 ASR 前缀的稳定时间与“前缀→终稿”的编辑距离分布，因此无法给出预取命中率保证。
- 5000 条规模结果使用修正后的唯一语料生成器；300 条另有两次旧生成器结果（P90 148.1 / 150.4ms）作为重复性参考。

---

## 7. 复现

```bash
# 0a. CPU 基线：依赖仓库 pyproject 的 runtime；脚本会自动 re-exec 到 ~/.venvs/bumi
# 0b. GPU 复测：先建独立环境（不改 CPU 基线）
/usr/bin/python3 -m venv ~/.venvs/bumi-cuda
TMPDIR=~/.tmp/pip-tmp ~/.venvs/bumi-cuda/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
TMPDIR=~/.tmp/pip-tmp ~/.venvs/bumi-cuda/bin/pip install -r <(原 venv 的 pip freeze，去掉 torch 行)
TMPDIR=~/.tmp/pip-tmp ~/.venvs/bumi-cuda/bin/pip install --no-deps -e ~/git/Bumi_Mem
# 注意：pip 的 TMPDIR 必须在 /data1 上；本机 / 只有 20G，默认 /tmp 会被 2GB 的 torch 写满
# 跑 GPU 时统一加前缀（选一张空闲卡，并把 clash 的 socks 代理去掉）
export GPU_PREFIX="env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES=4 MEM0_PYTHON=$HOME/.venvs/bumi-cuda/bin/python"

# 1. 300 条：冷启动 / 写入 / 分阶段检索 / 引擎前台
$GPU_PREFIX tools/latency/latency_bench.py --corpus 300 --repeats 20 --with-engine --embed-device cuda \
  --data-root ~/mem0-runtime/latency-bench-final300 \
  --json ~/mem0-runtime/logs/latency-gpu-300.json

# 2. 串行拼装分解（必须独立进程：本地 Qdrant 单实例锁）
$GPU_PREFIX tools/latency/latency_bench.py --assembly --embed-device cuda \
  --data-root ~/mem0-runtime/latency-bench-gpu300 --assembly-read-timeout 5.0 \
  --json ~/mem0-runtime/logs/latency-gpu-300-assembly.json

# 3. 5000 条规模复测（同 1、2，换 data-root）
$GPU_PREFIX tools/latency/latency_bench.py --corpus 5000 --repeats 20 --with-engine --embed-device cuda \
  --data-root ~/mem0-runtime/latency-bench-gpu5000 \
  --json ~/mem0-runtime/logs/latency-gpu-5000.json
$GPU_PREFIX tools/latency/latency_bench.py --assembly --embed-device cuda \
  --data-root ~/mem0-runtime/latency-bench-gpu5000 --assembly-read-timeout 5.0 \
  --json ~/mem0-runtime/logs/latency-gpu-5000-assembly.json

# 4. Qdrant 本地 client 规模伸缩探针（随机向量，只测存储层）
$GPU_PREFIX tools/latency/latency_bench.py --scale-probe 300,1000,3000,10000 \
  --data-root ~/mem0-runtime/latency-scale-qdrant-gpu \
  --json ~/mem0-runtime/logs/latency-qdrant-scale-gpu.json
```

CPU 基线把 `$GPU_PREFIX` 换成 `MEM0_PYTHON=$HOME/.venvs/bumi/bin/python`、去掉 `--embed-device cuda`
并把 `--data-root/--json` 换回 `*-final*` 即可。

注：脚本在解释器退出时可能打印一行 `terminate called without an active exception`，发生在 JSON 落盘之后，
不影响计时与结果；如影响采集可重定向 stderr。

注：如果运行 shell 导出了 `ALL_PROXY=socks5://…`（clash 默认会设置），mem0 在初始化时构造 OpenAI
客户端会因缺少 `httpx[socks]` 直接报 `ImportError: Using SOCKS proxy...`。运行前 `unset ALL_PROXY all_proxy`
或 `pip install "httpx[socks]"`，否则压测和服务都会起不来。

---

## 附录 A：分查询明细（中位，ms）

300 条：

| 查询 | 总中位 | 语义路 | BM25 路 | 实体加成 | 命中 |
|---|---:|---:|---:|---:|---:|
| 咖啡 | 71.2 | 11.3 | 7.9 | 0.0 | 5 |
| 我住在哪个城市？ | 73.2 | 11.2 | 7.8 | 0.0 | 5 |
| 我早上几点起床？ | 73.3 | 11.2 | 7.8 | 0.0 | 5 |
| 推荐一家安静、适合工作的咖啡馆，我下午三点到五点有空 | 157.7 | 11.2 | 7.5 | 66.4 | 5 |
| 我朋友小王最近在忙什么？ | 129.2 | 11.2 | 7.9 | 52.1 | 5 |
| 机械键盘 | 116.6 | 11.2 | 7.9 | 46.1 | 5 |
| What coffee does the user like? | 76.4 | 11.2 | 8.4 | 0.0 | 5 |
| 我周末想去西湖边跑步，顺便看看天气怎么样 | 134.4 | 11.3 | 7.5 | 54.9 | 5 |

5000 条：

| 查询 | 总中位 | 语义路 | BM25 路 | 实体加成 | 命中 |
|---|---:|---:|---:|---:|---:|
| 咖啡 | 308.5 | 130.6 | 122.5 | 0.0 | 5 |
| 我住在哪个城市？ | 311.1 | 132.2 | 121.2 | 0.0 | 5 |
| 我早上几点起床？ | 310.4 | 132.7 | 121.2 | 0.0 | 5 |
| 推荐一家安静、适合工作的咖啡馆，我下午三点到五点有空 | 394.8 | 132.1 | 117.5 | 69.7 | 5 |
| 我朋友小王最近在忙什么？ | 368.9 | 131.7 | 122.6 | 54.3 | 5 |
| 机械键盘 | 357.5 | 131.2 | 123.7 | 48.1 | 5 |
| What coffee does the user like? | 321.0 | 130.9 | 131.3 | 0.0 | 5 |
| 我周末想去西湖边跑步，顺便看看天气怎么样 | 368.2 | 131.4 | 117.9 | 56.6 | 5 |

## 附录 B：流式前缀预取接口草案（待评审/未实现）

目标：在 ASR 稳定前缀出现时提前发起检索，让 60ms 检索与说话/识别重叠；终稿到达后串行只做拼装。
该草案只做候选预热，**不写 L1、不创建轮次、不发布共享引用**。

```json
POST /v1/memory/context
{
  "operation": "prefetch",
  "client_turn_id": "run-c7d3:turn-000001",
  "speaker": {"status": "matched", "profile_id": "profile-001"},
  "query_prefix": "我周末想去西湖",
  "prefix_revision": 3
}
```

返回（不返回 messages，避免被误用为回答上下文）：

```json
{
  "operation": "prefetch",
  "prefetch_id": "pf_<server-id>",
  "client_turn_id": "run-c7d3:turn-000001",
  "recall_status": "ok",
  "candidate_count": 5,
  "expires_at": "2026-09-22T10:00:03+08:00"
}
```

复用规则（`prepare` 携带可选 `prefetch_id`）：

1. 同一 `agent_id` + 同一说话人（`profile_id`/`user_id`），且 `client_turn_id` 一致；
2. 终稿在归一化后以 `query_prefix` 为前缀，且新增字符数 ≤ 阈值（初值 12 字）；
3. 预取年龄 ≤ TTL（初值 3s），期间无删除/更正导致的 context epoch 变化；
4. 命中则复用候选并重新渲染/落引用；任一条件不满足则静默重算，`prepare` 不因预取失败而失败。

预算与观测：预取分支 P90 ≤60ms；命中率、陈旧重算率、`prefetch_wait_ms`、`assembly_ms` 必须打点。
安全边界与现有一致：预取不绕过 `user_id/agent_id` 门控，不提前发布 `shared_memory_refs`，
`forget_all()` 提升 epoch 后缓存立即失效。
