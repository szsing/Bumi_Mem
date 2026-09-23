# Bumi 记忆读写时延压测报告

| 项目 | 内容 |
|---|---|
| 任务 | Memory-B1-S4「定义记忆读写 API 与延迟预算」（链 A 并行分支） |
| 验收指标 | 检索 P90 ≤ 60ms（#7）；串行关键路径只计最后的上下文拼装，≤ 20ms |
| 测试日期 | 2026-09-22 ~ 09-23 |
| 执行人 | 张冉 |
| 被测版本 | Bumi_Mem `6315aaf`（main，schema 7 / v0.4 Agent API）、mem0 2.0.20 |
| 压测脚本 | `tools/latency/latency_bench.py`（等价于本机 `~/.local/bin/bumi-latency-bench`） |
| 结论 | **检索 P90 ≤ 60ms：达标**（GPU + Qdrant Server + `BUMI_ENTITY_BOOST=off`）：300/5000/20000 条 P90 = **35.0 / 35.9 / 37.1ms**；**串行拼装 ≤ 20ms：通过**（P90 14.2–15.5ms） |

> 一句话结论：目标部署形态是 **H20 + GPU 编码 + Qdrant Server**。本报告给三档实测：CPU 基线、
> GPU 复测、GPU + Qdrant Server 复测（`~/.venvs/bumi-cuda`，torch 2.8.0+cu128，`CUDA_VISIBLE_DEVICES=4`，
> `BUMI_QDRANT_URL=http://127.0.0.1:6333`，Qdrant Server 1.19.1）。
> 完整检索 P90 的演进：CPU 156.8/393.2ms → GPU+本地 client 64.9/221.7ms →
> **GPU+Server 62.2/66.8/63.0ms（300/5000/20000 条）**：
>
> - **规模问题已经解决**：Qdrant Server 让 300→20000 条的检索 P90 基本持平（62.2/66.8/63.0ms），
>   稠密路 P90 稳定在 4–6ms、BM25 路 1.6–1.9ms；本地 client 在 5000 条时这两项还是 89.6/82.5ms。
> - **只剩实体加成这一项**：不含实体加成的查询 P90 只有 **36.4–40.9ms（三档全部达标）**，
>   含实体加成的是 **63.3–67.5ms**；实体加成本身 P90 26.3–29.5ms，是一次额外的批量编码。
>   把它并入同一次前向、后置到下一轮或低置信才触发，即可让所有查询进入 60ms。
> - 串行关键路径 P90 **14.2ms（5000 条，Server）**，满足 ≤20ms；首轮冷启动 25–26ms 仍需预热或豁免。
> - 尾延迟仍要盯：三档的 max 是 167.8/176.1/179.1ms，来自冷启动/单次慢样本，验收时建议同时看 P99。
>
> 结论：三个否决项全部解决——CPU 编码 → GPU、本地 client → Qdrant Server、实体加成移出热路径
> （`BUMI_ENTITY_BOOST=off`，消融与 ID 级 A/B 都显示没有召回损失，见 §3.9）。**检索 P90 达标**
> （35.0 / 35.9 / 37.1ms），串行拼装 14.2ms 达标。提前发起（附录 B）仍是把检索移出串行路径的独立优化，
> 不是达标的前提。
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
| B1 | L3 检索（预取分支，目标部署/目标规模） | P90 ≤ 60ms | **通过**：GPU+Server+`BUMI_ENTITY_BOOST=off` 下 35.0 / 35.9 / 37.1ms（300/5000/20000 条），见 §3.9 |
| B2 | 串行上下文拼装（ASR 终稿 → memory_context） | P90 ≤ 20ms | 通过：CPU 16.6ms、GPU 14.5ms、GPU+Server 14.2ms（首轮冷启动 25–33ms 待处理） |
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
| 向量库 | 两档：**Qdrant local client**（`path=...`，单进程精确扫描）用于 §3.1–3.7；**Qdrant Server 1.19.1**（`BUMI_QDRANT_URL=http://127.0.0.1:6333`，HNSW + payload 索引，二进制 `~/.local/opt/qdrant-musl-1.19.1/qdrant`）用于 §3.8 |
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

### 3.8 GPU + Qdrant Server 复测

服务端：Qdrant **1.19.1**（musl 静态二进制），`127.0.0.1:6333`（REST）/ `6334`（gRPC），storage 在
`mem0-runtime/qdrant-server/storage`。切换方式：环境变量
`BUMI_QDRANT_URL=http://127.0.0.1:6333`（实现见 `src/bumi_mem/long_term.py`；不设置时仍是本地
path client）。写入改为 **500 条/批**——Server 单请求上限 32MB，5000 条整批是 84MB，会直接 400。

总检索时延（`memory.search`，n=160，单位 ms）：

| 语料 | 实际点位数 | 均值 | 中位 | **P90** | 最大 | 语义路 P90 | BM25 路 P90 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 300 | 49.8 | 56.2 | **62.2** | 167.8 | 4.8 | 1.8 |
| 5000 | 5000 | 52.5 | 57.2 | **66.8** | 176.1 | 4.0 | 1.6 |
| 20000 | 20000 | 51.0 | 60.9 | **63.0** | 179.1 | 5.5 | 1.9 |

分组（P90，单位 ms）：

| 语料 | 不含实体加成 | 含实体加成 | 稠密编码 | 实体加成 |
|---:|---:|---:|---:|---:|
| 300 | **40.9（达标）** | 64.4 | 26.8 | 27.5 |
| 5000 | **39.2（达标）** | 67.5 | 26.7 | 29.5 |
| 20000 | **36.4（达标）** | 63.3 | 23.6 | 26.3 |

后端对照（P90，单位 ms）：

| 规模 | GPU + 本地 client | GPU + Server | 说明 |
|---:|---:|---:|---|
| 300 | 64.9 | **62.2** | 小数据下本地 client 已不慢，差距主要在 BM25 路 5.3→1.8 |
| 5000 | 221.7 | **66.8** | 本地 client 稠密 89.6 + BM25 82.5 → Server 4.0 + 1.6 |
| 20000 | 未测（本地 10000 条单路已 171.8ms） | **63.0** | Server HNSW 把检索与规模解耦 |

引擎前台 0.2s 截止：三档都是 0/4 超时、`recall_status=ok`，`prepare_prompt` 中位
52.9 / 56.4 / 55.9ms。串行分解（5000 条，n=8）：检索 P50 49.7 / P90 70.9ms；
**串行 P50 13.9 / P90 14.2ms**，首轮 25.7ms。

尾延迟：三档 max 167.8 / 176.1 / 179.1ms，`entity_boost` 出现单次 125–128ms 慢样本（冷启动/偶发）。
建议验收同时看 P99，不要只看 P90。

### 3.9 实体加成移出热路径（`BUMI_ENTITY_BOOST=off`）

**质量证据（先证明可以关）**：

- 消融实验 `mem0-runtime/logs/retrieval-ablation-20260922-184417.*`（317 条记忆 / 13 条查询）比较
  「语义 only」「语义 + BM25」「完整（含实体加成）」三种配置，**Recall 0.92、MRR 0.71、逐查询命中排名
  三者完全相同**——两条旁路在中文语料上没有召回收益。
- 本次在 20000 条 GPU + Server 上做 ID 级 A/B：8 条查询在实体加成 on/off 下返回的 top-5
  **ID 与顺序 8/8 完全一致**（Jaccard 1.00）。

**实现**：`BUMI_ENTITY_BOOST=on|off`（默认 `on`，保持原行为）。`off` 时只在检索热路径跳过
`_compute_entity_boosts`，实体库仍照常写入，可随时切回；代码见 `src/bumi_mem/long_term.py`，
环境证据写入 JSON 的 `environment.entity_boost`。

注意：**实体加成是 mem0 2.0.20 自带的行为**（`Memory.search` → `extract_entities` →
`_compute_entity_boosts` → `mem0/utils/scoring.py` 把它加进 `semantic + bm25 + entity_boost`），
上游没有开关，我们的 `off` 是给实例换个 no-op 的内部钩子。因此升级 mem0 后必须复核：
若该方法改名/移除，`off` 会**显式报错**而不是静默失效（`tests/test_entity_boost_policy.py`）。
另外 `extract_entities` 在查询侧仍会跑（约 4ms），`off` 省掉的是第二次批量编码与实体库检索。

**时延（GPU + Qdrant Server，n=160，P90）**：

| 规模 | 实体加成 on | 实体加成 off | 降幅 |
|---:|---:|---:|---:|
| 300 | 62.2ms | **35.0ms** | −44% |
| 5000 | 66.8ms | **35.9ms** | −46% |
| 20000 | 63.0ms | **37.1ms** | −41% |

`off` 后所有分查询的中位都落在 33–36ms；引擎前台 0.2s 截止三档 0/4 超时、`recall_status=ok`。
原始数据：`latency-eboff-300.*`、`latency-eboff5000.*`、`latency-eboff20000.*`。

**结论：在 GPU + Qdrant Server + `BUMI_ENTITY_BOOST=off` 这一配置下，检索 P90 ≤60ms 达标**
（35.0 / 35.9 / 37.1ms）。

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

### 4.3 三个否决项都已解决

（开）实体加成时三个否决项解决两个，关掉后是三个全解决：

| 项 | 状态 | 证据 |
|---|---|---|
| 查询编码 | ✅ GPU 已解决 | P90 64.4 → 23.1ms |
| 向量库规模 | ✅ Qdrant Server 已解决 | 300 / 5000 / 20000 条检索 P90 = 62.2 / 66.8 / 63.0ms；本地 client 在 5000 条是 221.7ms |
| 实体加成 | ✅ 已移出热路径 | `BUMI_ENTITY_BOOST=off` 后三档 P90 = **35.0 / 35.9 / 37.1ms**；消融与 ID 级 A/B 均无召回差异 |

最终配置：**GPU 编码 + Qdrant Server/HNSW + `BUMI_ENTITY_BOOST=off`**，检索 P90 35–37ms、
串行拼装 P90 14.2ms，两项预算都达标。后续如果要把实体加成重新上热路径，应改成
「并入同一次前向 / 后置到下一轮 / 低置信才触发」，并重新验证召回与 P90。
`docs/stage0/REPORT.md` 的独立探针（CUDA FP16 单条编码 P95 约 10.3ms）说明编码还有压缩空间
（本次 23.1ms 含批处理与框架开销），但它已不是限制项。

---

## 5. 结论与整改建议

### 5.1 结论

1. **三个否决项全部解决。** GPU 把编码 P90 从 64.4 降到 23.1ms；Qdrant Server 把规模解耦
   （本地 client 5000 条 221.7ms → Server 66.8ms）；`BUMI_ENTITY_BOOST=off` 把实体加成那次
   额外编码（P90 26.3–29.5ms）移出热路径。
2. **检索 P90 ≤60ms 达标。** GPU + Qdrant Server + `BUMI_ENTITY_BOOST=off` 下，300/5000/20000 条
   P90 = **35.0 / 35.9 / 37.1ms**；消融与 ID 级 A/B 显示关掉实体加成没有召回损失。
3. **串行关键路径 ≤20ms 通过（有保留）。** CPU 16.6 / GPU 14.5 / GPU+Server 14.2ms（n=8）；
   首轮冷启动 25–26ms 需预热或豁免；样本量小，需要扩到 ≥50 轮回归。
4. **前台 0.2s 截止在 GPU+Server 下三档 0/4 超时、`recall_status=ok`**，不再是降级状态。
5. **尾延迟要单独盯。** 开实体加成时三档 max 167.8–179.1ms；关闭后 300/5000 条 max 降到
   39.9 / 41.0ms，20000 条仍有单次 110.3ms 慢样本。验收建议同时看 P99。

### 5.2 整改建议

| 优先级 | 事项 | 验收标准 |
|---|---|---|
| P0 | 部署配置里设 `BUMI_ENTITY_BOOST=off`（代码默认仍是 `on`，避免改既有行为）；要恢复必须先做召回回归 | 含实体与不含实体的检索 P90 差 ≤10ms（已达成：全部分查询 33–36ms） |
| P0 | Qdrant Server + HNSW + payload 索引固化到部署（`BUMI_QDRANT_URL`）；本地 client 仅限开发 | ≥20k 向量、带 `user_id/agent_id` 过滤，单路 P90 ≤15ms（已达成 4–6ms，需写入部署文档/服务配置） |
| P0 | GPU 部署固化：`--embed-device cuda` + fail-fast 不回退；观测实际 device | `resolved_embed_device=cuda`、编码 P90 ≤25ms（已达成 23.1ms） |
| P2 | 若将来要恢复实体加成：并入同一次前向 / 后置到下一轮 / 低置信才触发，并回归召回 | 恢复后含实体查询额外开销 ≤10ms，Recall/MRR 不低于关闭时 |
| P1 | 尾延迟：定位首轮冷启动与偶发慢样本 | P99 ≤100ms，首轮不进入 SLA 或有预热 |
| P1 | 实现流式前缀预取契约（附录 B）并打点 | 预取 P90 ≤60ms；命中率/陈旧重算率可观测；预取失败不影响 prepare |
| P1 | 串行拼装回归：把 `--assembly` 纳入部署验收/CI | 排除首轮后串行 P90 ≤20ms，样本 ≥50 轮 |
| P1 | 扩展预热：覆盖 BM25、spaCy、实体编码器与首轮 Qdrant | ready 之后第一轮不再出现 32ms 串行 / 1.3s 首查 |
| P2 | 服务端观测 `prefetch_wait_ms`、`prefetch_age_ms`、`assembly_ms`、`recall_status` | 线上可分解首轮/稳态时延，能定位是预取未命中还是拼装劣化 |

---

## 6. 限制与未验证项

GPU + Qdrant Server 复测已把结论推进到“只剩实体加成一项”，但**仍不能证明目标部署已达标**：

- GPU 复测是**单卡（H20，`CUDA_VISIBLE_DEVICES=4`）、单进程、无并发**；多用户并发下的编码排队未测。
- CPU 基线与 GPU 复测用了两个 venv（`bumi` = torch+cpu，`bumi-cuda` = torch+cu128），
  依赖版本除 torch 外来自同一份 freeze；GPU 只测了 `--embed-device cuda` 一条路径，
  没有对比半精度/量化/`torch.compile` 等变体。
- Qdrant Server 是**单机单节点**、默认参数、无副本/无 TLS/无鉴权，且与压测进程同机（本机回环）；
  分布式、跨机网络、慢盘与容量上限都未验证。Server 写入改为 500 条/批是为了绕开 32MB 单请求上限。
- 合成语料 + 8 条固定查询，不是真实记忆分布；20000 条仍小于真实长期记忆规模（趋势已持平，但未测 10 万级）。
- 实体加成 `off` 的召回结论来自合成语料（317 条消融 + 20000 条 ID 级 A/B、8 条查询）；
  真实用户语料仍需回归，不能据此断言所有语料都无收益。
- 串行分解 n=8，样本偏小；首轮冷启动与稳态混在一起，需要用 ≥50 轮、排除首轮的统计复测。
- 不含 HTTP/Pydantic/JSON 序列化/反向代理、ASR、声纹、DDS 播放和回答模型；这些是另外的指标。
- 未做并发、长稳、内存增长、进程重启恢复、多用户同时检索的压测。
- 未测真实写入端到端（`remember`/后台抽取/`forget`）及其对检索线程的占用。
- 未测 ASR 前缀的稳定时间与“前缀→终稿”的编辑距离分布，因此无法给出预取命中率保证。
- 5000 条规模结果使用修正后的唯一语料生成器；300 条另有两次旧生成器结果（P90 148.1 / 150.4ms）作为重复性参考。
- 尾延迟未系统采样：三档 max 167.8–179.1ms，P99 需要单独用更长样本量复测。
- 后台整理与前台检索共享同一个 SDK 线程：50 轮里约 5% 轮次被整理任务阻塞（最长 1s），
  修复方案（延后整理/独立进程/优先队列）已给出但未实现验证。
- 单进程 8 路并发检索在 0.2s 截止下有 40% 降级；跨 agent 的真实并发与长稳未测。

---

## 7. 复现

本目录三个工具：`latency_bench.py`（时延基准：检索/串行/并发/规模探针）、
`recall_ablation.py`（召回对照：语义 / +BM25 / +实体加成）、
`preflight_check.py`（验收前环境预检，见 §8.1）。

部署脚本：`scripts/qdrant_server.sh`（启停/状态/日志/copytruncate 轮转）、
`scripts/make_service_env.py`（生成仓库外的 service.env）、`scripts/run_memory_service.sh`（起服务）。

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

# 5. Qdrant Server：起服务端（1.19.1 musl 静态版，REST 6333 / gRPC 6334）
curl -fL -o /tmp/qdrant.tar.gz \
  https://github.com/qdrant/qdrant/releases/download/v1.19.1/qdrant-x86_64-unknown-linux-musl.tar.gz
mkdir -p ~/.local/opt/qdrant-musl-1.19.1 && tar -xzf /tmp/qdrant.tar.gz -C ~/.local/opt/qdrant-musl-1.19.1
cd ~/.local/opt/qdrant-musl-1.19.1 && QDRANT__STORAGE__STORAGE_PATH=~/mem0-runtime/qdrant-server/storage \
  setsid nohup ./qdrant > ~/mem0-runtime/logs/qdrant-server.log 2>&1 < /dev/null &
curl -sS http://127.0.0.1:6333/healthz   # healthz check passed / all shards are ready

# 6. 达标配置：Server + 实体加成 off（不同规模用不同 --collection，避免互相污染）
export SRV="env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES=4 \
  MEM0_PYTHON=$HOME/.venvs/bumi-cuda/bin/python BUMI_QDRANT_URL=http://127.0.0.1:6333 \
  BUMI_ENTITY_BOOST=off"
$SRV tools/latency/latency_bench.py --corpus 5000 --repeats 20 --with-engine --embed-device cuda \
  --collection srv5000 --data-root ~/mem0-runtime/latency-bench-srv5000 \
  --json ~/mem0-runtime/logs/latency-srv-5000.json
$SRV tools/latency/latency_bench.py --corpus 20000 --repeats 20 --with-engine --embed-device cuda \
  --collection srv20000 --data-root ~/mem0-runtime/latency-bench-srv20000 \
  --json ~/mem0-runtime/logs/latency-srv-20000.json
$SRV tools/latency/latency_bench.py --assembly --embed-device cuda --collection srv5000 \
  --data-root ~/mem0-runtime/latency-bench-srv5000 --assembly-read-timeout 5.0 \
  --json ~/mem0-runtime/logs/latency-srv-5000-assembly.json
```

CPU 基线把 `$GPU_PREFIX` 换成 `MEM0_PYTHON=$HOME/.venvs/bumi/bin/python`、去掉 `--embed-device cuda`
并把 `--data-root/--json` 换回 `*-final*` 即可。

注：脚本在解释器退出时可能打印一行 `terminate called without an active exception`，发生在 JSON 落盘之后，
不影响计时与结果；如影响采集可重定向 stderr。

注：如果运行 shell 导出了 `ALL_PROXY=socks5://…`（clash 默认会设置），mem0 在初始化时构造 OpenAI
客户端会因缺少 `httpx[socks]` 直接报 `ImportError: Using SOCKS proxy...`。运行前 `unset ALL_PROXY all_proxy`
或 `pip install "httpx[socks]"`，否则压测和服务都会起不来。

## 8. 验收与部署固化（四项收尾）

### 8.1 环境预检：一条命令确认“到底跑在什么环境上”

`tools/latency/preflight_check.py` 逐项检查 GPU/CUDA、实际编码设备、Qdrant 后端、集合点数、
实体加成策略，硬性项失败返回非零退出码，可直接接进部署脚本：

```bash
BUMI_QDRANT_URL=http://127.0.0.1:6333 BUMI_ENTITY_BOOST=off \
MEM0_PYTHON=$HOME/.venvs/bumi-cuda/bin/python \
tools/latency/preflight_check.py --require-gpu --require-server \
  --require-entity-boost-off --collection bumi_latency_bench
```

本次结果（`mem0-runtime/logs/preflight-gpu-server.json`）：11 项全部 PASS，退出码 0。

### 8.2 召回回归：合成语料已过，真实语料按同一套流程

**语料分级（先说清各自的证据力，避免把机翻数据当真实验收）**：

| 语料 | 来源 | 语言 | 能证明 | 不能证明 |
|---|---|---|---|---|
| `latency_bench` 合成语料 | 程序生成（混合进制模板） | 中英混合 | 时延、规模、排队 | 召回质量（无标准答案） |
| `datasets/recall_sim_zh.json` | **LLM 生成**（DeepSeek） | 中文 | 七类场景覆盖 | 真实分布；且待人工校验 |
| `datasets/recall_demo.json` | 手写小样本 | 中文 | 流程自检 | 任何统计结论 |
| LoCoMo 原版 | **真实两人对话 + 人工标注 evidence** | 英文 | 跨域召回、证据级召回 | 中文表达、本产品场景 |
| LoCoMo 中文子集 | 同上 + **机器翻译** | 中文 | 中文跨域召回、明显退化筛查 | 真实中文口语分布（机翻伪影） |
| 真实用户语料（待补） | 我们自己的用户（脱敏） | 中文 | 最终验收 | — |

LoCoMo 原始对话和 evidence 标注是真人真标注，**数据来源是真实的**；但中文版是我们机翻的，
场景也是欧美日常（人名、LGBTQ 互助小组、慈善跑），和 Bumi 的中文陪伴场景分布不同。
所以它应归类为**跨域回归**，不能替代真实用户语料验收；它最有价值的地方是：有 evidence 级标准答案、
规模够大、能筛出"关掉某条检索路会不会掉召回"这类明显退化。

已做：

- 2000 条干扰记忆 + 13 条带标准答案查询 × 10 次（GPU + Server，`tools/latency/recall_ablation.py`）：
  语义 only / +BM25 / 含实体加成三档 **Recall 0.92、MRR 0.69、逐查询命中排名完全相同**，
  中位时延 32.1 / 33.7 / 33.7ms。
- 20000 条上实体加成 on/off 的 ID 级 A/B：**8/8 查询 top-5 ID 与顺序完全一致**。

真实语料上线前的流程（已固定）：把脱敏记忆灌进隔离集合，准备 ≥30 条带标准答案的查询（覆盖称谓、
地点、时间、偏好、计划、否定/更正、跨语言），`BUMI_ENTITY_BOOST=on` 与 `off` 各跑一次
`recall_ablation.py`。通过标准：

1. Recall@5 与 MRR 不低于关闭前（允许 ±0.02 噪声）；
2. 检索 P90 ≤60ms、P99 ≤100ms；
3. 任一查询命中排名变差 → 该语料不适用“关实体加成”，改用“同一次前向/后置到下一轮”的方案。

工具已支持外部语料，不需要改代码：把数据整理成

```json
{"memories": ["记忆原文（脱敏后）", "..."],
 "probes": [{"query": "…", "expected": "必须与某条 memories 逐字一致的原文", "group": "分组名"}]}
```

然后：

```bash
$SRV tools/latency/recall_ablation.py --dataset /path/to/real.json \
  --collection recall_real --data-root ~/mem0-runtime/recall-real --repeats 5 \
  --json ~/mem0-runtime/logs/recall-real.json
# 再以 BUMI_ENTITY_BOOST=on 跑一次同一条命令（换 --collection / --json 便于对比）
```

仓库里有一份可直接跑通的最小示例 `tools/latency/datasets/recall_demo.json`（10 条记忆 / 11 条查询，
覆盖地点/时间/关键词/人物/编号/偏好/习惯/计划/称谓边界）；本次用它跑通：三档 Recall 1.00、
MRR 0.82、逐查询排名完全一致。真实语料建议放在 `~/mem0-runtime/recall-real/`（仓库外），
跑完把 JSON 结果落到 `~/mem0-runtime/logs/`。

#### 公开数据集 LoCoMo 回归（2026-09-23）

项目里只有 LoCoMo 的脚本与 manifest，数据集文件不在盘上（manifest 指向另一台机器的
`/home/legendre/.../locomo10.json`，`src/memory-benchmarks` 子模块为空）。本次从官方仓库
`snap-research/locomo` 下载 `locomo10.json`（2.8MB，10 段长期对话 / 5882 条话语 / 1986 条 QA），
用 `tools/latency/locomo_to_recall_dataset.py` 转成召回数据集：

- memories = 每条话语，带会话时间与说话人（`[<sample_id>] [<session_time>] <speaker>: <text>`）；
- probes = 每个带 evidence 的 QA，`expected` 是 evidence 指向的话语（多条 evidence 时为列表，
  任一命中即算召回，工具已支持多目标）；group = LoCoMo 的 category；
- 取 300 条 probe（cat1–cat5 各 60，轮询取样，结果可复现），其中 93 条是多证据。

同一批数据在 GPU + Qdrant Server 上跑三档对照（top_k=5，每条 1 次）：

| 配置 | Recall@5 | MRR | P50 | P90 |
|---|---:|---:|---:|---:|
| 语义 only | 0.513 | 0.345 | 34.5ms | 35.4ms |
| **语义 + BM25** | **0.530** | **0.375** | 37.1ms | 37.9ms |
| 完整（含实体加成） | 0.517 | 0.364 | **87.0ms** | 95.5ms |

结论（与中文合成语料互补）：

1. **BM25 在英文真实对话上有小正收益**（Recall +1.7pt、MRR +3.0pt，只贵约 2.6ms）→ 保持开启；
2. **实体加成在这份公开数据上也是负收益**（Recall −1.3pt、MRR −1.1pt），却让 P50 从 37ms 涨到 87ms
   （LoCoMo 查询是长英文句、实体多，加成开销更大）；
3. 因此“关实体加成”在中文合成语料与英文公开数据集上结论一致。**但当前部署仍按保守策略保留
   `BUMI_ENTITY_BOOST=on`**（见 §8.6），真实中文语料回归通过后再切换；切换是一行配置。

原始数据：`mem0-runtime/logs/recall-locomo.json`；
数据集与原始 LoCoMo 放在仓库外 `mem0-runtime/datasets/`。

#### LoCoMo 中文子集（规模更小，可选）

全量 LoCoMo 是 5882 条话语 / 1986 条 QA，跑一次约 2 分钟；如果只需要快速回归，转换器支持裁剪
并翻译成中文：

```bash
MEM0_PYTHON=$HOME/.venvs/bumi-cuda/bin/python \
tools/latency/locomo_to_recall_dataset.py --samples 1 --max-probes 80 \
  --translate --batch-size 30 \
  --out ~/mem0-runtime/datasets/recall_locomo_zh.json
```

- `--samples N`：只取前 N 段对话（1 段 ≈ 420–590 条话语）；另有 `--max-memories` / `--max-probes` 控规模；
- `--translate`：用 DeepSeek 逐批翻译，**先统一人名译法作为术语表**再翻正文，日期/数字保留；
  `expected` 直接复用译好的记忆原文，保证"逐字一致"这一约束不被翻译破坏；
- 翻译结果缓存在 `~/mem0-runtime/datasets/locomo_zh_cache.json`，重跑不重复计费、结果可复现。

本次产出：1 段对话 → **419 条记忆 / 80 条查询**（cat1–5 = 18/17/11/17/17，其中 24 条多证据），
约 22 次翻译请求、耗时约 1.5 分钟。三档对照（top_k=5，每条 3 次）：

| 配置 | Recall@5 | MRR | P50 | 分组（cat1/2/3/4/5） |
|---|---:|---:|---:|---|
| 语义 only | 0.49 | 0.33 | 32.4ms | 0.39 / 0.65 / 0.36 / 0.59 / 0.41 |
| 语义 + BM25 | 0.49 | 0.33 | 34.0ms | 0.39 / 0.65 / 0.36 / 0.59 / 0.41 |
| 完整（含实体加成） | 0.47 | 0.30 | 34.0ms | 0.39 / 0.65 / 0.36 / 0.53 / 0.41 |

在**中文**语料上：BM25 中性（与中文合成消融一致），实体加成略负；Recall 绝对值低于小规模示例集
是正常的——LoCoMo 对话长、事实分散，419 条记忆里还带时间前缀。原始数据：
`mem0-runtime/logs/recall-locomo-zh.json`。

#### LLM 生成的中文模拟语料（覆盖七类，待人工校验）

LoCoMo 是英文，覆盖不到中文的称谓/否定/跨语言等场景，因此另用
`tools/latency/make_sim_dataset.py`（DeepSeek `deepseek-chat`）生成一份中文模拟集：
16 条记忆 / 20 条查询，覆盖**称谓、地点、时间、偏好、计划、否定、跨语言**七类；生成后做机械校验
（`expected` 必须逐字等于某条 memory、group 必须合法、memories 不得重复），不过关就带错误重试。
产物已收进仓库 `tools/latency/datasets/recall_sim_zh.json`，文件里标了 `_comment`。

加 200 条干扰记忆后跑三档对照（top_k=5，每条 3 次）：**三档 Recall 都是 1.00、MRR 0.87–0.88**，
七个类别全部命中，P50 32–34ms。也就是说这份小集合能验证"类别覆盖"，但**区分不出三条检索路的差异**
——区分度主要来自 LoCoMo（5882 条记忆）与更早的 317 条中文消融。

**待人工校验的点**（脚本会把全部内容打印出来供核对）：生成结果里
“我生日是三月十二号吗？”“我计划十月去敦煌吗？”这类确认式查询**把答案写进了 query**，
区分度弱，建议改写成“我生日是哪天？”这类问法；另有“你该叫我什么？”与“我讨厌别人怎么叫我？”
同属称谓/边界，可按需合并。

### 8.3 尾延迟：稳态 P99 达标，真正的尾巴是后台整理

- 稳态（20000 条，8 查询 × 100 次 = 800 样本，GPU + Server + 实体加成 off）：
  检索 P50 34.7 / P90 36.0 / **P99 44.1** / max 101.0ms；max 来自单次 HNSW 67.5ms 的离群点。
- 但连跑 50 轮（400 轮次）时：**19/400（4.8%）的检索被后台整理任务堵住**，最长 **1008.8ms**
  （检索 P99 765ms），同期串行拼装完全不受影响。原因是单 worker + `SerialMemoryAccess` 单 SDK
  线程：窗口轮转触发的整理任务与前台检索抢同一线程；在 0.2s 截止下这些轮次会降级成
  `unavailable`（≈5% 的 L3 召回丢失）。

修复方向（按性价比）：

1. **整理延后到空闲**：窗口轮转后不立刻抽取，等最近一轮结束 ≥2s 再跑；
2. **整理独立进程/线程**：Qdrant 已是 Server，可让独立进程持有自己的 SDK 客户端，彻底不与读竞争；
3. `SerialMemoryAccess` 前台优先队列：读请求排在已入队的后台任务之前（不能抢占正在执行的任务）；
4. 保留 0.2s 截止兜底，但要监控降级率（现状 ≈5%，仅出现在轮转窗口附近）。

**已实现第 1 条并复测**：新增 `BUMI_CONSOLIDATION_IDLE_SECONDS`（默认 `0` 保持旧行为，建议 `2`）
与 `BUMI_CONSOLIDATION_MAX_DEFER_SECONDS`（默认 `60`，防止长会话把整理饿死）。有前台读写时
worker 不认领后台任务；空闲满窗口、或任务年龄超过最大延迟后照常执行。实现见
`src/bumi_mem/worker.py`（`note_foreground` / `_should_start_now`）与
`SQLiteSessionStore.oldest_due_job_age_seconds()`，策略有 4 个确定性单测
（`tests/test_worker_idle_policy.py`）。

同一条件（5000 条，50 轮 × 8 查询 = 400 轮次，GPU + Server + 实体加成 off）复测：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 检索 P99 / max | 765.3 / 1008.8ms | **38.5 / 45.7ms** |
| 检索被阻塞（>200ms）的轮次 | 19/400（4.8%） | **0/400** |
| 串行 P99 / max | 23.0 / 39.4ms | **14.6 / 15.3ms** |
| `recall_status` | 400 ok | 400 ok |

原始数据：`latency-eboff5000-assembly50.json`（修复前）与
`latency-eboff5000-assembly50-idle.json`（修复后）。

### 8.4 串行拼装：50 轮复测

`--assembly --assembly-repeats 50`（5000 条，n=400）：

| 指标 | P50 | P90 | P99 | max |
|---|---:|---:|---:|---:|
| 串行关键路径（总时长−检索） | 12.4ms | **13.3ms** | 23.0ms | 39.4ms |

P90 达标；P99 23.0ms 略超 20ms 预算，来自 SQLite/GC 抖动，验收建议同时看 P99 并决定是否收紧。

### 8.5 并发：8 路并发下 0.2s 截止降级 40%

`--concurrent 8 --concurrent-rounds 10`（5000 条，GPU + Server，实体加成 off，压检索队列）：

| 指标 | 值 |
|---|---|
| 单次检索 P50 / P90 / P99 / max | 175.6 / 206.5 / 208.0 / 208.2ms |
| 超过 0.2s 截止 | **32/80** |
| `MemoryReadUnavailable`（降级） | 31 |
| 每轮墙钟（8 个并发） | ≈207ms |

单 SDK 线程下 8 个检索排队 = 8×26ms ≈ 208ms，正好越过 200ms 截止。同一 agent 的轮次协议本来就要求
上层顺序发送（`docs/agent_api/README.md` §5），所以真实并发来自“一个进程服务多个 agent”。
建议按 agent 限定进程/队列深度，或对并发检索放宽截止；不要让一个进程同时服务大量 agent。

### 8.6 固化配置清单与验收顺序

```dotenv
MEM0_EMBED_DEVICE="cuda"
BUMI_QDRANT_URL="http://127.0.0.1:6333"
BUMI_ENTITY_BOOST="off"         # 验收通过配置（§9）；代码默认值仍是 on
BUMI_CONSOLIDATION_IDLE_SECONDS="2"
BUMI_CONSOLIDATION_MAX_DEFER_SECONDS="60"
BUMI_PYTHON="/data1/zhangran/.venvs/bumi-cuda/bin/python"
BUMI_MEMORY_READ_TIMEOUT="0.2"
BUMI_PORT="8099"                # 8088 可能被同机其他用户的服务占用
```

注意 `BUMI_ENTITY_BOOST`：三份语料（中文合成消融、LoCoMo 英文、LoCoMo 中文子集）都显示它
没有召回收益、只有时延；§9 的召回验收记录显示 `off` 的 Recall/MRR 反而略好，且**只有 `off`
能让检索 P90 进 60ms**。代码默认值保持 `on`（不改产品既有行为），部署配置按验收结论设 `off`。

Qdrant Server 启动方式见 §7 第 5 条；正式环境建议用 systemd 托管并固定 storage 路径。
验收顺序：

```text
preflight_check.py（环境）
  → 冷启动预热（ready 之后再开始计时）
  → latency_bench --corpus N --repeats 20（检索 P90/P99）
  → latency_bench --assembly --assembly-repeats 50（串行 P90/P99）
  → latency_bench --concurrent 8（并发降级率）
  → recall_ablation.py --dataset ~/mem0-runtime/datasets/recall_locomo_zh.json \
       --collection recall_locomo_zh --repeats 3（召回回归，结论见 §9）
```

### 8.7 服务与 Qdrant 的启停（已验证）

```bash
# 1) 首次生成 service.env（仓库外，权限 600；只从已有 secrets 读 DeepSeek Key，不打印）
scripts/make_service_env.py --python $HOME/.venvs/bumi-cuda/bin/python \
  --qdrant-url http://127.0.0.1:6333 --entity-boost off

# 2) Qdrant Server：启停 / 状态 / 日志 / 轮转（copytruncate，运行中可轮转）
scripts/qdrant_server.sh start | stop | status | logs 50 | rotate
# 常驻环境建议加 cron：0 3 * * * QDRANT_LOG_MAX_BYTES=52428800 <repo>/scripts/qdrant_server.sh rotate

# 3) 记忆服务
export BUMI_SERVICE_ENV_FILE=$HOME/mem0-runtime/memory-service/service.env
bash scripts/run_memory_service.sh     # BUMI_PYTHON 未设置时会自动探测 venv
curl -sS http://127.0.0.1:8088/health/ready
```

本次实测：`{"status":"ready","warmed_up":true}`；随后用带 Bearer 的真实请求连打 5 轮
`operation=prepare` + `operation=complete`，**端到端（含 HTTP/JSON/拼装）45–58ms，`recall_status=ok`**。

**端口分配与归属（本机约定，2026-09-23 确认）**：`8088` 是记忆服务的默认端口，但**端口是主机级
资源、不按用户隔离**——这台机器上 `0.0.0.0:8088` 已经是 **sunzhixing 的实例**
（`/data1/sunzhixing/Bumi_Mem/.venv/bin/python -m bumi_mem.api`）。
**决定：8088 归他，我们的验收实例固定用 8099**（`BUMI_HOST=127.0.0.1`）。

两个后果与做法：

1. 用 `BUMI_PORT` 换一个端口（本次自己用 `8099`）；
2. 启动后用**自己的 `BUMI_SERVICE_API_KEY`** 发一次 `operation=prepare` 验证归属
   （别人的服务不会接受你的密钥，且返回的 `agent_id` 应对得上自己的配置）；
3. 不要 kill 不属于自己的进程；`pgrep -f bumi_mem.api` 会同时列出其他用户的服务。
4. `scripts/make_service_env.py` 现在会在目标端口已被占用时给出警告；
   `configs/memory_service.env.example` 里也写明了本机约定。

本次验收实例：`127.0.0.1:8099`，`BUMI_ENTITY_BOOST=off`，长句 `prepare` 44–71ms（中位约 50ms），
`recall_status=ok`；同一问句在 `on` 下稳态约 72ms。

---

## 9. 召回验收记录（2026-09-23）

**语料**（按项目决定，本阶段以 LoCoMo 中文子集作为"真实用户语料"的代理）：
`~/mem0-runtime/datasets/recall_locomo_zh.json` —— 来源为公开数据集 LoCoMo（**真人两人对话 +
人工标注 evidence**），取 1 段对话后**机器翻译**为中文；419 条记忆 / 80 条查询
（cat1–5 = 18/17/11/17/17，其中 24 条多证据）。生成方式、翻译缓存与语料分级见 §8.2。

**环境**：H20 + `~/.venvs/bumi-cuda`（torch 2.8.0+cu128，`CUDA_VISIBLE_DEVICES=4`）+
Qdrant Server 1.19.1；`top_k=5`、`threshold=0.1`、每条查询重复 3 次（每档 240 个样本）。

**判定标准**（§8.2 的三条）：① Recall@5 与 MRR 不低于关闭实体加成前（±0.02 噪声内）；
② 检索 P90 ≤60ms、P99 ≤100ms。

| 配置 | Recall@5 | MRR | P50 | P90 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|
| 语义 only（无 BM25、无实体加成） | 0.487 | 0.334 | 32.4 | 33.7 | 37.6 | 357.9 |
| **实体加成 off**（语义 + BM25） | **0.487** | **0.334** | 34.0 | **35.6** | **37.3** | 38.6 |
| **实体加成 on**（完整） | 0.475 | 0.300 | 34.0 | **60.7** | 65.3 | 67.1 |

**结论**：

1. 标准① **通过**：关闭实体加成后 Recall **+0.013**、MRR **+0.034**（更好而非更差），
   远在 ±0.02 之内；
2. 标准② **只有关闭实体加成才通过**：off 时 P90 35.6ms / P99 37.3ms；on 时 P90 60.7ms
   已经越过 60ms；
3. 因此本语料下的验收通过配置是 **`GPU + Qdrant Server + BUMI_ENTITY_BOOST=off`**；
   代码默认值保持 `on` 不变，部署配置按此切换（改一行 + 重启，见 §8.6）；
4. 端到端参考：同配置下 HTTP `operation=prepare` 为 45–59ms（§8.7）。

原始数据：`mem0-runtime/logs/recall-locomo-zh.json`（含每档 240 个样本与分组召回）；
运行命令见 §8.2。

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
