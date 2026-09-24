# 原始结果索引（`docs/latency/raw/`）

这里是 `REPORT.md` 引用的原始 JSON——就是压测脚本 `--json` 的落盘产物，从 `~/mem0-runtime/logs/`
按文件名原样复制，内容未做任何修改。每份文件都自带：

- `environment`：宿主、torch/CUDA 证据（`torch_cuda_available`、`torch_cuda_devices`）、
  `requested_embed_device` 与**实际生效**的 `resolved_embed_device`、向量后端（`qdrant-local-path`
  还是 `qdrant-server(http://127.0.0.1:6333)`）、集合名、warmup 完成标志；
- `config`：语料条数、查询数、重复次数、`top_k`/`threshold`；
- 逐样本时延数组：§3、§8 的每一张表都能用这些数组直接复算，不需要重跑。
- 召回类文件另有 `results`（每档的逐查询命中）与 `bm25_hits`。

`指纹`列是复制时算的 sha256 前 12 位，用来确认仓库副本与当时落盘一致。
仓库外仍保留完整集合（含 `.txt` 人类可读版、探索性运行与 `qdrant-server.log`）：
`~/mem0-runtime/logs/`。

## 环境预检（§8.1）

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `preflight-gpu-server.json` | 11 项检查全部 PASS（GPU/CUDA、实际编码设备、Qdrant Server、集合归属） | `25667304b6ef` | 4 KB |

## CPU 基线（§3.1–3.6，`~/.venvs/bumi`，torch 2.8.0+cpu，Qdrant local path）

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `latency-final-300.json` | 300 条，8 查询 × 20 次 + 引擎前台 | `630f51db185b` | 44 KB |
| `latency-final-5000.json` | 5000 条，同上 | `dac56c368cca` | 44 KB |
| `latency-final-300-assembly.json` | 300 条，串行拼装分解（5s 等待上限） | `d4d82d16113f` | 4 KB |
| `latency-final-5000-assembly.json` | 5000 条，串行拼装分解 | `252dfdf0b360` | 4 KB |
| `latency-qdrant-scale.json` | Qdrant local client 规模伸缩探针（随机向量） | `ddf29751a8df` | 4 KB |

## GPU（H20）+ Qdrant local path（§3.7，`~/.venvs/bumi-cuda`，`CUDA_VISIBLE_DEVICES=4`）

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `latency-gpu-300.json` | 300 条，8 查询 × 20 次 + 引擎前台 | `3bb977d675ea` | 44 KB |
| `latency-gpu-5000.json` | 5000 条，同上 | `46d881169566` | 44 KB |
| `latency-gpu-300-assembly.json` | 300 条，串行拼装分解 | `6ca58e4140ca` | 8 KB |
| `latency-gpu-5000-assembly.json` | 5000 条，串行拼装分解 | `7376bd121b53` | 8 KB |
| `latency-qdrant-scale-gpu.json` | 规模伸缩探针（不经过编码模型） | `b7d84e01a934` | 8 KB |

## GPU + Qdrant Server（§3.8，`BUMI_QDRANT_URL=http://127.0.0.1:6333`）

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `latency-srv-300.json` | 300 条，实体加成默认 on | `f5835f680568` | 44 KB |
| `latency-srv-5000.json` | 5000 条，实体加成默认 on | `41bba2b35fff` | 44 KB |
| `latency-srv-20000.json` | 20000 条，实体加成默认 on | `be5e4c1aa742` | 44 KB |
| `latency-srv-5000-assembly.json` | 5000 条，串行拼装分解 | `b4271cec7432` | 8 KB |

## 实体加成移出热路径（§3.9、§8.3 尾延迟）

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `latency-eboff-300.json` | 300 条，`BUMI_ENTITY_BOOST=off`，8 查询 × 20 次 | `ee31bed10253` | 44 KB |
| `latency-eboff5000.json` | 5000 条，同上 | `63de8ccdceea` | 44 KB |
| `latency-eboff20000.json` | 20000 条，同上 | `3e799c5e32f8` | 44 KB |
| `latency-eboff20000-tail.json` | 20000 条 × 100 次/查询（800 样本）的稳态尾延迟 P99/max | `23cadd83b292` | 188 KB |

## 流式前缀预取（§3.10）

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `latency-prefetch-sim.json` | 5000 条，不预取 prepare / 预取 / 预取后 prepare 三段对照，50ms 提前量 | `761e96f3e63d` | 44 KB |

## 后台整理让路（§8.3）

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `latency-eboff5000-assembly50.json` | 50 轮 × 8 查询，修复前（19/400 轮被整理阻塞） | `4f01e147d0a5` | 128 KB |
| `latency-eboff5000-assembly50-idle.json` | 同条件，`BUMI_CONSOLIDATION_IDLE_SECONDS=2` 修复后（0/400） | `393c9abe4e1a` | 128 KB |
| `latency-gap-idle2.json` | 轮间静默 2.0–5.0s，`IDLE=2`：0/48 阻塞 | `0cc1f242cf40` | 16 KB |
| `latency-gap-idle0.json` | 同条件，`IDLE=0`（旧行为）对照：0/48 阻塞 | `10a6540e7df1` | 16 KB |
| `latency-gap-sdk.json` | 整理任务占用单 SDK 线程的时长（P50 716 / P90 945 / max 1073ms） | `786f28972667` | 20 KB |

## 并发（§8.5）

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `latency-eboff5000-concurrent.json` | 8 路并发 × 10 轮，修复前（32/80 超 0.2s 截止） | `27f941017a6b` | 48 KB |
| `latency-eboff5000-concurrent-postfix.json` | 同条件复测（45/80，随调度抖动） | `e14786f821d7` | 48 KB |

## 召回（§8.2、§9）

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `recall-locomo.json` | LoCoMo 全量英文：5882 条记忆 / 300 条 probe，三档对照 | `ab309ea2cefe` | 44 KB |
| `recall-locomo-zh.json` | **验收依据**：LoCoMo 中文子集 419 条记忆 / 80 条查询 × 3 次，语义 only / 实体加成 off / on 三档（§9 的表） | `84fa25a645a8` | 28 KB |
| `recall-demo.json` | 仓库内最小示例集流程自检 | `0a3914da080f` | 8 KB |
| `recall-sim-zh.json` | LLM 生成中文模拟集（七类覆盖）三档对照 | `adeaf99b1d1a` | 8 KB |
| `retrieval-ablation-20260922-184417.json` | 317 条记忆 / 13 条查询的检索路消融（§3.9 质量证据） | `353910e1e268` | 8 KB |
| `retrieval-ablation-gpu-2000.json` | 2000 条干扰记忆 + 13 条带标准答案查询 × 10 次 | `43a58055e3aa` | 16 KB |

数据集本身（`memories` + `probes`，与上面的结果文件配套）：

| 文件 | 内容 | 指纹 | 大小 |
|---|---|---|---|
| `tools/latency/datasets/recall_locomo_zh.json` | §9 验收用的中文子集：419 条记忆 / 80 条查询（cat1–5 = 18/17/11/17/17，24 条多证据） | `14b3f3877a0d` | 116 KB |
| `tools/latency/datasets/recall_sim_zh.json` | LLM 生成的中文模拟集（七类覆盖） | 见文件内 `_comment` | 4 KB |
| `tools/latency/datasets/recall_demo.json` | 手写最小示例 | — | 2 KB |

未提交（第三方整库语料，避免在仓库里整库分发）：LoCoMo 官方 `locomo10.json`
（2.8MB，sha256 `79fa87e90f04`）、其派生的全量英文数据集 `recall_locomo.json`（1.1MB，sha256 `536e9d302b5f`）、
以及机翻缓存 `locomo_zh_cache.json`（169KB，sha256 `51dc375c3e34`）。三者都能按 §7 的命令重建：
`locomo10.json` 从 `snap-research/locomo` 下载，后两者由
`tools/latency/locomo_to_recall_dataset.py` 生成（缓存命中后不再调用翻译 API，结果逐字节可复现）。
