# Bumi 阶段0执行报告

执行日期：2026-09-10。状态：**离线阶段0门已通过；真实DeepSeek API联调未运行；尚未开始阶段1。**

## 1. 正式工作区与改动边界

仓库：`/home/legendre/data/ZhixingSun/Bumi_Mem`。基线HEAD：`ad59a1f4f9e82d8eb103ac6aa96a34245a7a1a20`。

原有41个已跟踪文件哈希保持不变。没有修改业务源文件、两个submodule、既有数据库或Conda依赖；没有提交或推送Git。新增内容仅为阶段0文档、合成fixture、结构合同、测试与独立探针。

隔离运行目录：`/home/legendre/data/ZhixingSun/mem0-runtime/stage0/runs/run-20260910T181852-qGws8r`。legacy/typed分开，typed目前只是预留目录，不是已实现的新引擎。Key只检查当前环境是否存在，不读取或输出秘密文件内容。

## 2. 测试与准备结果

| 检查 | 结果 |
|---|---|
| 改动前旧测试 | 71 passed，7.56s |
| 添加阶段0工具后旧测试 | 71 passed，7.24s |
| 新增结构/工具测试 | 24 passed |
| 合并测试 | 95 passed，7.05s |
| 合成用例schema | 12例合法；**不是12例记忆行为已通过** |
| 完整离线启动脚本 | exit 0 |
| 无Key非交互启动保护 | exit 2；未发送请求 |
| 真实API、端到端语义效果、完整Bundle P95 | 未运行/未测量 |

## 3. 环境

CPU：Intel(R) Core(TM) Ultra 7 265K；逻辑CPU数：20；可见内存：131252520 kB。GPU：NVIDIA GeForce RTX 4090。

- 旧mem0环境：Python 3.11.16，SQLite 3.53.4，FTS5可用；torch 2.13.0+cpu，transformers 5.16.1。
- GPU探针复用既有vllm环境：torch 2.6.0，CUDA 12.4，transformers 4.51.3；没有升级主环境。
- 现有mem0环境为CPU版PyTorch，不能简单添加cuda参数就当作已支持GPU。
- Docker可用；6333/6334没有Qdrant Server监听。本阶段只核验本地client能力，没有启动计划中的服务端。

## 4. Embedding组件复测

同一套本地模型文件、同一池化与测试逻辑；两环境库版本/dtype不同，不把差异只归因于GPU硬件。每个长度复测50次，先预热8次；总tokens含查询instruction，batch=1，包含分词/传输/模型/池化/结果准备。OS缓存没有清空。

| 编码tokens | CPU P50 ms | CPU P95 ms | CUDA FP16 P50 ms | CUDA FP16 P95 ms |
|---|---:|---:|---:|---:|
| 32 | 50.45 | 53.00 | 10.16 | 10.29 |
| 128 | 112.97 | 115.55 | 9.90 | 10.01 |
| 256 | 190.63 | 215.59 | 10.19 | 10.35 |

CPU查询排在已开始的batch4×256tokens索引之后：P95 **936.20 ms**（n=10）。CUDA同类探针：P95 **27.76 ms**（n=20）。这是单FIFO worker排队场景，不是未来优先级调度器的验收。

首次当前进程模型加载：CPU 437.76ms，GPU 939.42ms；该“cold”不是清空OS缓存后的物理冷读。GPU峰值模型分配约1.15GiB。探针结束后未留运行中的GPU模型服务。

早期小样本pilot与复测文件均保留，没有删掉较慢样本。**不能用这些组件P95证明完整Memory Bundle已达到60ms，也不能把各组件P95简单相加。**

工程判断：不建议以当前CPU路径承担256token查询下的60ms目标；后续应采用常驻GPU编码、短索引批次与查询优先，同时将抽取API从在线读取资源隔离。阶段0不直接改主环境或旧SDK线程模型。

## 5. 独立存储微探针

1000条合成记录/随机归一化向量，n=50；SQLite探针为独立文件、WAL/FULL，不打开既有数据库。

| 项目 | P95 ms |
|---|---:|
| SQL查找 | 0.054 |
| FTS5 BM25 | 0.273 |
| 单行持久提交 | 3.010 |
| 预编码向量查询 | 4.591 |

向量后端是 **Qdrant local in-memory client，不是Qdrant Server/HNSW**；随机向量不提供语义质量证据。这些是接入/风险探针，完整双通道与权限过滤性能留待阶段4/6。

## 6. 实施文件

- `/home/legendre/data/ZhixingSun/Bumi_Mem/tools/stage0/contracts.py`：严格结构合同与计时字段。
- `/home/legendre/data/ZhixingSun/Bumi_Mem/tools/stage0/stage0.py`：环境、fixture、测试、存储与显式API smoke工具。
- `/home/legendre/data/ZhixingSun/Bumi_Mem/tools/stage0/embedding_probe.py`：隔离编码/排队探针。
- `/home/legendre/data/ZhixingSun/Bumi_Mem/scripts/run_stage0.sh`：默认不使用API。
- `/home/legendre/data/ZhixingSun/Bumi_Mem/scripts/run_stage0_deepseek_smoke.sh`：交互式安全输入Key。
- `/home/legendre/data/ZhixingSun/Bumi_Mem/tests/fixtures/stage0/cases.json`：12个合成验收种子。
- `/home/legendre/data/ZhixingSun/Bumi_Mem/tests/stage0/test_stage0_artifacts.py`：结构/工具测试，不是共享推理实现。
- `/home/legendre/data/ZhixingSun/Bumi_Mem/configs/stage0/synthetic.json`：隔离配置；14天自动清理尚未实现，不能声称已有。
- `/home/legendre/data/ZhixingSun/Bumi_Mem/docs/stage0/CONTRACTS.md`与README：接口、计时范围、运行与限制。

## 7. 需要真实DeepSeek时的启动指令

在服务器终端执行：

```bash
bash /home/legendre/data/ZhixingSun/Bumi_Mem/scripts/run_stage0_deepseek_smoke.sh
```

终端会无回显询问DeepSeek API Key，不要把Key发到聊天中。仅一个小型付费请求：公开测试文本、最多64输出tokens、thinking disabled、无自动重试；不存Key、不source既有.env、不携带私密聊天数据。

准备命令明确使用deepseek-flash并记录服务返回model。旧Bumi Provider仍默认deepseek-chat且没有显式thinking配置，本阶段未修改。API smoke通过也只代表端点/认证，不代表旧Provider、Mem0抽取或完整聊天已验证。

## 8. 阶段门与下一步

S0.1正式仓库、S0.2环境固定版本核验、S0.3离线旧基线、S0.4隔离、S0.5种子样例、S0.6结构合同、S0.7组件探测、S0.8计时合同/探针已准备。完整Bundle挂接不存在于旧代码，本阶段不伪造。

当前可以进入不依赖LLM的阶段1数据/权限底座；真实API联调仍待用户运行上述命令。没有自动进入阶段1。

报告原始证据位于`/home/legendre/data/ZhixingSun/mem0-runtime/stage0/runs/run-20260910T181852-qGws8r/reports`，含测试日志/XML、environment、CPU/GPU每样本耗时、存储微探针、源文件/模型指纹。模型/API运行边界与未测量项须与本报告一起保留。
