# 阶段0：基线、隔离环境、合同与探针

日期：2026-09-10。正式仓库：`/home/legendre/data/ZhixingSun/Bumi_Mem`。

此目录不是阶段1实现。新增内容限于准备脚本、结构合同、合成用例与测试；`src/bumi_mem`、上游submodule、现有数据库与Conda环境保持不修改。

## 默认运行：不需要API Key

```bash
cd /home/legendre/data/ZhixingSun/Bumi_Mem
bash scripts/run_stage0.sh all
```

每次默认创建新的隔离run目录，不覆盖原测量。输出放在：

```text
/home/legendre/data/ZhixingSun/mem0-runtime/stage0/runs/<run-id>/
  reports/   环境、测试、合同导出、微基准
  legacy/    旧基线的临时数据
  typed/     预留的新模式隔离数据，不是已经实现的新引擎
  home/      测试HOME
  cache/     测试缓存
```

`all`执行doctor、12例fixture结构校验、原有测试、stage0合同测试、独立存储微探针。不会读取密钥文件、调用DeepSeek或打开现有记忆库。现有测试中的网络行为使用mock；它们不证明真实模型效果。

需要在同一次run中执行单项时，可设置`STAGE0_RUN_DIR`为上述runs目录下的一个新子目录；每个报告名只写一次，重测请创建新run。

## Embedding组件探针

```bash
# 使用既有mem0环境；当前该环境的PyTorch是CPU版
bash scripts/run_stage0.sh probe-cpu

# 使用既有vllm环境的CUDA版PyTorch，不升级/修改mem0环境
bash scripts/run_stage0.sh probe-cuda
```

探针只加载本地Qwen3-Embedding-0.6B，不下载模型，不使用私密对话。模型路径：`/home/legendre/data/ZhixingSun/models/Qwen3-Embedding-0.6B`。

测量包含：当前进程首次模型载入、32/128/256个总编码tokens（包含instruction）、预热后batch1、查询排在已开始的batch4索引工作之后。模型只有一个worker执行，不测试未来优先级调度器的实际实现。CPU和GPU来自不同既有环境，dtype/版本不同，不能把差值只归因于硬件。

**不是以下测量**：完整Memory Bundle的P95、真实语义召回质量、计划中的Qdrant Server/HNSW性能或DeepSeek生成耗时。操作系统磁盘缓存没有被清空。小样本P95只是阶段0组件观察，不是最终SLA证明。

## DeepSeek：仅在用户准备好Key后手动启动

```bash
cd /home/legendre/data/ZhixingSun/Bumi_Mem
bash scripts/run_stage0_deepseek_smoke.sh
```

脚本在终端无回显读取Key；不source原有.env，不把Key写到文件或命令参数。只发一个公开测试字符串，最多64个输出tokens，无自动重试，明确关闭thinking；有API费用，只有手动执行此命令才发送。

当前准备命令明确指定`deepseek-flash`。仓库旧Provider仍默认`deepseek-chat`，阶段0没有改它。不要将“固定别名字符串”当作服务端模型权重永久固定：报告记录requested_model和reported_model。默认思考模式等行为也应在真实集成时显式核验。

该脚本只验证API端点和认证，不运行Bumi记忆抽取、共享权限或完整聊天流程。不应将API smoke成功说成记忆系统验证通过。Key不要发到聊天中。

官方设置依据（2026-09-10查阅，未进行真实API调用）：

```text
https://api-docs.deepseek.com/
https://api-docs.deepseek.com/guides/thinking_mode
https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
```

## 准备材料

- `tools.stage0.contracts`：Pydantic严格结构合同，不包含权限引擎或检索实现。
- 合成fixtures：12例，包括本人事实、当前覆盖、共享忙闲、字段保密、禁止间接泄露、身份未知、计划取消、时间不明、候选模式、历史时间、撤回/迟到任务、访客。
- `expected`是后续阶段的验收要求，不是阶段0已测得的输出。
- `configs/stage0/synthetic.json`只用于隔离配置和未来约束。14天测试保留是设计要求，阶段0尚未实现自动到期清理；测试后应按已核验目录显式清理，不得称已有自动删除能力。

## 运行结果与阶段门

实际结果放在run目录，并汇总到本目录的执行报告。阶段0区分：

- offline gate：环境、基线、结构合同、合成输入和组件风险可检查。
- live API integration：没有用户Key就不运行，不能伪造成功。
- future behavior：共享推理、全链路删除与60ms Bundle均尚未实现/测量。

停止任何后续实现前，保留此基线和测量记录；不要自动进入阶段1，不提交或推送Git变更，除非用户另行要求。
