# 阶段0接口合同与计时合同

版本：stage0-2026-09-10-v1。实现范围仅为严格数据外壳、fixture校验和探针，不是完整业务模型。

## 数据合同

| 合同 | 已固定的关键项 | 阶段0不宣称解决 |
|---|---|---|
| Scope | space、actor、audience、用途、policy/deletion epoch；身份未知不能冒充某人 | 认证、授权匹配与听众感知 |
| Observation | ID/revision、conversation、角色、文本、带时区的原始时间、final、合成配置 | ASR、真实原始数据许可 |
| SourceSpan | 来源ID/revision、Unicode code point左闭右开范围 | 模型来源是否语义真实，须后续对照原文 |
| Memory | identity/behavior/episode/record_index、快照version、状态、断言类型、涉及者、evidence | kind内部payload仍需阶段1进一步细化 |
| MemoryView | view及其revision、源版本、current/historical、投影、显示文本、允许听众/用途 | 字段级共享算法或间接泄露证明 |
| QueryView | 原句、检索表示、人物、时间模式、as_of、purpose | 自然语言解析和检索规划 |
| MemoryBundle | 分区视图、证据、不确定性、真实检索/持久状态、可选计时 | 当前不存在完整prepare_turn链路，不能报60ms已通过 |
| SeedCase | 合成初始观察、当前查询、setup、未来must/must-not与目标阶段 | fixture验证不是回答/权限行为验证 |

`Memory.version`是这份传输快照的版本；数据库根对象的`current_version`由阶段1实现。`view_revision`和`source_memory_version`不是同一个数字。`verified_by_upstream`只表示受信上游传入的身份状态，不是阶段0具备身份识别。

公共外壳采用strict类型、extra=forbid。模型不能通过添加`read_all_private_memories`获得权限。即使结构合法，display_text仍可能包含不该公开的内容，必须由后续政策/投影模块检查。

合同JSON Schema由`run_stage0.sh validate`导出到隔离报告目录；源定义在`tools.stage0.contracts`，避免维护两份不一致的手写schema。

## 时钟与观测字段

以下字段为统一挂接位置；未测量写null/不报告，不填0冒充：

- ingest_ms：服务收到最终文本后的必要接入。
- query_compose_ms：QueryView形成。
- embedding_ms：查询分词、设备传输、模型/池化及结果返回。
- sql_ms / fts_ms / vector_ms：各自查询与等待。
- merge_ms / bundle_pack_ms：去重/版本与权限复核/打包。
- memory_bundle_ms：从服务收到最终文本到Bundle交付的单调时钟差，**不是各阶段P95相加**。
- extraction_visibility_ms / index_lag_ms：新语义可见时间与索引滞后，和读取耗时分开。
- retrieval_status：完整、partial、degraded、unknown等。

阶段0探针已实际采集编码、FIFO排队和独立SQL/FTS/本地向量查询；接入/合并/完整Bundle等待阶段3–4实现后挂接。样本量、设备、dtype、模型/库版本、输入长度、cold定义必须与指标同时保存。

不得把Qdrant local in-memory微探针当成Qdrant Server基准；不得把CPU与GPU不同软件环境的差值解释为纯GPU加速；不得把合同负例测试当成已实现私密共享。

## 后续服务边界

阶段1起稳定实现Repository、PolicyEngine、ViewProjector、SourceRegistry；阶段2实现CandidateExtractor与增量任务；阶段3实现最小C/P及Bundle；阶段4接完整Index/Embedding/QueryComposer；阶段5实现行为模式确认生命周期。

当前工具不替换`bumi_mem.engine`、`long_term`、`session_store`或`providers`，旧模式继续作为隔离基线。
