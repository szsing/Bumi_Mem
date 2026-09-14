# 阶段1：可控数据、版本、权限与手工视图

实现位置：`/home/legendre/data/ZhixingSun/Bumi_Mem/src/bumi_mem/typed`。

## 当前能力与明确边界

这是可运行的SQLite基础库，不是只有DTO/schema。支持人工构造的原始记录、来源片段、四种语义对象、版本、授权和读取视图；不调用API/模型，不解析自然语言，不自动生成跨成员推断。

**旧聊天CLI没有接入这套库。** 旧Bumi/Mem0流程继续保留；不能因此宣称旧CLI已经获得typed权限保护或双通道能力。新模块需显式导入`bumi_mem.typed`，并使用独立数据库。

## 运行人工验收Demo

```bash
bash /home/legendre/data/ZhixingSun/Bumi_Mem/scripts/run_stage1_demo.sh
```

无需API Key。每次创建新的合成测试库，不覆盖原有数据库。输出包含：父亲自己的私有视图、孩子获准的最小忙闲视图、孩子无法读取的私人地点/原文、访客场景拒绝、撤回后旧票据失效和重启后仍不可读。

Demo报告含父亲私有合成数据作对照，只用于开发者验收，不能整份作为孩子的聊天上下文。

时间安排为人工构造，不是LLM从文本提取；`availability=possibly_busy`明确保留计划的不确定性，不把计划当作已完成的行为。

## 核心接口

| 模块/接口 | 已实现 |
|---|---|
| provisioning | `create_principal/create_space`：只供可信初始化代码，不暴露为聊天工具 |
| membership/relation | 空间成员、显式关系；管理员不能替另一主体授予共享许可；关系不自动授权 |
| conversation | 创建会话、显式邀请参与者；同空间成员不能猜到ID就向私聊插话 |
| observation/span | final、存储许可、带时区来源时间、Unicode范围+原句匹配、修订、幂等 |
| memory/version | identity/behavior/episode/record_index；类型payload校验、当前/历史、生效区间与来源链 |
| grants | 来源所有者与涉及主体都需许可；按接收者、类别、字段、用途、时效执行；deny优先 |
| manual view | 人工选择现有字段；由固定规则生成投影，不接受任意display_text；推断对象先保留candidate |
| read API | `read_view/list_views/read_turns/read_observation`统一检查范围、听众、状态、期限与deny |
| read ticket | 包含上下文指纹与content/policy/deletion代次；`materialize`重新检查数据库，不把旧payload当授权 |
| revocation | `revoke_grant/delete_memory/withdraw_observation`与来源依赖传播、旧视图失效 |

Pydantic边界开启strict、extra=forbid、实例重新校验，避免`model_copy/model_construct`跳过校验后直接写入。输出保留assertion_kind，不把转述或候选伪装成确定事实。

## 权限与信任边界

1. Scope.actor_id和identity_status必须由可信上游确认。此库不提供声纹、人脸、登录或HTTP认证；不能把用户自己提交的JSON当作已认证身份。
2. 请求者始终计入实际接收者，即使调用方从audience列表中省略它；未知听众/人物不读取个人长期内容。
3. 共享许可由每个主体对自己设置，不能由父亲替母亲授权。字段需要满足来源与涉及主体的许可，不是所有人allow做并集。
4. 私人view只给所有者独处上下文；共享view不附带私人原文或来源正文。
5. 带多个主体的源不能被派生记忆悄悄删掉主体；派生关系声明使用了哪些源字段，读取时仍检查这些源字段的权限。
6. `block_inference`采用保守的事件/来源阻断，防止把被禁止的字段改名后间接公开。它不是任意自然语言推断的零泄露证明，依赖可信写入方如实声明主体和属性依赖。
7. Raw/L1/carry-over也检查原来的听众与后续禁止/撤回。受影响的整条Raw会保守隐藏，阶段1不做自动文本局部脱敏。

## 时间、修订与删除

- 新发言使用新observation ID；同一ASR/记录修订递增revision。旧revision不能重放为当前来源。
- `temporal_update`表示过去曾成立，关闭旧有效区间；`correction`标记旧值有误，不能以historical方式复活。
- 修改后旧视图失效，**需要人工重建视图**；自动再投影留到阶段2–3。历史版本可人工创建historical view。
- 时间查询使用as_of选择业务有效区间，但授权与保留仍使用当前服务端时钟，不能倒回权限。
- 来源撤回按版本依赖传播；有独立新来源的当前版本不因旧历史来源撤回而一起丢失。
- 删除语义对象与撤回原始观察是不同操作；重复删除/撤回幂等，不能因为旧撤回请求重试而让新明确授权失效。应用读取会阻止已删/被拒绝内容通过关联Raw旁路回来；数据行物理清理、备份与索引全链路擦除留待后续阶段，不能宣传为已经完成不可恢复删除。
- 当前只接受合成测试配置，读取时对Raw和语义对象执行14天保留上界；成员注册/策略/审计仍需命名空间清理，不能当作已经自动删除；**没有后台物理清理器**。不把这套测试保留设置当作真实用户政策。

## 数据库保护

新库使用独立application_id与user_version=1001。旧Bumi只支持到2，因此旧库与新库相互拒绝误打开；不把旧业务库自动迁移。新文件权限600，WAL、FULL同步、外键、短事务、乐观expected_version。

Schema创建、语义版本和来源边、grant与epoch更新均在事务内。原文、API Key和私密payload不写到audit_events；审计只记录操作与对象ID。

## 阶段1暂不包含

- 自动抽取、自动事件合并、自动共享类别判断、自动Behavior归纳/生效。
- Dense/BM25/Qdrant检索、API回答生成、完整MemoryBundle及60ms性能验收。
- 自动物理擦除、分布式缓存、多机并发、生产认证、语音识别。
- 修改同一memory ID的kind/owner/主体集合；首版禁止静默换主体，未来必须通过受控迁移扩展。
- 公共的“绕过权限查看所有历史/删除内容”调试入口。

## 测试含义

阶段1测试调用实际存储与权限代码，覆盖允许/拒绝、版本、回滚、竞争更新、来源撤回、重启、缓存与carry-over。测试期间禁止网络连接。它们不证明未实现的自然语言理解或全部现实隐私风险已解决。

最终服务器测试结果和人工Demo证据见本目录执行报告；阶段0文档与已有基线均保留。
