# Bumi 阶段1实现与验收报告

日期：2026-09-14。**阶段1约定范围已实现并通过人工数据验收；没有进入阶段2。**

## 1. 已实现

新增独立的 `bumi_mem.typed` 包，使用实际SQLite数据库，而非仅定义schema或返回模拟结果：

- 通用space/principal/membership/relation，以及独立conversation/参与者/每轮听众。
- Raw记录、最终文本/存储许可、修订、原始时间、Unicode来源片段与引用核验。
- identity/behavior/episode/record_index四种语义对象、类型payload、版本与声明的证据关系。
- current/historical查询、真实时间变化与纠错分离；撤回某旧来源时不误删独立的新版本依据。
- 字段、主体、接收者、用途、有效期上的allow/deny；需要来源所有者与涉及主体的许可。
- 手工私有/共享视图、最小字段发布、计划不确定性与assertion_kind保留。
- Raw/L1/carry-over保护、上下文/版本票据、撤回与删除代次、重读时重新授权。
- 来源依赖传播、旧视图失效、幂等撤回、expected_version竞争更新与事务回滚。

没有自动抽取、自动共享类别判断、行为模式学习、向量检索或生成聊天回答。

## 2. 实际验证

| 项目 | 结果 |
|---|---:|
| 实施前原有测试 | 95 passed |
| 新增阶段1功能测试 | 68 passed |
| 最终合并回归 | **163 passed，19.51秒** |
| 原阶段0基线入口兼容性 | 仍只执行原有71项基线测试，71 passed |
| 独立进程撤回验证 | 撤回前新进程可读；撤回后另一新进程不可读 |
| 手工Demo | exit 0 |
| 本阶段模型/API调用 | 0 |

新增测试访问真正的SQLite和策略代码，测试过程中阻止网络连接。测试覆盖允许/拒绝、来源伪造、助手内容经派生记录转成用户事实的阻断、历史版本、纠错、缓存票据、访客、成员退出、Raw/carry-over、删除后重开、竞争更新、回滚和幂等重试。

这是固定测试集的结果，不是任意自然语言隐私推断都零泄露的证明，也不是完整Memory Bundle的60ms验收。

## 3. 你可以运行的Demo

在服务器执行，无需DeepSeek Key：

```bash
bash /home/legendre/data/ZhixingSun/Bumi_Mem/scripts/run_stage1_demo.sh
```

每次新建隔离合成数据库，不覆盖已有文件。实际输出证明：

| 场景 | 结果 |
|---|---|
| 父亲查看自己的私有视图 | 可见合成餐厅等私有内容 |
| 未获全部相关主体授权就发布 | 拒绝 |
| 父母分别授权后孩子读取 | 仅有获准的忙闲、时间、主体字段 |
| 孩子读取父亲私有view或Raw | 不可读 |
| 同时有访客接收回答 | 不可读这份仅对孩子授权的view |
| 撤回后使用旧读取票据 | StaleRead |
| 重新打开数据库读取旧共享view | 不可读 |

孩子获准的数据示例：

```json
{
  "availability": "possibly_busy",
  "subject_ids": [
    "dad",
    "mom"
  ],
  "time": {
    "day_part": "evening",
    "end_at": null,
    "local_date": "2026-09-14",
    "precision": "day_part",
    "start_at": null
  }
}
```

`possibly_busy`保留“这是计划，尚未确认实际发生”的含义，不直接变成“没人陪孩子”。Demo数据与权限都由代码人工构造，不是模型自动识别。Demo报告包含开发者对照用的父亲私有合成信息，不能整份送入孩子的聊天上下文。

## 4. 改动范围

正式仓库：`/home/legendre/data/ZhixingSun/Bumi_Mem`。

新增实现：

- `/home/legendre/data/ZhixingSun/Bumi_Mem/src/bumi_mem/typed/models.py`
- `/home/legendre/data/ZhixingSun/Bumi_Mem/src/bumi_mem/typed/store.py`
- `/home/legendre/data/ZhixingSun/Bumi_Mem/src/bumi_mem/typed/policy.py`
- `/home/legendre/data/ZhixingSun/Bumi_Mem/src/bumi_mem/typed/schema.py`
- `/home/legendre/data/ZhixingSun/Bumi_Mem/src/bumi_mem/typed/errors.py` 与包入口。

测试/工具/说明：

- `/home/legendre/data/ZhixingSun/Bumi_Mem/tests/stage1/`
- `/home/legendre/data/ZhixingSun/Bumi_Mem/tools/stage1/demo.py`
- `/home/legendre/data/ZhixingSun/Bumi_Mem/scripts/run_stage1_demo.sh`
- `/home/legendre/data/ZhixingSun/Bumi_Mem/docs/stage1/README.md`

现有文件仅有两项有意改动：

1. `pyproject.toml`：声明直接依赖 `pydantic>=2,<3`，没有安装或升级当前Conda依赖。
2. `tools/stage0/stage0.py`：把旧基线测试选择固定为原有五个测试文件，避免阶段1新增测试被误计入“旧基线”。

其他52个既有已跟踪/未跟踪文件保持哈希不变。没有改旧聊天编排、Mem0适配器或submodule，没有迁移旧数据库，没有修GPU/驱动，没有Git提交或推送。

## 5. 重要使用边界

- **旧聊天CLI尚未接入新库。** 新模块是显式调用的阶段1底座，不能说旧CLI已经具备这些权限能力。
- Scope身份必须由可信上游确认；此模块不提供声纹、人脸、登录或HTTP认证，不能直接相信终端用户提交的verified字段。
- 手工字段投影与声明的来源属性依赖已经检查；自然语言究竟涉及谁、是否支持某个结论，由后续提取/核验负责。
- 来源集合/类型不允许在同一memory ID上偷偷换主体；后续要扩展必须显式设计受控迁移。
- 当前实现逻辑不可读、依赖失效与读时期限检查；**不是已经完成物理擦除、备份清除或第三方API删除**。
- Raw与语义对象的14天上界只用于合成测试；成员/授权/审计仍需命名空间清理，尚无后台物理清理器。
- 政策读取在SQLite事务快照上做决定；上层在生成/下发前必须重新验证票据。已发送内容无法靠撤回取回，完整在途回答处理留到集成阶段。

## 6. 报告与后续

原始验证目录：`/home/legendre/data/ZhixingSun/mem0-runtime/stage1/runs/run-20260914T120720-w7vYSW/reports`。

Demo目录：`/home/legendre/data/ZhixingSun/mem0-runtime/stage1/runs/manual-demo-20260914T144356-dqWAyK`。

当前可以进入阶段2：把最终文本接入增量候选抽取与结构化写入。阶段1的权限/来源/版本接口应继续作为门槛，而不是让模型直接写生效记忆或授予共享权限。本轮未自动开始阶段2。
