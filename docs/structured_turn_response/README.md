# 0.6.0 结构化单次响应与 schema 9 升级

日期：2026-09-24。输入六字段和每轮一次调用方式不变。HTTP API从1.3.1升级到1.4.0，主数据库从schema 8升级到schema 9。

## 新响应

顶层严格为：

```json
{
  "turn_id": "...",
  "current_user": "...",
  "long_term_memory": [{"user_id":"...","content":"...","time":[]}],
  "short_term_memory": [{"role":"user","speaker_id":"...","content":"...","time":"..."}]
}
```

不再返回`prompt`字符串。长期数据直接来自当前可见`shared_memory_refs`、`semantic_current`和明确历史查询的`semantic_versions`；短期数据来自经现有窗口算法选择的`MessageRecord`。服务不会把内部score、任务ID、数据库ID、token_count或context_only下发。

为满足最简协议，召回/写入/裁剪状态不再单独下发。Agent不得将空长期数组解释为用户从无长期记忆，也不得根据返回条目承诺已永久保存。需要运维诊断时使用服务日志和管理检查脚本，而不是扩展普通Agent响应。

## schema 9

0.5.x缓存的是`turn_id + prompt`。升级时：

- 当前未结束的旧字符串缓存被清除，原消息不重复写入；相同请求可以重新构造结构化响应；
- 已关闭旧请求仍按原生命周期返回冲突，不重新运行旧问题；
- 无法识别的损坏缓存标记为ambiguous，不猜测重放；
- 原始消息、长期向量、核心版本和来源记录不改写。

升级前暂停Agent并补存最后确认回复，停止HTTP服务，备份整个`BUMI_DATA_DIR`，再更新代码启动。回滚必须恢复schema 8备份，不能手改user_version。

## 测试边界

测试覆盖请求幂等、上一回复、用户切换、guest、长期来源时间、核心状态、历史版本、L1窗口、程序事件、预算、旧缓存迁移、并发重试和末轮补存。测试使用临时SQLite/Fake Provider，不代替真实模型、GPU或真机生命周期验收。
