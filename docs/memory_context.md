# 事实正文、来源追溯与精简上下文

普通回答使用人物归属、事实正文和必要时间；内部来源ID不应成为事实的一部分。来源ID、向量记录ID、完整抽取分块元数据仍保留在后台，用于撤回、删除、去重和排错。

## 写入与来源

`Mem0LongTermMemory.add_session()`在原生SDK的两处边界适配抽取：

1. 抽取提示要求将支持消息ID放在每个memory对象的`source_message_ids`数组，不放在`text`中。
2. SDK执行embedding和hash去重前，解析抽取JSON。兼容移出旧提示生成的、带明确标签且属于本块的来源注记，保留同一括号里的日期和其他事实。编造的结构化来源ID、仍夹带输入来源ID的正文会在写入前作为抽取错误拒绝。
3. 原生SDK批量插入时，把识别到的来源写入`fact_source_message_ids/fact_source_messages`，并在声明成功前回读核验。两处适配都位于已有串行写作用域中，结束或失败后恢复SDK方法。

完整分块级`source_message_ids/source_messages`仍保留，不缩小既有来源撤回范围。`fact_source_*`是模型提供、经输入集合校验的逐事实引用，并非额外完成了语义蕴含证明；模型未提供精确引用时继续使用完整分块来源，不猜一个来源。

事实示例（省略其他存储字段）：

```json
{
  "memory": "Jon plans to host a dance competition in May 2023.",
  "metadata": {
    "source_message_ids": ["chunk-message-1", "chunk-message-2"],
    "fact_source_message_ids": ["chunk-message-2"]
  }
}
```

## 回答展示

chat、旧exchange/Agent prepare的共享引用，以及结构化单轮接口，使用同一投影策略，引用字段为：

```json
{"user_id":"Jon的真实用户ID","content":"Jon plans to host a dance competition in May 2023.","time":[]}
```

- `user_id`始终保持真实所有者，门控、跨用户引用可见性不改变。
- `content`保留否定、数量、计划/完成等限定；历史或其他改变事实含义的状态在正文中表达，不逐条下发默认状态。
- 结构化接口`time`维持既有的时间字符串数组契约。chat在数组元素里加短标签：`event_at/event_date`为事件时间，`source_at`为原始来源时间，`input_at`为抽取输入时间；全局规则明确后二者不等于事件时间。
- 不发送记忆UUID、存储更新时间、来源UUID、默认`active`和默认事实状态。后台API/存储和日志仍有内部ID，供诊断使用。
- 旧记忆只在展示时兼容清理带明确来源标签、且ID能匹配后台来源集合的注记；不批量改写原库、历史向量或显式remember原文，不盲删括号、业务编号或未知引用。
- 已缓存轮次保持既有幂等回执；新轮次按新投影组装，不追改历史回答。

6000字符预算仍按最终序列化的共享引用数组计算，包括字段和JSON分隔符，放不下则省略整条。L1、系统规则、核心状态和历史状态不属于该共享引用配额。本次没有调整top-k、引用优先级或预算大小。

## 排错

按请求/轮次找到`shared_memory_refs`中的内部记忆ID，再从对应存储行的`fact_source_message_ids`和完整来源链找到原始`messages`。引用选择日志仍输出选中/遗漏的所有者与内部ID，避免为了让模型看到ID而浪费上下文。日志不输出事实正文。

## 验证记录

- 回归覆盖真实原生SDK的embedding、hash去重、Qdrant写入/回读；LLM/embedding受控替身用于可重复断言。
- 真实DeepSeek + Qwen3-Embedding + 本地Qdrant独立冒烟：3段Jon原始发言生成3条记忆，正文均无来源ID，3条均保存逐事实来源，完整分块来源保留，实体加成off。
- 冻结LoCoMo前两组304题的同一6080条top10候选，只做离线投影/选择：6000预算下从3327条提高到6032条，平均10.94→19.84条/题，262/304题放满20条。使用新chat的带类型时间标签，检索、排序、原库与答案未改。
- 容量提升不等于准确率提升；本次没有重新生成304题答案或宣称新准确率。仍有42题未放满20条，原有用户间引用优先级也保持不变。

本地核查产物目录：`/data1/zhangran/mem0-runtime/results/memory-presentation-fix-20261015`，含真实抽取请求/响应、真实存储结果、模型视图和离线逐题复核脚本。
