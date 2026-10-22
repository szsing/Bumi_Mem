# PR #35 与组织 main 整合

2026-10-16，将组织 main（c5205b2）合入 Graphiti release 分支，保留两边历史，无强制推送、不改 main。

## 运行入口与冲突取舍

- HTTP 与 CLI 继续使用 Graphiti / Neo4j、异步事件循环访问、本地 embedding 与 BGE；保持原 10 轮抽取、社区水位和不确定写入授权机制。
- Graphiti HTTP/CLI 两档检索默认均为30秒；保留普通/显式检索独立配置能力。引擎的历史默认及旧 Mem0 配置不作为 Graphiti 服务配置。
- 使用 main 的本轮可见召回优先、来源分离、引用选择日志与精简投影。Graphiti 候选在相应优先级内按类内排名交错，episode 展示清理来源 JSON，保留角色、时间与正文。
- Graphiti 的 valid_at / invalid_at / expired_at 同时进入结构化时间列表；来源时间与事件时间保留，存储更新时间不冒充事件时间。
- 合并 main 的降级、冷却与半开恢复探测，适配 AsyncMemoryAccess。状态判定按派发序号更新，后台认领真实作业前探测。保持前台读不在后台写之后串行等待。
- 保留 main 对旧 Mem0 作用域与已有记忆来源过滤的修复；协议和替身兼容 existing_memory_filter 与 Graphiti replay_authorization 两个可选参数，Graphiti 写入不接受旧后端过滤回调。
- 路径继续用 BUMI_* 和独立 Graphiti 运行目录，预检保持不依赖 Mem0 SDK。旧文档与旧后端代码保留，不作为新部署入口。

## 验证

- 独立 worktree 完整测试：622 passed，7 skipped，33.95秒。
- 复用 memory_v4_002 已保存 raw_rows，q6/q11/q12/q13/q16/q27/q54/q58 的8处关键原文仍进入24000字符预算。
- 新增 Graphiti 暂时不可用、冷却期不重复探测、恢复后三态更新测试；已有慢写不阻塞前台读取测试通过。
- 未重新构图、未重新运行真实模型评分，不将合并前的96.8%成绩宣称为合并后实测。
- 真实凭证、Neo4j/SQLite、运行日志、独立测评产物不纳入提交。
