> v0.3 补充：个人 L3 的直接管理/读取保持隔离；经本人门控召回的引用会进入共享 L1，包括陌生人可见。`__guest__` 不拥有个人 L3。核心状态、时间、删除与迁移见 `../companion_memory/README.md`。下文原有验收记录不代表新增路径已通过真实模型验收。

> 2026-09-16 更新：本文的“用户隔离”指长期记忆及个人整理任务。短期上下文现按 agent 共享并保留说话人归属，详见 `../shared_short_term/README.md`。历史 REPORT 反映旧版验收，不作为当前 L1 行为说明。

# 本轮仅验收：恢复与 user_id 隔离

## 范围

- 对服务明确确认保存的每条记忆，恢复后memory_id、user_id、正文hash一致。
- 3个固定user_id模拟3人，读写、指定ID读取、检索和回答上下文不串人。
- 不启用家庭共享，不接typed库，不做行为归纳、GPU/检索性能优化。
- 不运行全量回归；只运行本目录专项检查与隔离恢复验收。

## 最小接口

原BumiMemory仍按每次请求显式的user_id/agent_id工作，不保存“全局当前用户”。空、缺失、首尾空白或控制字符ID被拒绝，不回退到demo_user。

- `remember(user_id, agent_id, contents)`：明确写入已给定的长期记忆，不调用LLM提取。真实后端保存后逐条按ID读回核验归属和内容，再返回 `MemoryWriteReceipt(confirmed=True)`。正文按strip规范化，以回执中的实际保存内容hash作为恢复比较依据。
- `read_memory(user_id, agent_id, memory_id)`：知道别人的memory_id也不能读到其内容。
- 既有 `search_long_term/get_memories/end_session/forget_all` 等入口均要求明确身份。真实适配器再次检查返回数据归属。
- 元数据不能覆盖user_id/agent_id/id等归属字段。
- 普通聊天按 agent 的共享窗口触发后台整理，按用户分别保存快照；queued/pending或已保存会话，不等于已经成功写入LTM。

部分写入或核验失败不会返回confirmed，也不会自动重放；本轮不承诺未知结果重试严格exactly-once。

## 开发CLI

```text
--user-id user_A     显式模拟已由可信前端确认的身份
/user user_B        切换；共享本 agent 的 L1，但只检索 B 的 L3
/logout             清除当前身份，不删除历史记忆
/remember 我喜欢绿茶  明确写入并输出确认和memory_id
/read <memory-id>   只读当前用户有归属的记录
```

不提供user-id时默认未识别，memory-enabled CLI锁定长期记忆相关操作，不自动选择上次用户或demo_user。未识别短聊可由前端独立Chat通道提供，本轮不增加匿名持久会话。

CLI只是开发控制台，不是声纹认证或安全的多人显示界面。生产前端必须先完成声纹/可信身份匹配，再传稳定user_id；user_id本身不是密码。共享现场对话允许保留显示；个人记忆管理结果仍须单独保护，并用 `UserIdentity` 票据在最终显示时拒绝迟到回复。当前CLI使用同一轻量检查保护回调。

新的UserIdentity实例/重连从未绑定开始。记忆归属持久保存，不代表当前登录身份自动继承。

## 专项验收

在服务器运行：

```bash
bash /home/legendre/data/ZhixingSun/Bumi_Mem/scripts/run_user_memory_acceptance.sh
```

只创建独立测试目录。每个场景3人×5条明确合成记忆，调用真实Mem0/Qdrant磁盘后端；LLM接口被禁止，聊天Provider仅用于检查组装上下文，不产生自然语言质量结论。

工具进行：
1. 写入并获得确认回执；快照清单原子写入并fsync。
2. 正常退出写入进程，再以新进程逐条验证。
3. 另一个隔离场景中，确认写入后强杀本工具自己创建、正在等待的子进程，再以新进程验证。
4. 模拟前端重连、A/B/C切换、无身份请求、用错误user_id读取记忆ID，以及相同/误导性查询下的归属。

拒绝用空数据库、空检索结果或只比较Top-K来证明恢复100%。快照至少包含15条实际confirmed记录。若期间执行了forget，generation变化会让旧快照失效，不会把删除误当丢失后复活。

恢复检查等待后端初始化，不测时延；工具内部等待上限不改变线上既有读取超时/模型/向量参数。get_memories原有500条列表限制不作为完整恢复依据，恢复逐条按回执ID核验。

## 整机休眠/断电：不由脚本执行

可在独立新run中调用工具的`prepare-host`保存基准，用户安排操作后再用`verify`检查。verify记录前后boot_id和人工事件标签，但boot_id变化只说明发生重启，不能独立证明是断电；休眠和真实断电均需额外操作记录。

任何本轮的进程SIGKILL通过，都不能写成“真实断电已通过”。工具不含reboot、suspend、poweroff或sudo操作。硬盘损坏、错误的数据目录、外部手工篡改、未确认的中间写入不在本轮成功恢复的承诺范围内。

已有磁盘数据不会自动重新绑定用户，未绑定/错误归属的遗留记录不会被塞给当前人。主 SQLite 库升级到 v3 时回填消息归属并构建共享短期索引，不更改 L3 归属、不新增权限数据库。

## 兼容性说明

原来直接使用session_id的`record_user_message`、`record_assistant_message`、`build_context`现在必须同时传入user_id和agent_id，并校验会话归属；仓库内调用点已同步调整。前端不要绕过带身份的服务接口直接访问底层store/SDK。

用户切换前的终端显示不等于安全的前端消息历史：正式前端应区分共享现场消息与个人记忆管理结果，按身份票据过滤异步输出。声纹误识别和调用者伪造user_id不由字符串过滤自动解决，本轮只约定可信上游传入已确认的稳定ID。
