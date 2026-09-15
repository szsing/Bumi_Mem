# 用户ID隔离与恢复：本轮实现和专项验收

日期：2026-09-15。范围按用户提供的两项要求收敛；不扩展家庭共享、Behavior、typed架构或性能优化。

## 本轮实际改动

1. 原Bumi长期记忆入口必须显式user_id/agent_id；缺失、空白、带首尾空白或控制字符的ID被拒绝，不回退到demo_user。
2. 每条记忆仍存入原Mem0/Qdrant后端，绑定请求中的user_id。元数据不能覆盖归属，返回结果和回答上下文再次核验。
3. 增加`remember`明确写入：实际保存后逐条按ID读回检查内容/归属，再返回confirmed及memory_id/hash。没有另建权限库或恢复真相库。
4. 增加按user_id读取单条memory_id；知道其他用户的ID也不能读取。
5. CLI增加`/user`、`/logout`、`/remember`、`/read`。新的身份连接默认未绑定；迟到回复用易失身份票据判断是否仍可展示。
6. session_id不能代替身份：直接记录消息/组装上下文的接口也需user_id/agent_id，避免把A的session交给B使用。
7. 后台任务继续按原快照的用户ID执行；queued/accepted与已确认长期保存明确区分。

**保留既有实现，不删除typed包，不改模型/索引配置，不修驱动，不测60ms性能。** 检索模型与后端构造配置已作静态一致性检查。

## 真实磁盘恢复与三用户隔离结果

每种进程场景用3个模拟user_id，每人5条明确合成记忆。写入使用实际Mem0/Qdrant磁盘后端，生成确认回执和落盘快照；检验在独立新进程中进行，不用Fake存储证明恢复。

| 场景 | 已确认记录 | ID/归属/内容均一致 | 恢复率 | 跨用户污染 |
|---|---:|---:|---:|---:|
| process_restart | 15 | 15 | 100% | 0 |
| process_sigkill | 15 | 15 | 100% | 0 |

正常退出的子进程exit=0，强杀子进程exit=-9。SIGKILL发生在本工具自己创建的子进程完成全部确认写入后；没有杀其它服务或重启服务器。两种场景都检查缺失ID、正文hash、user_id/agent_id绑定、错误用户读取ID、检索候选和组装上下文。

本轮还验证了新身份连接开始时未绑定、无ID长期读写被阻止、A/B/C切换、旧票据结果拒绝。**这是可信user_id输入的前端重连模拟，不是真人声纹识别测试。**

## 专项代码检查

只运行22项与ID/回执/会话归属/任务归属相关的检查，1.07秒通过；没有跑完整163项回归。模拟聊天Provider只用于检查上下文，LLM调用被禁止，不评价抽取或自然语言回答质量。

## 未做的验收

- 真实整机休眠/唤醒：未执行。
- 真实整机断电/上电：未执行。
- 声纹模块及真人身份识别：未实现、未验收。
- 60ms时延：本轮不测不优化，由同事负责。

因此不能把本报告写成“全部断电场景已通过”。成功率针对固定合成集合与已执行的进程场景，不是磁盘损坏、外部篡改、所有未来故障或任意自然语言问法的保证。

## 如何自行复验

```bash
bash /home/legendre/data/ZhixingSun/Bumi_Mem/scripts/run_user_memory_acceptance.sh
```

无需DeepSeek Key；只在新的隔离目录中操作真实测试库。不会执行reboot/suspend/poweroff，不影响同事的GPU或其它服务。

需要安排整机事件时，用工具`prepare-host`生成基准，操作后再`verify`。它记录boot_id与人工事件标签；boot_id变化只说明有重启，不独立证明电源被切断。未执行的场景必须保留“待验证”。

## 使用与兼容性

- 新CLI默认不选人，必须明确`--user-id`或`/user`才能使用长期记忆。CLI的身份切换是开发模拟，不是认证机制。
- user_id是可信前端确认后的索引/归属字段，不是密码。生产前端须清理旧显示，在最终显示处检查身份票据。
- 旧自动整理流程没有改成新抽取架构；`/new`仍是保存记录和整理任务，不能把pending叫做LTM已确认。
- 回执以实际保存的规范化正文hash为准，列表Top-K不承担完整恢复证明。恢复逐条按ID核对。
- 发生forget导致generation变化时，旧清单会失效，工具不会把删除的数据自动复活。
- 不确定/部分写入不返回confirmed，也不自动盲重试。

服务器实现：`/home/legendre/data/ZhixingSun/Bumi_Mem`。

实现记录：`/home/legendre/data/ZhixingSun/mem0-runtime/acceptance/user_memory/runs/implementation-20260915T120352-QFKuMN`。

最终真实验收数据：`/home/legendre/data/ZhixingSun/mem0-runtime/acceptance/user_memory/runs/check-20260915T121715-bAI9ZC`。

本轮代码已落服务器，未自动提交或推送GitHub。原有程序入口中的会话助手方法签名有所收紧，外部直接调用者需补上user_id/agent_id；仓库内相应调用与一个直接调用测试已更新。
