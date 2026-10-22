# 管理员记忆网页（第一版）

本地 FastAPI 页面 `/admin`，直接管理当前 Neo4j 实例中的所有用户/agent分组。沿用现有服务Bearer密钥，不增加账户、角色或用户级权限。

## 使用

当前服务器已启动在 `127.0.0.1:18109`。在VS Code的“端口”面板转发18109，浏览器打开 `http://127.0.0.1:18109/admin`。也可从电脑建立SSH隧道：

```bash
ssh -L 18109:127.0.0.1:18109 用户名@服务器
```

连接密钥使用服务器私有文件 `/data1/sunzhixing/Bumi_mem_2_test/configs/local.env` 中的 `BUMI_SERVICE_API_KEY`。页面不内置密钥，不将密钥放在URL或浏览器持久存储中。页面本身不包含记忆，数据接口沿用现有认证。

重新启动示例（端口已运行时不要重复启动）：

```bash
cd /data1/sunzhixing/Bumi_mem_2
export BUMI_SERVICE_ENV_FILE=/data1/sunzhixing/Bumi_mem_2_test/configs/local.env
export BUMI_ADMIN_DATA_DIR=/data1/sunzhixing/Bumi_mem_2_test/runtime/admin_console/data
export BUMI_ADMIN_GPU=4
bash scripts/run_memory_admin.sh
```

管理端使用独立SQLite和agent，不与聊天进程争用相同SQLite文件锁；Neo4j连接复用外部配置。默认仅监听本机18109。

## 第一版功能

- 查看全部用户分组；有来源JSON的分组显示user_id和agent_id，否则显示group_id。
- 分页查看事实关系，按文字与有效/已删除状态筛选。
- 查看关系端点、事实时间、原始episode与最近50条修订历史。
- 编辑事实文字，重新生成1024维向量；提交版本号，旧版本覆盖返回409。
- 可恢复的逻辑删除与恢复，删除不等于删除原聊天、备份或审计记录。

## 存储与检索

关系原来的UUID和group_id保持不变。编辑同步更新 `RELATES_TO.fact` 与 `fact_embedding`；关系端点、关系名和事实有效期暂不重新抽取，第一版不提供结构编辑。

Neo4j的 `BumiAdminMemory` 保存管理员修订/删除状态，`BumiAdminRevision` 保存操作前后内容及时间。同一UUID后台重新写入时，修订层仍优先于自动结果。若未来抽取生成不同UUID但语义相似的新事实，当前版本没有语义级防复活机制，需要后续按来源撤回或管理规则处理。

手动修订通过当前分组的长期召回优先提供；删除条目被过滤。只要分组存在修订，暂不向回答端提供旧的实体摘要、episode和社区摘要，以免重新带入旧内容，其余事实继续可用。该行为不是永久重建图谱摘要，后续可增加按来源重建。

同一管理服务SQLite中的共享引用和上下文缓存会失效。其他运行中服务需要加载这版源码后重启；其他独立SQLite实例已经保存的共享引用、缓存响应不会被跨进程主动清除。新轮检索会应用共享Neo4j修订。同一turn_id的历史幂等响应不能当作新请求验证。

## 接口

| 方法与路径 | 功能 |
|---|---|
| `GET /v1/admin/groups` | 全图分组 |
| `GET /v1/admin/memories?group_id=…&q=…&status=active&offset=0&limit=30` | 列表 |
| `GET /v1/admin/memories/{id}?group_id=…` | 内容、来源、历史 |
| `PATCH /v1/admin/memories/{id}` | edit/delete/restore |

修改请求示例：

```json
{"group_id":"bumi_实际分组哈希","version":0,"action":"edit","content":"用户现在喜欢蓝色。"}
```

删除/恢复分别将action改为delete/restore，省略content，并使用详情中的最新version。

## 验证

真实Neo4j测试仅创建临时管理员测试分组，完成后清理该分组：全图分组、来源标识、列表、详情、编辑、1024维向量更新、新内容实际召回、409版本冲突、逻辑删除筛选、恢复与修订历史均通过。未修改已有用户事实。

已验证HTTP页面响应、API与数据库链路；当前环境没有浏览器控制工具，尚未完成自动化浏览器点击和视觉验收。

## 三维知识图谱

在页面点击“**三维知识图谱**”切换。鼠标拖动旋转、滚轮缩放；节点用透视深度和球面明暗呈现立体效果，可隐藏对话片段/社区或关闭名称。点击节点显示摘要与相邻关系；点击事实连线进入来源查看和编辑。其他连线分别表示episode提及和社区归属，不当作事实编辑。

本地Canvas实现三维坐标布局与透视投影，不依赖外部CDN，也不增加模型调用。节点空间距离只是布局，不代表相关度或记忆强度。图谱只显示未删除的事实关系，管理员修订内容优先。

`GET /v1/admin/graph?group_id=…&limit=500` 返回节点和关系。页面默认最多500个节点、1500条连线，接口节点上限800；超出时显示部分图谱提示。选择具体用户可以查看更完整的分组图，避免全库一次渲染阻塞。图谱节点包含实体、episode与社区，详情文字预览最多2000字符，不返回embedding。

验证：原memory_v4_003完整图谱读取、分组范围、所有连线端点存在、截断标志、事实连线详情入口均通过真实数据库只读检查。页面脚本语法、三维坐标投影和旋转经JS引擎检查；尚未完成真实浏览器视觉与鼠标点击自动化验收。
