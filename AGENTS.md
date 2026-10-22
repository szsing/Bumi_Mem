# Bumi_Mem

> 中心级约定，改代码前先看。仓库自身的说明见 `README.md`。

## 提交方式（中心 2026-09-18 定，Noetix-AI 下所有仓库通用）

**任何代码更新走分支 + PR，不直推 `main`——包括中心负责人。**

```bash
git config core.hooksPath .githooks   # clone 后装一次：本地拦住推 main
git switch -c <类型>/<简述>            # feat/ fix/ docs/ chore/ …
git push -u origin HEAD
gh pr create                          # PR 里写：改了什么、怎么验证的
```

- `main` 由仓库负责人或中心负责人合并；自己开的 PR 可以自己合，但要先有人能看见改动。
- GitHub 免费版的私有仓库开不了分支保护，所以技术上挡不死：`.githooks/pre-push` 挡手滑，绕过去（`--no-verify`）也会被 `guard-main` Action 开 Issue 记一笔。
- 不要 force push `main`。
- commit 信息：`<类型>: <摘要>`；PR 里写清验证方式（跑了什么、结果是什么），别只写「已测试」。

## 给 code agent 的话

以上对 AI 编码助手同样适用：**不要代替人直推 `main`**，改完推分支、开 PR，把验证结果写进 PR 描述。
