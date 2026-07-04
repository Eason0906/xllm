---
name: git-push
description: Git 推送工具：自动 add, commit 并 push 代码变更到远程仓库。使用 `git-push <commit-message>` 提交所有修改。
---

# Git Push Skill

用户叫你推代码时，执行以下步骤：

1. 用 `cd /d/code/xllm` 进入仓库目录
2. 运行 `git add -A` 暂存所有变更
3. 运行 `git commit -m "<commit-message>"` 提交（如果用户没有提供 message，使用 "fix: OTPColumnParallelLinear weight loading and quant resolution"）
4. 运行 `git push` 推送到远程

如果 git push 失败（比如远程有新的提交），先 pull 再 push。

报告操作结果给用户。
