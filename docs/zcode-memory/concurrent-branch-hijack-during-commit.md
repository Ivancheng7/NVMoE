---
name: concurrent-branch-hijack-during-commit
description: 工作树被并发会话切分支，commit 会落到错的分支上；改完先 git branch --show-current
metadata:
  node_type: memory
  type: project
  originSessionId: sess_05af1aca-e68c-4bc0-b4d1-2594310bf351
---

2026-09-18 实测：在 `localmoe-clean` 这个 worktree 上，创建并切到新分支后，**另一个并发会话中途把
HEAD 切走了**（reflog 显示 `checkout: moving from exp/attention-device-chain to exp/routed-pinned-replay`）。
结果 `git commit` 落了错误的分支（`exp/routed-pinned-replay` 上多出一个 attention 提交）。

处理办法（不改写别人的分支历史）：

1. `git branch --show-current` —— **每次 commit 前都查**，别信上一次看到的分支名。
2. 落错了就 cherry-pick 到正确分支：`git checkout <正确分支> && git cherry-pick <sha>`。
3. 把被污染的分支指回它自己的 tip：`git branch -f <被污染分支> <它原来的tip>`。
   （前提：那个 tip 已经被别的分支包含、不会被误删；见 `git branch --contains`。）

同一个仓库里还有多个 worktree
（`localmoe-fixed-host-reuse`、`localmoe-pinned`、`localmoe-prompt-chunk`…）和 WSL 侧的 clone
`/srv/localmoe/build/localmoe-clean-git`，都可能被别的会话动。相关：[[concurrent-agent-sessions]]、
[[experiment-branch-hygiene]]。
