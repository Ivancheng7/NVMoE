---
name: experiment-branch-hygiene
description: 实验性代码必须放 exp/* 分支，交付分支只保留已验证状态；回滚用追加式 revert，不改写历史（多会话共享同一分支）
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess_ac8d02c7-eb80-4431-a625-03d658fb6503
---

用户在 2026-09-16 对我把"固定参数常驻显存"的实验代码直接提交到共享分支 `codex/business-restart`
提出异议：**"这是实验代码…感觉怎么管理…分支管理更好吧"**。

**Why:** 这个仓库同时有多个 ZCode 会话在同一分支上工作（见 [[concurrent-agent-sessions]]），
交付分支上堆积未验证的性能改动会让别人分不清"哪些是已验证的路径"。即使改动默认关闭、
输出逐位不变，也不该占用交付分支的历史。

**How to apply:**

- 实验性改动（性能、新机制、开关式特性）放 `exp/<主题>` 分支；交付分支只收已验证状态和文档记录。
- **不要改写共享分支的历史**（另一个会话可能已经拉过）：用追加式 `git revert` 撤回，
  代码留在 exp 分支上，重新拿到用 `git cherry-pick <hashes>`。
- 记录（`docs/experiments`）留在交付分支，并在里面写明代码在哪个分支、怎么 cherry-pick 回来——
  否则文档里的复现路径会指向不存在的代码。
- 撤回后要重新构建并在真机上确认基线仍可复现，再报告"已回滚"。来源：`docs/experiments/2026-09-16-sm120-fixed-residency-ab.md`。
