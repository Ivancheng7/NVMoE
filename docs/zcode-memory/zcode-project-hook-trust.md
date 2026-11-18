---
name: zcode-project-hook-trust
description: ZCode 的项目级 hook 被「工作区信任」闸门挡住，以及这个客户端根本没有 Grep/Glob 工具——写 hook 前必看的两条实测事实
metadata:
  node_type: memory
  type: project
  originSessionId: sess_d6840045-250c-49a4-b765-ec7d16d5d0b2
---

给 ZCode 写项目级 hook（`<repo>/.zcode/config.json` 的 `hooks`）有两个坑，都在 2026-09-16 实测确认：

1. **没有 Grep/Glob 工具**。ZCode 的日志里出现的工具名只有 `Bash`（1700+ 次）、`Read`、`Edit`、`Write`、`TodoWrite`、`Agent`、`Skill`、`WebFetch`、`AskUserQuestion` 和 `mcp__*`，没有 `Grep`、`Glob`。所以从 Claude Code 抄来的 `"matcher": "^(Grep|Glob)$"` 会注册成功但**永不触发**，日志里一条 hook 记录都没有。要用 `^(Grep|Glob|Bash)$`。
2. **项目级 hook 有信任闸门**。日志出现 warn `config.project_hooks.pending_trust`（"Project hooks pending workspace trust and remain blocked"）就说明 hook 被挡着不执行。放行要人工在 ZCode 界面 设置 → 钩子 点「信任」（i18n key `settings.hooks.review.trust`；提示语"钩子可在沙盒外运行，因此请审查最近安装或修改的所有钩子"）。信任状态绑定 hook **声明摘要** `hookDeclarationDigest`，所以每次改 `.zcode/config.json` 里的 hooks，都要重新点一次信任（状态回 `pending_trust`/`stale_digest`）。用户级配置 `~/.zcode/cli/config.json` 的 hooks 没有这个闸门。

**Why:** 这两条不写下来，下次会配出一个永不触发的 matcher，或者以为 hook 装好了其实一直被信任闸门挡着——两种都是「看起来装好了、实际没跑」。

**How to apply:** 写项目级 hook 时 matcher 带上 `Bash`；交付后查 `~/.zcode/cli/log/zcode-*.jsonl` 里有没有 `pending_trust` 或 hook 执行记录，再告诉用户去 设置 → 钩子 点信任。相关：[[codebase-memory-mcp-install]]、[[concurrent-agent-sessions]]
