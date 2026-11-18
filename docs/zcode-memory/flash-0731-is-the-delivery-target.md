---
name: flash-0731-is-the-delivery-target
description: 交付目标是 Flash-0731 revision 9e165c30…；文档里长期写的 60d8d707… 是另一份 checkpoint，2026-09-16 已统一
metadata:
  node_type: memory
  type: project
  originSessionId: sess_a9e0716d-7d0a-462a-bfe0-b5dd06f16f66
---

`AGENTS.md` 和 `重启说明.md` 长期写着交付目标是官方 `deepseek-ai/DeepSeek-V4-Flash` revision `60d8d70770c6776ff598c94bb586a859a38244f1`。2026-09-16 用户澄清"**我们应该走 0731 版**"。

两个 revision **都真实存在**，是同一个 Flash 系列的两份不同 checkpoint：`60d8d707…` 是基础版 `deepseek-ai/DeepSeek-V4-Flash`，`9e165c30…` 是 `deepseek-ai/DeepSeek-V4-Flash-0731`。`60d8d707` 在代码里**一次都没出现**，权重也从来没下过；代码 pin 的、本地有权重的、实验跑的全是 `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`。

**Why:** 一个写在文档里、却从没在代码或磁盘上出现过的"目标 revision"，下次会有人照着它去下 167GB 权重。

**How to apply:** 只认 `9e165c30…`。`.zcode/plans/` 里还留着提 `60d8d707` 的旧计划文件，别照抄。写文档时定一条自查：**目标 revision 必须能在代码里找到 pin**（`v4_spec.rs` 的 `FLASH_0731_REVISION`）；对不上就是这个坑。同类自查也适用于"运行时会拒绝什么"这类描述——`AGENTS.md` 曾写"拒绝 `/mnt/*`"，实际代码只拒 `/mnt/c` 和 `/mnt/d`，而 README 示例用的 `/mnt/colibri-native` 本来就是合法的。

相关：[[flash-0731-baseline-run-recipe]]、[[concurrent-agent-sessions]]
