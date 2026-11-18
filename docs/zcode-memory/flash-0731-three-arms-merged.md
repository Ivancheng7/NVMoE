---
name: flash-0731-three-arms-merged
description: 三件已量改动已合进交付分支 codex/business-restart（dc22c8e/628c8c0/db35193）——attention 设备链 + routed 读锁页 + final head 并行；合并树真跑 8 token、数值逐位不变、同批 A/B 增益叠加
metadata:
  node_type: memory
  type: project
  originSessionId: localmoe-clean-a1f0d1dadf0989bf
---

2026-09-18 把三件**已在实验分支量过、各自数值逐位不变**的改动合进交付分支
**`codex/business-restart`**（不是 main；main 是本仓库另一条、落后交付 115 笔的线）：

1. `business-restart` **fast-forward** 到 `exp/routed-pinned-replay`（`14ddffb`）——
   该分支已含 attention 设备链 `6c66759`，所以不用单独合 attention。
2. `--no-ff` 合并 `exp/final-head-parallel`（只碰 `v4_state.rs`）→ 合并提交 `dc22c8e`。
3. 文档提交 `628c8c0`（实验记录 + 现状页）、`db35193`（指向新交付树）。

**验收（合并树真跑，`/srv/localmoe/build/deliver-merge-0918`，二进制 SHA `b924f876…`）**：
1 步四个钉死值全 OK；8 步对交付参照 `defaults-pc-8step.jsonl` 的 hidden digest / 43 层摘要链 /
argmax / generated_ids / stop / layers **六项全 MATCH**，**只有峰值显存不同**。生成的 8 个 token
仍是 `55222,33832,42549×5,127384`。

**峰值显存 `1823014912 → 1825112064`（`+2 MiB`）**，恰好是 attention 设备链那套 scratch
（`AttentionDeviceScratch`）；final-head 只多主机侧 `Vec`，不进这个数。**代价不是数值**。
⚠ 现状页长期钉的 `1500053504` 是更早、开关没全接上时的旧值，别拿它当当前基线。

**同批交错 A/B（同一二进制，只切两个开关；减开臂 = `COLIBRI_V4_ATTENTION_DEVICE_CHAIN=0`
+ `COLIBRI_V4_FINAL_HEAD_THREADS=1`）**：`attention 18.6 → 13.5 ms/层`、
`final head 稳态 440 → 102 ms/token`——**各自与它们在实验分支上单独量到的增益一致，
说明三条改不同相位、叠加不打架**。routed 读锁页**没有开关**（切不了），用
`routed.diagnostics.alloc_ns = 0.00`（旧路径约 `4.94 ms/层`）验它生效。

**Why:** 用户问"直接合并然后交付"，但目标分支是 `codex/business-restart`（项目文档里的交付分支），
不是字面 main。这次是"量完就搬"那条规矩的收口：三件都从 `exp/*` 搬进了交付树。

**How to apply:** 备份 ref `backup/business-restart-20260918` 指向合并前的 `0e732e4`。
合并用的隔离 worktree 是 `Desktop\localmoe-deliver`（Windows 侧），**别在 `localmoe-clean`
上动**——那棵有并发会话、还有未提交的 `AGENTS.md`。三件改动里 routed 读锁页**只覆盖
`FLASH_BASE + PREFETCH` 路径**（QD4/stationary/prefill 的读槽池是 None，回落旧 Vec 路径）。
**服务面没重验**：三件在 serve 上默认生效但一次 serve 没跑，要按钉死值复跑。

相关 [[flash-0731-attention-plus-pinned-stack]]（那两件在实验分支上的合成 A/B）、
[[flash-0731-final-head-parallel]]、[[flash-0731-pinned-lane-read-landed]]、
[[flash-0731-per-token-floor]]（新账目要按这次重构）、[[experiment-branch-hygiene]]、
[[concurrent-branch-hijack-during-commit]]。
