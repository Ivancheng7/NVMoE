---
name: flash-0731-attention-plus-pinned-stack
description: "attention 设备链 + routed 读锁页两个改动改同层不同相位，合并后各自保持增益、18 跑 digest
  逐位同；合并树在 exp/routed-pinned-replay 提交 52433f4"
metadata:
  node_type: memory
  type: project
  originSessionId: sess_e71e7b77-4ac9-47ac-8cf0-b0f4d3e91485
---

2026-09-18 把隔壁 `exp/attention-device-chain`（attention 18 步搬设备侧，
`6c66759`+`b694858`+`c9631c1`）合并进本分支 routed 读锁页（`52433f4`），
两个改动都在 `layer_rows.rs` 但改不同区域，自动合并无冲突、编译通过（62 警告）。

三臂交错 A/B 6 轮（改动前 / 只 routed / 两个都带），**相位账叠加成立**：
attention 16.99→13.33 ms/层、routed 22.63→18.44 ms/层，两个数各自与"只带自己"
时相同（±0.01），说明互不干扰；`outer` 每 token 省 264 ms。**18 跑 8 步 digest
全逐位相同**。记录 `docs/experiments/2026-09-18-sm120-attention-plus-pinned-lanes.md`。

**Why:** 用户问"隔壁 attention 改动没合进来，你测的带了吗"——答案是没带，两个
分支分叉在 `0e732e4`。分开报收益会让人误以为能直接相加，所以实测合成。

**How to apply:** 墙钟仍噪声主导（6 轮配对中位数 both−base −1.18 s、范围
−7.48…+0.86，`base>routed>both` 只 4/6 轮），**判据一律用相位账不用墙钟**——
这台机同一天能飘 ±3 s。两个改动都还没合进交付分支 `codex/business-restart`；
要合建议顺序 attention → routed。相关 [[flash-0731-pinned-lane-read-landed]]。
