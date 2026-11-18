---
name: flash-0731-steady-state-vs-first-token
description: "8 步总时长里 ~40% 是 token0 建固定参数常驻库的一次性成本；判收益要用稳态 s/token，不是步数固定的墙"
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess_e71e7b77-4ac9-47ac-8cf0-b0f4d3e91485
---

2026-09-18 看清的一件影响所有历史数字的事：**token 0 要建整个固定参数常驻库**
（43 层 host payload 解码+页锁定），在 `fixed.read_alloc` 里一次性花 ~327 ms/层 × 43
≈ **14 s**；token 1–7 该项为 0。所以任何"固定步数总时长"里，这 14 s 只付一次、
和步数无关，N 步时占比 ≈ 14/(14+3N)。

实测（`both-changes`，8 步）：token0 = 15.09 s（fixed 278 ms/层），token 1–7 =
2.75–3.38 s（fixed 16–26 ms/层）。**token0 占 8 步总时长的 42%。**

三臂稳态（token 1–7）：改动前 3.386、只 routed 3.235、两个都带 **3.033 s/token**。
两个改动稳态省 **0.35 s/token**，但 8 步墙只看到 2.1 s。按 N 步外推省：
8 步 6%、43 步 9%、100 步 10%。

**2026-09-22 补上反过来的一面**：只改稳态的改动也会被 token 0 吃掉。routed fan-out 的专家身份缓存
稳态省 `0.07 s/token`，但 token 0 多花 `1.2 s`（多锁页那一堆 lane + 第一遍位置全 miss），8 步整条
请求 `68.64 → 69.09 s` 反而略慢。所以报收益要给三个数：**稳态 s/token、整条请求 e2e、token 0**。

**Why:** 之前把"8 步省 2 s"读成"收益就这么大"，是把一次性建库成本摊进了分子。
步数不同、常驻开没开，这个坑都会移动基线。用户问"这是上限吗"时也是被这个数误导。

**How to apply:** 报收益必须同时给**稳态 s/token** 和 N 步总时长，只给后者不算数；
跨记录比数字前先确认两边的步数、常驻状态一致。稳态当前相位格局（两个都带，
ms/token）：routed 785(27.5%)、fixed 783(27.4%)、shared 630(22.1%)、attention
550(19.3%)、router 106——**没有单块占多数**，继续要同时动几块：fixed 稳态每 token
还要 H2D 6.71 GB（27.5 GB/s，已到带宽量级，只能显存常驻或减字节）、routed 读
3.45 GB/785 ms≈4.4 GB/s（已贴近盘上限）。记录
`docs/experiments/2026-09-18-sm120-attention-plus-pinned-lanes.md`。
相关 [[flash-0731-attention-plus-pinned-stack]]、[[flash-0731-per-token-floor]]。
