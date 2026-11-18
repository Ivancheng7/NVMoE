---
name: flash-0731-pinned-lane-read-landed
description: "读盘直接进锁页已落地实验分支：routed 窗口 22.9→18.5 ms/层、alloc=0、8 digest 逐位同、八步省
  2.4 s；分支 exp/routed-pinned-replay"
metadata:
  node_type: memory
  type: project
  originSessionId: sess_e71e7b77-4ac9-47ac-8cf0-b0f4d3e91485
---

2026-09-18 完成"省掉 restage"的引擎改动（分支 `exp/routed-pinned-replay`，
提交 `575d36a`+`d04fed5`+`95157fc`+`2c3d000`，**未合并交付**）：routed 预取
worker 直接读进请求级常驻的 6 条锁页读槽（`RoutedReadLane`/`RoutedReadPool`，
挂 `RoutedSlotScratch`），消费端用新增的 `ExpertSlot::upload_lane` 从读槽 H2D，
设备侧仍只有一个执行槽位。

同批交错 A/B 六轮（清缓存）：`routed.wall 22.86→18.45 ms/层`、`routed.setup
10.66→6.24`、`alloc 4.94→0.00`（恰好归零=路径真走通）、八步墙 38.95→36.54 s、
**八个 token digest 逐位不变**。记录 `docs/experiments/2026-09-18-sm120-pinned-lane-read.md`。

**Why:** [[flash-0731-pinned-read-green-light]] 量过"读进锁页不慢"；这次是引擎里
兑现。收益**全额上了墙**：6 轮配对 outer 省 319 ms/token、e2e 也省 319 ms/token
（残差 5 ms）。早前一次总结写成"差额被 fixed 抢盘吃掉"是单位算错，已收回——
两条流的读盘字节一字未变（routed_source 80,216,064/层、fixed_source 19,502,759），
谈不上抢得更凶。

**How to apply:** 两个坑记下来——(1) 第一版把 `run_layer_shared_expert` 挪到
spawn 读线程之前，`shared` 窗口就从 `max(mlp,worker)` 变成 `mlp+worker`，重叠
被合掉；共享探针 `COLIBRI_V4_FLASH_SHARED_PROBE=1` 能一眼看出（window≈max
才是对的）。改这条流水必须保证"先 spawn 读、再跑 MLP、再 join"。(2) 只覆盖
`FLASH_BASE/PREFETCH` 路径，QD4/stationary/prefill 没有 scratch，读槽池为 None、
回落旧 Vec 路径。想再拿收益要去动 fixed 的盘竞争，不是继续抠 routed。

A/B 脚本第一版用 `>/dev/null` 吞错，导致一次 A/B 没跑成却报 done（旧 JSONL
mtime 不变被我抓到）；`run_pinned_ab.sh` 已改成跑不进就 FAIL。判据看
`alloc_ns` 是否为 0。
