---
name: flash-0731-pinned-read-green-light
description: "restage 省掉的前置量测已做：读进锁页 vs 复用缓冲无差异，go；工具与证据的分支是 exp/routed-pinned-replay"
metadata:
  node_type: memory
  type: project
  originSessionId: sess_e71e7b77-4ac9-47ac-8cf0-b0f4d3e91485
---

2026-09-17 量完"省 restage"的前置关卡：`disk_ceiling.c` 加 `pinned` 模式（真
`cuMemHostAlloc`，dlopen），6 线程 buffered、真实路由轨迹（slotreuse-8step 的
route_ids，12,384 次 pread / 27.594 GB），与 reuse 档同轮交错三轮：pinned
7.111/4.336/4.721 s vs reuse 7.743/4.655/4.357 s，方向不一致、在噪声带内，判
**没有惩罚**；digest 两种模式逐位相同。证据在
`/srv/localmoe/targets/io-ceiling/pinned-ab-0917.jsonl` + `plan-real-routed.tsv`，
记录 `docs/experiments/2026-09-17-sm120-pinned-read-replay.md`。

**Why:** 用户分析里"锁页是第三种状态，没人量过"是做改造前唯一的未知数；
量过了它就不再是拦路虎——fresh 缓冲的 1.3 s/token 惩罚来自缺页+清零，锁页
是预 fault 的复用档。[[flash-0731-expert-slot-cost]] 里 90% 是锁页不是拷贝的
旧账与此不冲突：那是"每次重建锁页"的成本，池化后只剩 6×17.4 ms 一次性。

**How to apply:** 下一步做引擎改造时不用再插量测：预取 worker i 认领第 i 套
锁页缓冲（静态、零锁），读用现成的 `RoutedExpertRangePlan::read_into`（QD4
lane 已在用这个形状），别让下一层的读排上一层的槽位（会弄丢 worker∥shared
MLP 和 fixed 跨层流水两个重叠）。工具分支是 `exp/routed-pinned-replay`，
**不是** `exp/routed-pinned-read`——后者被并发的另一会话占用指到 final-head
提交上了；多会话共用工作树，提交前先 `git branch --show-current` 核对。
