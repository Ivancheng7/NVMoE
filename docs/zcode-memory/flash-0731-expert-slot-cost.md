---
name: flash-0731-expert-slot-cost
description: routed 专家槽位是每 token 最大单项；实测 90% 花在建 8 块 pinned 主机缓冲上，开销跟字节走不跟调用次数，修法是复用槽位而不是动同步
metadata:
  node_type: memory
  type: project
  originSessionId: sess_ede91cfb-ffeb-433b-af0d-e7f7921c2f7c
---

2026-09-16 实测（插桩，非估算）：`ExpertSlot::new` 每次 **14.40 ms**，其中 **8 块 pinned 主机缓冲的构造占 12.96 ms（90%）**；权重 H2D + stream 同步 1.16 ms；7 块设备缓冲、5 个 CUDA event、两次 stream 同步、fp8 块、fp4 臂查询加起来不到 0.3 ms。每 token 调用 258 次（43 层 × 6 专家）= **3.71 s**，与 8 步稳态算出的 84.3 ms/层 × 43 = 3.62 s 相符。

**开销跟字节数成正比，不跟调用次数。** 4.4 MB 的 `w1` 一块 3.66 ms（约 0.83 ms/MB），16 KB 的小缓冲只要 3.9 µs。所以问题不是"调用太频繁"，是"每次重新锁 13.4 MB 页再拷一遍"。（我先前怀疑的"每专家同步两次 stream"是错的，别再往那个方向走。）

`PinnedHostBuffer::from_slice` 来自依赖 `cuda-host`（NVlabs/cuda-oxide，rev 8d02eac）：`cuMemHostAlloc` + `copy_nonoverlapping`，析构走 `cuMemFreeHost`。依赖内部改不了，**只要我们自己不重复创建就行**。

**修法方向**：槽位复用——pinned 缓冲只建一次，之后每个专家往 `as_mut_slice()`（依赖已提供该接口）memcpy。上限收益 3.34 s/token，占稳态 12.5 s 的 27%。约束：pinned 缓冲是异步 H2D 的源，复用它之前必须保证上一轮拷贝已完成（现在靠函数体内那两次 stream 同步保证，不能顺手删）。

**页锁定和 memcpy 的拆分已经测出来了（2026-09-16 晚）：页锁定是大头。** 复用改动保留了 memcpy、去掉了重复锁页，每专家从 14.40 ms 掉到约 2.4 ms，反推 13.4 MB 的拷贝只要约 1.2 ms（约 11 GB/s）。所以"开销跟字节走"里的那一份，主要是 `cuMemHostAlloc` 的锁页成本，不是拷贝带宽。

**测量方法**（下次要拆别的热点照抄）：`nsys` 用不了，2024.6.2 在这台 WSL2 上抓不到 CUDA 数据（导出时报 `does not contain CUDA trace data`），别再试；改用插桩，脚本 `.zcode/slot_probe_patch.py`（apply/revert，只在 `new` 函数范围内替换并留 `.probe.bak`，因为 `new_empty_for` 有同样的行），打在 WSL clone 上，测完立即 revert + 重建，确认 `git status` 干净且四个钉死值仍复现。插桩构建 60 条警告，干净构建 59。

证据：`/srv/localmoe/targets/slot-probe-1step-c.{jsonl,log}`、`post-probe-revert-1step.jsonl`；结论写进 `docs/experiments/2026-09-16-sm120-nonresident-baseline.md`（提交 `c8baf83`）。

**已经做完并验证过（2026-09-16 晚，`exp/slot-reuse` 提交 `c3c1be8`，未合并到交付分支）**：`ExpertSlot::new_or_restage` 每层只建一次槽位，6 个专家轮流重填 pinned 缓冲（和 QD4 那条路的 `load_range_execute_into` 同一个做法）。**8 步 `103.38 s` → `70.06 s`（−32%），稳态每 token `12.5 s` → `8.28 s`（−34%）**；八个 token 的 hidden digest 逐个相同、四个钉死值复现、峰值显存不变——数值零变化。记录：`docs/experiments/2026-09-16-sm120-slot-reuse.md`（提交 `bb4b4af`）。

**顺带发现的两个位移**：`shared` 降 23.5 ms/层（与预取线程自己的 read+alloc 降幅一致——锁页 churn 原来在拖慢预取线程）；`fixed` 反而涨 22 ms/层（**没解释**，猜测是固定参数预取每层仍要 148 ms 而层时间只剩 193 ms，藏身余地变少）。下一个瓶颈大概在 fixed 那边。

**合并之后残留的 `routed.setup` 是 `24.8 ms/层 = 1.07 s/token`**（`slotreuse-8step.jsonl` 344 层平均，交付分支 `67e80c3`），仅次于读盘那 `35.9 ms/层`。2026-09-16 的裸盘量测证明**这笔钱不在盘上**（属锁页/pinned 构造那条线），别把它和读盘那件事混着解决——见 [[flash-0731-routed-read-ceiling]]。

**2026-09-17 晚更新（当前帧的数字 + 下一步的口子）**：`routed.setup` 现在实测 **`10.3 ms/层`**
（`arenaab-arena-fix-b`，不是上面那个 24.8 ms/层——池子接上之后又降了一截），
`= 0.44 s/token`，是"专家路径 1.40 s"里除读盘和 kernel 之外唯一多出来的部分。
它**具体在干什么**：`ExpertSlot::restage_weights` 把刚读到的 `RoutedExpert` 六个切片
`copy_from_slice` 进槽位的锁页缓冲（12.75 MiB/专家 × 6 = **76.5 MiB/层**），再 H2D。
**同一份 3.3 GB/token 被搬两遍**：读盘 → 普通宿主 `Vec`，再 memcpy → 锁页，再 DMA → 显存。

**要省这趟 memcpy 的代价（分析，未实测）**：

- **锁页缓冲从 1 份变 6 份**（读盘是 6 条并发流，上传只有一个槽位在漏）：12.75 → 76.5 MiB。
  锁页成本 17.4 ms/12.75 MiB 是一次性的，摊到 8 token 约 +11 ms/token，可忽略。
- **worker 与上传之间的缓冲所有权变成池子问题**：现在 worker 的目标缓冲是私有的、槽位是主线程独占的；
  合成一个就得交接，并保证没有 worker 因等缓冲而卡住。
- **真正要担心的**：这块盘对"读到什么缓冲里"极其敏感——[[flash-0731-routed-read-ceiling]] 里量过，
  同样单线程 buffered，"读进新申请缓冲"对"缓冲复用"差 **1.3 s/token**。**锁页是第三种状态，没人量过**；
  省 memcpy（约 0.33 s/token）之前先花一次量测确认读进锁页不会更慢。
- **别把跨层重叠弄丢**：今天第 L 层的读在 shared 窗口、上传在 routed 窗口，两层之间是重叠的；
  合成一块缓冲最容易顺手把这个重叠也合掉。

相位账怎么读见 [[flash-0731-phase-profile-reading]]，分支与回滚规矩见 [[experiment-branch-hygiene]]，
这一项在 token 预算里的排名见 [[flash-0731-per-token-floor]]。
