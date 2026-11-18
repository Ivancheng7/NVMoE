---
name: flash-0731-routed-expert-cache
description: routed 专家按身份跨 token 缓存（A2，默认关）：复用 63%、读盘 −57~79%，但墙只 −3.6%（读被藏在
  shared 窗口）；命中要 ~258 槽/3.2 GiB，低于它命中率为 0
metadata:
  node_type: memory
  type: project
  originSessionId: sess_267fb0a4-2742-48a8-8acb-43fc2647d3c4
---

2026-09-21，交付分支 `codex/business-restart`，代码 `10f0958`、记录 `1f73973`
（`docs/experiments/2026-09-21-sm120-routed-expert-cache.md`）。开关
`COLIBRI_V4_ROUTED_EXPERT_CACHE=<槽数>`，**默认关**（0）。

- 按 `(layer, expert)` 键存**整条锁页 lane**，命中即零拷贝交 H2D（`RoutedExpertCache`）；
  `RoutedReadPool` 改成按值 `take_lanes`/`give_lanes`，池与缓存共用一条 worker/排空路径。
  逐位不变：1 步四个钉死值复现、8 步 digest 对 off 与对 `defaulton` 参照都相同。
- **复用结构（可离线数）**：引擎在结果 JSONL 的每层记 `route_ids`，**不用跑新东西就能数出复用率**。
  真 32 token prompt 上连续 token 同层重叠 `63.2%`，8 步内每层只碰约 24/256 个专家；
  单 token prompt（`TOKEN_IDS=0`）会自回归成重复模式（第 5/6/7 个 token 的 route 行逐字节相同），
  复用率虚高，**别拿它当样本**。
- **缓存要多大**：复用距离是**一整趟 43 层的扫**，所以 LRU 必须约等于**一个 token 的工作集 =
  43×6 = 258 槽（3.2 GiB 锁页）**才起作用——256 槽命中率就掉成 `0`（258→43%、344→52%）。
  静态"钉最热专家"上界在 258 槽也只有 `54%`，不比 LRU 好多少。
- **实测**（`a2-ab/`、`a2-final/`，盘在 1.7–1.8 慢档）：`cache_hits = 1028`（与离线数一致）、
  每 token 磁盘读 `258 → 111`（单 token 形态）/ `54`（真 prompt），`source_bytes 3.21 → 1.38/0.68 GiB`；
  但墙只 `shared 0.699 → 0.614 s/token`、`outer −0.080 s/token（−3.6%）`（单 token 形态，三轮不重叠），
  真 prompt 形态在噪声里（一轮撞上慢盘离群）。
- **为什么读省了墙没动**：读盘本来就被藏在 `shared` 窗口里（只有 join 尾巴露出来），砍短 lane 读墙
  只露出一小截。这正是 [[flash-0731-phase-profile-reading]] 那条"读盘是 routed 那摊的、墙记在
  shared 名下"。**它单独不是加速**；与 A1 叠加的净结果见 [[flash-0731-routed-fanout]]（只再省 0.07）。
- 代价：258 槽 × 12.75 MiB ≈ **3.2 GiB 锁页主机内存**（机器共 15 GiB），所以默认关。
  `cudaMallocHost` 实测 344 × 12.75 MiB = `4.28 GiB` 能分配成功（`ulimit -l` 显示 64 MiB，但**不限制**它）。
- **坑：构建脚本只同步了 3 个文件**，漏了发字段的 `b2_generate.rs`/`phase_profiles.rs`，于是二进制是旧的、
  `cache_hits` 全 0。**把源码抄进 WSL 树时要按 `git status` 同步全部改动文件。**
  另：WSL 树里 `git diff` 报"5801 行改动"是行尾假象，`--ignore-cr-at-eol` 后正好等于真 diff。

相关：[[flash-0731-routed-fanout]]、[[flash-0731-phase-profile-reading]]、[[flash-0731-per-token-floor]]、
[[flash-0731-routed-read-ceiling]]
