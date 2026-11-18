---
name: flash-0731-phase-profile-reading
description: Flash-0731 引擎自记的相位账怎么读才对（顶层是互不重叠的墙、diagnostics 会重复、shared 是 max() 而不是自己的耗时），以及定版不常驻基线的每 token 分解
metadata:
  node_type: memory
  type: project
  originSessionId: sess_ede91cfb-ffeb-433b-af0d-e7f7921c2f7c
---

`phase_profile`（结果 JSONL 里，总是写出）只有**顶层那几段是互不重叠的墙**，代码里有闭合等式并在运行时校验：`top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns`（发射点在 `b2_generate.rs` 的 `flash_b1_layer_phase_profile_json`）。按顶层墙读就永远不会重复计算。

**不能相加的东西**（这是 2026-09-16 之前被读错的地方）：

- `routed.diagnostics.read_ns` / `alloc_ns` 来自预取线程自己记的 `expert.read_ms` / `expert.alloc_ms`，**不在关键路径上**；`fixed.read_alloc_ns` 同理（后台每层 151 ms 读 137 MB）。把它们和 `setup_ns` 相加是重复计算。
- `shared.wall_ns` **不是 shared expert 的耗时**：那段计时从启动预取线程之前开始、到 `worker.join()` 之后结束（`layer_rows.rs` 的 `thread::scope`），量的是"shared 计算 与 routed 专家读盘 重叠之后的墙"，所以它会随读盘快慢抖。**2026-09-17 晚用 `COLIBRI_V4_FLASH_SHARED_PROBE=1` 把它拆开了**（`sharedprobe-shared-probe-a`）：默认帧里 `shared.wall = 16.64 ms/层`，其中**主机的 shared MLP 只有 `4.34 ms`**（关预取时量到它的纯净值是 `3.31 ms`）、最慢那个读盘 worker `16.20 ms`、MLP 返回后还要等 `12.30 ms`，**97% 的层是"等读盘"收尾**。所以 `shared` 那一行约四分之一是 shared MLP、四分之三是 routed 读盘——**读盘是 routed 那摊的，只是墙记在 shared 名下**；按 `shared` 当独立开销读会读错，且会让 `routed` 白白背上"没余量"的判词。
- `routed.diagnostics.setup_ns` 是 `ExpertSlot::new(...)` 的**主机墙上时间**（`layer_rows.rs` 约 1092/1142），每层 6 次。
- **`final_heads` 是层循环外面的独立数组**，`phase_profile.final_heads[].outer.wall_ns` 才是它整步的墙（稳态 `0.42–0.51 s/token`）。**任何"层内五项相加"的表都看不见它**——2026-09-17 之前它就被漏成 `~0.13 s`，那个数其实只是 `attention_share.py` 打的 `head_read`（盘读，页缓存命中后 `0.07 s`）。调整体预算时必须把它加上。
- **交付发射器把那一格叫 `fixed`，不叫 `layer_prepare`**：交付帧 `top_level` 的键是 `attention / fixed / routed / router / shared / wo_a_decode_gpu_sync`（`attention_share.py` 的 `PHASES` 对的是这一套）；`phase_profiles.rs` 的静止发射器才用 `layer_prepare`。层内那点 `unattributed_gap`（`0.52 ms/层`，实测 `0.588`）**就是 `wo_a_decode_gpu_sync`**，不是无名开销。
- **"8 步整条请求"要引用 `request_e2e_ns`，不是把层墙相加**（2026-09-22）：`sum(outer.wall_ns)` 比
  e2e 少约 2 s（层循环外的 `final head` + 启动）。而且**一条层记录可以是好几个位置的折叠**：真 32 token
  prompt 的 token 0 是 43 条记录、每条 `chunk_positions: 32`，量的是那一层跑 32 行的墙（≈ 单行层的
  32 倍），`stage` 各字段是 32 个单行 pass 的和（`routed.fanout` 是它们的"与"）。所以报"8 步请求"
  要同时写**哪种 prompt**和**哪个度量**：单 token 形状 `34.1 → 30.8 s`、真 32 token 形状
  `81.9 → 67.6 s`，差一倍多——混了就会得出"8 步请求 80 s"这种一眼就不对的数。

## 定版不常驻基线（HEAD `04bcf43`）：稳定态每 token 12.46 s

| 项 | 每层 | 每 token |
| --- | --- | --- |
| routed 专家 | 143.7 ms | 6.18 s |
| shared（重叠后的墙，见上） | 68.9 ms | 2.96 s |
| fixed 固定参数搬进显存 | 43.2 ms | 1.86 s |
| attention | 31.7 ms | 1.36 s |
| router | 2.3 ms | 0.10 s |

- **最大单项是 routed 槽位准备 `84.3 ms/层 = 3.62 s/token = 29%`**，而且在所有层所有 token 上几乎不动（79.7–88.5 ms）——固定开销的形状。`ExpertSlot::new` 每造一个槽位新建 8 块 pinned + 14 块设备缓冲 + 5 个 CUDA event，并在函数体里同步整条 stream 两次，每 token 调 258 次。`expert_slot.rs:50` 留着当初的注释"先一次一个专家，等实测到多专家流量再说"——那个实测现在有了，**还没改**。**那 14 ms 已经在 2026-09-16 拆开**：90% 是 8 块 pinned 主机缓冲的构造，开销跟字节走不跟次数；事件和设备分配加起来不到 0.3 ms，所以"把两次 stream 同步换掉"那条路是错的。细节与修法见 [[flash-0731-expert-slot-cost]]。
- **`fixed` 的墙已经拆完（2026-09-16 深夜插桩实测）**：43.2 ms = h2d 23.2 + **释放那 137 MB 主机缓冲 17.4**（`load_b2_raw_runtime_from_host_for` 末尾的 drop，在 H2D 计时器停止之后，主线程上）+ spawn 0.11，剩下的就是等预取线程（基线里≈0）。那 137 MB 每层都新建（`v4_state.rs` 的 `read_bf16`/`read_f32`/`read_u8` 是 `vec![0; n]`），一个 token churn 5.9 GB。槽位复用后 `fixed` 涨到 65 不是这条路的回归，是空闲从 244 ms 压到 128 ms、预取线程（147 ms）装不进去：实测 `join = max(0, worker 墙 − 上一层空闲)`，336 层偏差 sd 1.25 ms。记录在 `docs/experiments/2026-09-16-sm120-fixed-window-split.md`，探针在 `exp/fixed-window`。常驻显存拿掉的是那 1.86 s/token（= A/B 实测那 15%）。
- attention 的 31.7 ms 里 `compressor` 16.0 ms + `indexer` 5.1 ms 占大头，每个子步骤的 `sync_wall_ns ≈ gpu_event_ns`（主机每步都停下来等 GPU，一层十来个这样的小步骤）。2026-09-16 深夜按层型摊开：一个 token 的 attention 是 1368 ms（compressor 653 + indexer 215），Csa4 层一层 41–48 ms、Hca128 层 19–22 ms，且 compressor **不随上下文增长**——见 [[flash-0731-attention-compressor]]。

## 常驻主机负载之后（2026-09-16 深夜，`exp/fixed-host-reuse` 开启时）

一个 token 约 8.2–8.8 s，每层约 190 ms（同一时段关缓存的对照约 204 ms）：

| 项 | 每层 | 什么 |
| --- | --- | --- |
| `shared` | 72.5 ms | = routed 专家读盘 70.9 + 1.6（见上） |
| `routed` | 52.0 ms | 槽位重填 24.8 + kernel 10.8 + H2D 5.7 |
| `fixed` | 36.0 ms | H2D 23.5（不可省）+ 等预取 3–9；`release` 归零（不再每层还那 137 MB） |
| attention | 29.1 ms | 见 [[flash-0731-attention-compressor]] |
| router / wo_a / gap | 3.3 ms | |

- **进 SSD 的固定参数字节 `53.67 GB` → `6.71 GB`**（每 8 token），H2D 仍是 `53.67 GB`；峰值 RSS `0.73 → 7 GB`。
- **routed 专家每 token 要读 `3.45 GB`**（8 步 `27.59 GB`），实测读速只有约 1 GB/s（同字节在固定参数那条路也是 0.9–1.2 GB/s）——目前最大单项，但**裸盘上限还没量过**（下一步最便宜的实验）。
- `hash_layers = 3`：只有前 3 层的专家集由 token id 决定，其余 40 层依赖该 token 的 hidden，所以**专家集不能靠"把专家常驻"解决**。

## 交付跑（`slotreuse-8step.jsonl`，43 层 × 8 token，`67e80c3` 之后）的每层平均

| 项 | 每层平均 | 每层中位 | 8 步合计 |
| --- | --- | --- | --- |
| `shared`（= read + alloc + 1.7 ms 残差） | 46.1 ms | 45.9 ms | 15.86 s |
| `routed.wall`（setup 24.8 + kernel 10.8 + h2d 5.7） | 47.9 ms | 47.2 ms | 16.49 s |
| `fixed`（h2d 23.5；`fixed.read_alloc` 后台 146.5 仍不要相加） | 62.6 ms | 57.6 ms | 21.55 s |
| `attention` | 30.6 ms | 23.0 ms | 10.52 s |
| `outer` | 190.8 ms | 187.1 ms | 65.62 s |

`shared − (read + alloc) = 1.7 ms`，"`shared` 是盘的状态表"这条解释这次是**独立复算**出来的，不是沿用。同期的 `baseline-nonresident-8step.jsonl`（合并前、`103.38 s`）对照：read 35.0、alloc 29.8、`routed.wall` 143.6、`outer` 288.1 ms/层——读盘那部分两边几乎一样，差的全在槽位与固定参数上。

三处误读的完整更正写在仓库 `docs/experiments/2026-09-16-sm120-nonresident-baseline.md`（提交 `bba4cfd`、`ef6e669`），`交接说明-时间去向.md` 顶部加了指向它的更正行。

跑法与环境细节见 [[flash-0731-baseline-run-recipe]]；分析方法上的坑见 [[real-runs-over-proxy-verification]]。
