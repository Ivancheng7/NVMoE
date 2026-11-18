---
name: flash-0731-attention-compressor
description: Flash-0731 的 attention 每 token 1.0 s（新默认帧，占 27%）；构成已翻转——16 个小步
  0.59 s 比 compressor+indexer 0.42 s 还大；compressor 的权重主机往返在常驻默认下自己降了 3 倍
metadata:
  node_type: memory
  type: project
  originSessionId: sess_a7d58632-cd32-4ed2-a570-513bfe2bf1cc
---

**2026-09-17 晚：两档的根因找到了，池子也修好了（先读这条）**。attention `1.34 → 0.79 s/token`，
两档消失，数值逐位不变。三句话：

1. **两档的载体是"压缩器权重的搬运"**：`outputs=1024,ratio=4` 那次调用的九段里，只有
   `expand_wkv`+`expand_wgate`（主机上把 bf16 权重展开成 f32 的两个循环）在动——
   慢档 `5222+8063 us`、快档 `1189+1702 us`，差 `10.4 ms` 正好是这趟 attention 的差。
   `H2D`（`2247/2272/2307`）、`gemm`、`row_d2h`、`row_loop` 三跑全平。
2. **`7e07056` 那个复用池从来没生效**：池子用 `v4_state::HIDDEN`（`7168`，Pro 的几何）建，
   复用判据也拿这个常量比，而 Flash 喂进来的是 `4096` 宽。所以 `7e07056` 报的
   "两臂 `15.34` 对 `15.42`、池子拿不掉慢档"是**开 vs 开**——开关两边跑的是同一份代码。
   **教训：A/B 说"没差别"的时候，先证明那个开关真的改到了分支。**
3. `9efd662` 把宽度改成由调用方给（`spec.hidden` / full61 面 `v4_state::HIDDEN`），
   复用判据改成问池子自己。两批各 6 跑的交错 A/B：池子开 `0.784–0.810 s`（离散 ≤2.7%），
   关 `1.34`（快档 `0.95`）；`(1024,4)` 每次 `19.8 → 6.5 ms`。显存峰值 `-46 MB`
   （修之前是既付了池子的 `+153 MB` 又照旧每次申请）。
   记录 `docs/experiments/2026-09-17-sm120-shared-window-and-compressor-tier.md`，
   探针 `COLIBRI_V4_FLASH_COMPRESSOR_PROBE=1`（九段 + 分支判定），批脚本
   `rust/colibri-cuda/tools/run_arena_ab.sh`。

**没做的**：服务面没跑（同一条交付路径，应该一样，但没量）；"为什么同一段展开循环会两档"
只定位到段和"与分配有关"（池子复用那两块宿主缓冲后落到 `716/799 us`），**"首触缺页"仍是假设**。

**当前 attention 的构成（2026-09-17 晚，`arenaab-arena-fix-b`，token 7，`attention 0.794 s`）**：
18 个子步的 sync 墙加起来 `0.789 s`，对得上。**compressor `0.202`（41 层 × 4.91 ms）+ indexer `0.083`
（21 层 × 3.96 ms）只剩 36%；16 个小步合计 `0.504 s`（64%）。** 小步排头：
`wo_b 55.0` / `q_b_projection 53.5` / `commit_topk 52.5` / `mhc 49.2` / `q_a_projection 46.2` /
`kv_projection 44.4` / `kv_rope_quantize 37.7` / `wo_a 34.9` ms（都是 43 层的合计）。
每步都是"H2D + 小 kernel + D2H + 同步"，一层 18 个同步点。**所以 attention 里下一个该动的是这 16 个小步，
不是 compressor**（后者已经 4.9 ms/次）。脚本、批跑与相位排名见 [[flash-0731-per-token-floor]]。

**旧记录（已被上面取代，保留作历史）**：compressor+indexer 是整条链上唯一"纯主机内存"的活，
同一份代码同一天有 `7 ms` 和 `15.5 ms` 两档；当时已排除每次调用的分配、主机内存带宽、
CPU 频率、内存压力、别的会话、闲置恢复。当时"排除分配"是错的——那个开关是死的。

**2026-09-17 上午在新默认帧（常驻 8 GiB 开 + prompt chunk 开）重测**——`attn-split-pc-8step.jsonl`，
四件套与钉死值照旧复现（`argmax=55222`、`5ed8bf88beae3cfe`、峰值 `1500053504`）：

- attention 每 token `1.008 s`，占一个 `3.78 s` token 的 `26.6%`——和 fixed（1.01）、routed（0.96）
  并列第一，是唯一一直没人动的一项。占比一路涨：15%（交付帧）→ 24% → 27%。
- **构成翻转了**：compressor `0.288 s`（41 层，7.03 ms/次）+ indexer `0.131 s`（21 层，6.25 ms/次）
  合计 `0.42 s`（42%）；另外 16 个小步合计 `0.59 s`（58%），最大的是 mhc 1.64 ms/层、
  commit_topk 1.39、q_b 1.39、wo_b 1.33、q_a 1.18、kv_projection 1.17。
- **所以我 9-16 深夜给的那个"设备侧展开省 0.9 s"的估计作废**：在常驻默认下 compressor+indexer
  总共才 0.42 s，那个改动的上限就掉到 ≤0.4 s，而剩下的大头变成了 16 个小步的同步结构。
  改优先级之前先看这一条。

前一天（非常驻帧，`bench-telemetry-8step.jsonl`）的同一张表：attention `1.37–1.46 s`，
compressor `0.689 s`（Csa4 层 23.35 ms/次、Hca128 9.94 ms/次）、indexer `0.219 s`（10.43 ms/次）、
16 个小步 `0.55 s`。两次都是遥测开，差别只在配置——**常驻默认把 compressor 每次 23.35 ms 砍到
7.03 ms、indexer 10.43 砍到 6.25**，小步基本不动（甚至略升）。这就是 routed-read-threads 记录里
那条"常驻打开 attention 从 31 掉到 22 ms/层、没查原因"的数，现在有量了；
成因仍只是推测：compressor 那次调用的主机侧展开/拷贝原来在和固定流的每层 1.56 GB H2D
与锁页抖动抢主机带宽，常驻之后主机安静了，它就快了。

### 非常驻帧那次（2026-09-16 深夜）
`/srv/localmoe/targets/bench-telemetry-8step.jsonl`；脚本 `.zcode/attention_split.py` 是我加的）：

- **一个 token 的 attention 合计 1.37–1.46 s**：compressor `0.689 s`（41 层）、indexer `0.219 s`（21 层）、
  另外 16 个小步合计 `0.55 s`（每步每层 0.32–1.39 ms，一层 18 个子步骤）。
- 按层型（token 7 实测）：带 indexer 的 21 层 compressor 每次 `23.35 ms` + indexer `10.43 ms`；
  只有 compressor 的 20 层 `9.94 ms`；layer 0/1 是 window-only，读数 0。
- **它是固定开销，不随上下文增长**：KV 从 1 个 token 涨到 8 个，compressor 整 token 只从 `0.647` 涨到 `0.689 s`。

## 机制（2026-09-16 深夜读代码得出，未插桩验证）

交付配置里 `COLIBRI_V4_FLASH_BASE=1` 且 mhc/router 臂 = cublas，所以 `strict_compressor_enabled()`
为真，compressor/indexer 走"认证"的 cuBLAS 主机路径（`layer_ops.rs` 的
`run_layer_compressor` → `run_compressor_official_cublas_from_device` → `run_compressor_official_cublas_host`）。
那条路**每次调用**都做：D2H 把 wkv/wgate 拉回主机 → 主机上 bf16→f32 展开 → 再 H2D 上去，然后才跑
两个小 SGEMM。算字节（**Flash 的 `hidden` 是 `4096`，不是 `v4_state::HIDDEN` 那个 7168**）：

| 层型 | 每次调用的 PCIe 流量 | 池子开时实测 |
| --- | --- | --- |
| Csa4 compressor（outputs 1024） | D2H `8.4 MB` + H2D `16.8 MB` = `25.2 MB` | `d2h_w 1.6` + `h2d_w 2.9 ms`（≈11 GB/s，pageable 的量级） |
| Hca128 compressor（outputs 512） | `12.6 MB` | `0.9 + 1.5 ms` |
| indexer（outputs 256） | `6.3 MB` | `0.6 + 0.9 ms` |

每 token 合计约 `0.91 GB` PCIe（21×25.2 + 20×12.6 + 21×6.3）。
**旧的 `3.2 GB / 3.5 GB/s` 那张表是用 Pro 的 `7168` 算的，作废。**

**更正一条旧结论**：早先记的"`sync_wall_ns ≈ gpu_event_ns` 说明 compressor 是真 GPU 计算"不成立。
这一步是串行的（主机干活时 GPU 空转），两个 event 之间夹着主机的墙钟时间，所以两个数本来就相等，
分不出主机展开、拷贝、kernel 各占多少。

## 下一步（2026-09-17 插桩后定序：先复用缓冲，别动算子）

2026-09-17 在另一份树（`/srv/localmoe/build/attn-probe`）上插桩把压缩器认证路径拆成六段实测
（脚本 `.zcode/compressor_probe_patch.py` / `_patch2.py`，结果 `attn-probe-1step.jsonl`、
`attn-probe2-8step.jsonl`）：

- **SGEMM 段中位 `0.05–0.06 ms`，占单次调用 `0.2–0.3%`**（而且量的是入队；GPU 执行的上界是逐行段的
  `0.5 ms`）。**在 attention 这条路上做算子优化没有空间。**
- **主机每次重新申请缓冲才是大项**：同一规模的工作 21 次调用里在 `0.6 ms` 和 `13–17 ms` 之间双峰
  （`to_f32` 每次 collect 出两块全新 29 MB Vec、`to_host_vec` 同样），典型缺页指纹。
- **A/B（只把展开改成复用线程本地缓冲）：8 步热帧 `420.5 → 369.4 ms/token`（`-51 ms`，约 12%），
  1 步冷帧 `701 → 444 ms`；两者数值逐位不变**（`.zcode/outputs_match.py` 对交付参照 `IDENTICAL`）。
  引用要引热帧那 `-51 ms`；冷帧那个 `-257` 是"页还没碰过"的效应。
- 现成的 `StrictCompressorArena` 就是复用缓冲的实现（持有 `wkv_f32`/`wgate_f32`/设备缓冲/工作区），
  只是交付解码路径（`full61_b1.rs`）给 `run_layer_compressor` 传 `None`。**先接它，再谈设备侧展开**
  （设备侧展开的上限是拷贝那 `~230 ms/token`）。
- 没拆过的：`mhc`（`1.64 ms/层`、`70 ms/token`，设备侧 cuBLAS 臂、无主机往返却要 1.64 ms）。

下面这条是 9-16 的旧估计，保留但已被上面的排序取代：


权重本来就已经在显存里（`DeviceBuffer<u16>` bf16）。把 bf16→f32 展开挪到设备侧（一个很小的 kernel）
写进可复用的 f32 缓冲（按几何建一对 arena，约 103 MB，跨层复用），再调用**同样的** cuBLAS SGEMM、同样的
形状——bf16→f32 是精确移位，只有"f32 位是怎么进显存的"变了，所以数值逐位不变。
**收益重估**：非常驻帧估 `0.9 s/token` 是在 compressor+indexer = 0.91 s 的前提下；常驻默认下这两项只有
`0.42 s`，所以上限 ≤0.4 s——动手前先确认它还剩多少。验证门槛照旧：四个钉死值 + 8 个 digest +
`ordered_layer_digest_sha256` 链。

注意 `StrictCompressorScratch` **不是**这个修法：它只省分配，权重照样主机往返一遍。
**它已经接进交付解码路径**（`7e07056` 接的，`9efd662` 才真正生效——见开头那条），
现在解码路径每次调用就是 `6.5 / 3.4 / 2.0 ms`（Csa4 / Hca128 / indexer）。
**还开着的是"设备侧展开"**——把权重的主机往返整个省掉（那才是剩下那 `6.5 ms` 里的大头）。

## 占比随帧变化（汇总）

attention 绝对值在非常驻各跑里几乎不动（`1.31–1.61 s`），常驻默认下自己降到 `0.89–1.01 s`；占比一路上涨，
因为别的项掉得更快：交付帧 `15–17%` → 昨日常驻帧 `19–30%`（末四 token 平均 26%）→
2026-09-17 新默认帧 `22.7–27.1%`（末四 token 平均 `26.7%`，绝对值 `0.981 s`）。

读数字注意机器负载：2026-09-16 22:20 那次 slot-resident 跑是带着另一个会话的 colibri-cuda
（已跑 18 分钟）测的，磁盘读代理指标（最后一次 final head 的权重读）从 0.15 s 涨到 0.86 s，
那次 fixed 涨到 6.5 s/token 不能跟交付的 2.1 s 比；routed 掉到 1.09 是可信的（3.2 GB/s，接近六流上限）。

## 还没查的

- 常驻默认下 compressor 为什么自己快了 3 倍（23.35 → 7.03 ms/次）：有量了但只有推测
  （主机侧展开/拷贝原来在和固定流的每层 H2D 抢带宽）。要证实得在 compressor 调用内部插桩。
- 16 个小步（当前帧 `0.50 s/token`，占 attention 的 64%）：每步都是一次 H2D/小 kernel/D2H + 同步，
  一层 18 个同步点，`sync_wall ≈ gpu_event` 说明**每步都在等**——成本是延迟不是算力
  （最大那步 `wo_b` 也才 1.28 ms/层）。所以修法只有一句"别往返"。
- **⚠ 但交付路径上没有现成的开关可以解**（2026-09-17 晚查代码更正）：`run_device_attention_row`
  **只接在层静止 / full61 那条面上**（`layer_stationary_prefill.rs` 传 `device_state`），
  而交付的 CLI 解码路径 `run_b2_layer`（`b2_runtime.rs`）**第 14 个实参写死 `None`**，
  永远走 18 步主机路径。所以这是**移植，不是放行**——别把它读成"有个门槛挡着、推开门就行"
  （旧记录里"存在但被门槛挡着（只允许 2–5 步、≤9 行）"那句说的是 prefill 面的校验，不是解码面）。
  另外把中间结果留在显存里、只删同步点，是不写新 kernel 的那条退路，没人试过。

读法见 [[flash-0731-phase-profile-reading]]；预算里的位置见 [[flash-0731-expert-slot-cost]]。
