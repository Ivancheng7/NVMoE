---
name: flash-0731-fixed-resident-vram
description: 整模型固定权重常驻显存已默认开（b620ef7）：稳态 −0.88 s/token（−28%）、峰值 1.83→8.18 GB、数值逐位不变；单步钉死峰值随之改
metadata:
  node_type: memory
  type: project
  originSessionId: localmoe-clean-a1f0d1dadf0989bf
---

2026-09-20 把"整模型固定权重常驻显存"落地并**拍成默认开**
（`COLIBRI_V4_FIXED_RESIDENT_GIB`，`e75d804` + `8e162a6` + 默认 `b620ef7`）。
这是 [[flash-0731-bottleneck-map]] 里那个"最大杠杆"，也是 2026-09-16
`exp/fixed-residency` 那次实验（15.4%）的重做。

**默认开只覆盖两个交付面**（`FLASH_BASE` / `FLASH_SERVE`，`FLASH_0731` 且单请求），
和 prompt chunk / 固定层槽位 / 压缩器池同款 gate（`b2_runtime.rs::flash_b1_resident_enabled`）；
`=0` 回到流式。预算默认 `10 GiB`，**按空闲显存减 2 GiB headroom 夹一次**
（`flash_b1_resident_budget_bytes` + `preflight_cache_budget`），小卡只少常驻几层、不会 OOM。

**单步钉死峰值随默认开而变：`1825112064 → 8175288320`**（那 `+6.35 GB` 就是常驻的权重）；
`=0` 时仍是 `1825112064`。别的钉死值一个没动，8 步 digest / 层链 / argmax 逐位不变。
验收（不设任何常驻变量）：`defaulton/` 三跑，1 步四值复现、8 步 `301 layer-visits served from VRAM`、
与显式常驻 IDENTICAL、与流式只差峰值。

**实测（同一二进制只切这一个开关，8 步真实 token，每跑前 drop_caches；4 轮交错批 = 两臂各 4 跑、
每对换先后。数据在 `residentfill/`，是消掉 token0 开销之后的）：**

| | 不常驻 | 常驻 |
| --- | --- | --- |
| 整请求 e2e（n=4 均值） | `39.48 s` | **`32.37 s`（−7.11 s / −18.0%）** |
| 稳态每 token（第 1–7，含层循环外 final head） | `3.183 s` | **`2.305 s`（−0.879 s / −27.6%）** |
| 第一个 token | `16.39 s` | `15.57 s`（**−0.82 s，不再变慢**） |
| 峰值显存 | `1825 MB` | **`8175 MB`（+6.35 GB）** |
| `fixed` H2D（8 步） | `53.67 GB` | `6.71 GB`（−87%） |

**8 跑两档不重叠**：e2e 不常驻 `38.19–40.76`、常驻 `30.49–34.00`。8 个 `hidden_bf16_digest` /
8 条层链 / 8 个 `argmax` / `generated_ids` **8 跑逐项全部相同**。
stderr：`43 layers resident (6708864200 bytes), 301 layer-visits served from VRAM`。
结果文件 `/srv/localmoe/targets/residentfill/r{1..4}-{stream,resident}.jsonl`。

**⚠ e2e 的百分比会随机器漂，别用它判这个改动**：同一改动两批 e2e 报过 `−10.0%`（`residentbatch/`）
和 `−18.0%`（`residentfill/`），差别全在 token 0 的绝对值（`15.65` 对 `16.39 s`）被摊进 8 步的比例。
**判据一律用稳态 `−0.879 s/token`**（两批一致到 0.008 s）。

**⚠ 第一个 token 那 +2.4 s 曾经是我自己引入的**：第一版为了让常驻生效把预取整个关了
（`flash_b1_prefetch = !resident_enabled && ...`），于是 token 0 那 43 层全是 `cache_status=miss`、
每层 156 MB 在主线程同步读，相位账里 `window.worker_ns` 从 `14.9 s` 掉到 `0`。
`8e162a6` 改成**预取开、只在下一层已常驻时跳过 spawn**，token 0 就回到噪声内
（预取窗口回到 `13.8 s`）。教训：**改动引入的"结构性开销"要拿相位账对照，别当成本质代价**。

**为什么这次比 2026-09-16 那次大（15%→28%）**：层循环本身在这之后快了很多（attention 设备链、
final head 并行、routed 读锁页），固定流那 6.71 GB 从"被预取重叠掉大半"变成稳态里一块明显的
东西（常驻时 `fixed` 恰好 `0.0 ms`，不常驻约 `0.72 s`）。

**实现要点（`generate.rs` 的 b2 贪心循环）**：按层捕获 `B2RawRuntime`、命中时
`fixed_source/h2d_bytes` 记 0；**常驻开着时关掉固定层设备槽位**（常驻项自己就是那个槽位，两个
机制同时开会把同一份权重交给两个所有者），**但预取保持开**（只在下一层已常驻时跳过 spawn，
`8e162a6`）；**服务完必须把层放回槽位**，否则第一次命中就清空槽位而字节仍被计数（2026-09-16
那个 bug，`ba2b2a8`）。常驻的是源字节 6.71 GB（FP8 wo_a 保持原样、用时解码进共享 scratch），
不是展开后的 8.55 GB。

**2026-10-02 补：服务面曾经每请求重建常驻池，已修**（`exp/lane-contention` `2a49dbb`）：常驻池
和 routed scratch 都是 `run_v4_b2_generate_arm` 的局部变量，而服务面每请求调一次 arm——相位账里
每个请求都背 6.25 GiB `fixed` 读账 + 1.2–2.1 s 墙。提升到服务循环外后，同一引擎第 2 请求起
每请求省 4.6–5.2 s e2e（TTFT 19→14 s 档）、`fixed` 归零、id 串与 CLI 钉死值全过。捕获预算只数
本次新捕获；服务循环外建一次、逐请求复用是唯一行为改动。**2026-10-02 已合并交付**
（`codex/business-restart` 合并提交 `9dc963a`；合并闸门 `merge-gate-9dc963a/`：二进制 sha 与
A/B 的 new 臂一致、钉死值复现、服务冒烟 3 请求 17.7→14.8→14.1 s）。记录
`docs/experiments/2026-10-02-sm120-serve-resident-persistence.md`。

**没做/没声称**：~~没测服务面~~（2026-10-02 已测并修掉上面那坑）；服务面**并发**请求的
显存预算没算（16 GB 卡单请求峰值 `8.18 GB`，只剩约 8 GB 空闲；常驻池现在跨请求占住不释放）。
记录 `docs/experiments/2026-09-20-sm120-fixed-resident-vram.md`，复现 `.zcode/resident_fill_batch.sh`。

相关 [[flash-0731-bottleneck-map]]、[[flash-0731-per-token-floor]]、
[[resident-on-is-the-comparison-baseline]]、[[flash-0731-steady-state-vs-first-token]]。
