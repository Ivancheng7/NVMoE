---
name: flash-0731-routed-read-ceiling
description: routed 专家读盘的天花板实测——盘不是瓶颈（六条并发流 3.79 GB/s vs 单线程 0.8–1.4），另一项独立开销是每次读进新申请缓冲；含分片/专家布局与证据位置
metadata:
  node_type: memory
  type: project
  originSessionId: sess_aece88d9-47da-4e15-bca2-80411a0d140a
---

2026-09-16 晚实测（**引擎一行没改、没跑引擎**，只量盘）。结论：**盘不是 1 GB/s 的天花板，瓶颈是并发深度。**

`dd` 单流 1 GiB（`/dev/sdd`）：buffered 冷 `1.5–1.6 GB/s`、页缓存热 `5.7–6.1`、`O_DIRECT` 冷 `2.7`。独立复核：六个并发 `dd`（六个分片，各 `bs=4M count=171 iflag=direct`）合 `3.31 GB/s`（单流 `2.56`）。

按引擎真实 I/O 形状回放（43 层 × 6 专家 × 6 张量 = `1548` 次 `pread`、`80216064` 字节/层、`3.449 GB/token`，与引擎的 `routed.source_bytes` 逐字节一致；每次冷跑前 `drop_caches`）：

| 读法 | 墙 | 吞吐 |
| --- | --- | --- |
| 1 线程，新申请缓冲（引擎现状），buffered | 4.1–4.4 s/token（87–95 ms/层） | 0.78–0.84 GB/s |
| 1 线程，缓冲复用，buffered | 2.5–3.4 s（58–80 ms/层） | 1.0–1.4 GB/s |
| 2 / 3 线程，buffered | 2.4–2.6 / 1.63–1.68 s | 1.3–1.4 / 2.06–2.11 |
| **6 线程，缓冲复用，buffered** | **0.91 s** | **3.79 GB/s** |
| 6 线程，`O_DIRECT` | 1.15–1.19 s | 2.9–3.0 GB/s |
| 12 线程 | 1.07–1.08 s | 无更多收益 |

两个可分开的杠杆，别混成一条：**并发深度（主项，约 3 倍）** + **每次读进新申请的零初始化缓冲**（buffered 单线程 `30 ms/层 ≈ 1.3 s/token`）。第二项的钱不在申请调用本身（`alloc_s` 全量 0.009–0.012 s，约 `6 µs`/次），在 `pread` 往全新页拷贝时的缺页/清零；`O_DIRECT` 下这项基本消失，但整体还是赢不了"buffered+复用+六线程"，**所以不建议换读模式**。

## 分片与专家布局（回放计划就是这么生成的）

- **一层一个分片**：`model-{layer+2:05d}-of-00048.safetensors`，43 层对 43 个分片一一对应（逐个核对过）。
- 片内先排 `256` 个专家的 `scale`（成块），再是权重块；权重块 **expert-major、每专家连续 12 MiB**（`w_stride = 12582912`，w1/w2/w3 各 4 MiB 相邻）。
- 每专家 6 次 `pread`，顺序与 `v4_expert.rs::load_with_prefix` 一致：`w1.weight, w1.scale, w2.weight, w2.scale, w3.weight, w3.scale`（3 × 4 MiB + 3 × 256 KiB，跨"权重区/scale 区"两次寻道）。
- 引擎侧是**单线程**预取：`layer_rows.rs` 的 `thread::scope` 里一个 worker 顺序 load 6 个专家，与主线程的 shared expert 重叠；开关条件是 `flash_b1_prefetch_enabled && qd4_lanes.is_none() && expert_slots.is_none()`（即 `COLIBRI_V4_FLASH_PREFETCH=1` 的 base 路径）。`slotreuse-8step.jsonl` 里 344 条层记录全是 `host_prefetched = true`。

## 对引擎数字的更正

交接说明里 `3.1 s/token`（`70.9 ms/层`）是**冷态悲观端**。交付跑八步平均是 read `35.9 ms/层`（`2.23 GB/s`）+ alloc `8.5 ms/层` ≈ `1.9 s/token`，token 间 `21–48 ms/层`（页缓存吃掉了重复路由的专家，但命中率没直接量过）。所以收益要分场景说：冷态约 `2 s/token`，按八步平均约 `0.9 s/token`。

## 证据与工具

- 原始数据：`/srv/localmoe/targets/io-ceiling/`（`dd-ceiling.txt`、`matrix.jsonl`、`matrix2.jsonl`、`matrix3.jsonl`、`plan-expert.tsv`、`plan-sequential.tsv`、`src/`）。工具源码另在 Windows 侧 `.zcode/io-ceiling/`（按惯例未跟踪、不入库）。
- 记录：`docs/experiments/2026-09-16-sm120-disk-ceiling.md`（提交 `29419ae`，同时把交接说明的 ① 收尾并更正了那个 `3.1 s/token`）。
- 工具：`disk_ceiling.c`（按计划回放 `pread`，读/申请分开计时；`reuse`/`fresh`/`fresh_mmap` × buffered/引擎式 `O_DIRECT`（对齐扩读+拷出窗口，抄 `read_direct_aligned`）× 线程数）、`gen_plan.py`（从分片头取真实偏移）、`layer_averages.py` / `token_breakdown.py`（读结果文件的分层/分 token 平均）。
- 复现要点：`sync; echo 3 > /proc/sys/vm/drop_caches` 后立刻跑（WSL 默认用户是 root）；`DIGEST_STRIDE=0` 可关校验，实测校验开销可忽略。

## 下一步（未做）

把六个专家的读并行起来（每专家一条 12 MiB 流）+ 缓冲复用；数值上是无操作（同样的字节，顺序不影响结果），验收照旧 = 四个钉死值 + 8 步 8 个 digest 逐个相同。**这台机器只有 6 个物理核**（Ryzen 5 7500F，12 线程），读线程会和主线程的 shared expert 抢核，3 线程和 6 线程都要量一档。按用户规矩，这类改动先上 `exp/*` 分支（[[experiment-branch-hygiene]]）。

相关：[[flash-0731-phase-profile-reading]]（`shared` = read + alloc + 1.7 ms 残差，这次独立复算过）、[[flash-0731-expert-slot-cost]]（槽位那 24.8 ms/层不在盘上）、[[wsl-run-environment]]、[[flash-0731-baseline-run-recipe]]。
