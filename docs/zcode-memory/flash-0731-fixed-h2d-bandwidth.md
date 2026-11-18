---
name: flash-0731-fixed-h2d-bandwidth
description: 固定权重现在只剩"每层 H2D 137.4 MB"（1.01 s/token、占 27%），其中分配/释放占 414 ms；
  锁页只值 10%；老设备缓存开关已被主机常驻旁路，两条修法价码都量出来了
metadata:
  node_type: memory
  type: project
  originSessionId: sess_a615c8b1-894c-4181-84b2-ced4dcf9ab31
---

2026-09-17 实测（微基准脚本 `.zcode/h2d_probe.py`，ctypes 调 `libcudart`，**没跑引擎**；
引擎侧的相位读数来自 `attn-split-pc-8step.jsonl`）。新默认帧里 `fixed` = `1.01 s/token`（占 27%），
内容只剩"每层把 137.4 MB 固定权重推上显存"（`host_resident_hit`、零读盘、`h2d_ns=20.6ms`）。

| 搬法 | 每层 | 带宽 |
| --- | --- | --- |
| 普通内存 1×137.4 MB | 11.07 ms | 12.97 GB/s |
| 锁页 1×137.4 MB | 10.07 ms | 14.26 GB/s |
| 普通内存 20×6.8 MB（引擎的形状） | 11.60 ms | 12.39 GB/s |
| 锁页 20×6.8 MB | 10.06 ms | 14.28 GB/s |

- **锁页只值约 10%**——"锁页能让 H2D 快 2–3 倍"的假设不成立，别按它派活。
- **锁页本身 `1.083 ms/MB`**（137 MB = 147 ms，5.9 GB = 6.4 s），与 [[flash-0731-expert-slot-cost]]
  量到的 1.36 ms/MB 同量级、互相印证。小请求锁整模型是净亏。
- **缺口在这里**：引擎 `20.6 ms/层`（6.7 GB/s）vs 同样字节可达 `11.6 ms`（12.4 GB/s），
  差 `9–10 ms/层 ≈ 0.4 s/token`。最像的解释是 `DeviceBuffer::from_host` 每次调用都 `cuMemAlloc`
  且 `LayerDeviceWeights::new_layer` 每层建约 20 个张量 → 每层每 token 约 20 次设备内存分配/释放
  （微基准里缓冲是复用的，引擎里不是）。**未插桩验证。**
- 下一步两条：可复用设备槽位（+137 MB 显存，上限 0.4 s/token），或整模型常驻显存
  （+5.9 GB，省 0.89 s/token）。后者当年因"15% 换 6.56 GB"被否，现在同样的绝对收益是 3.78 s 的 23%，
  **值得用当前数字重新决定**。

## 引擎内插桩（2026-09-17 同日）

探针打在 `layer_loading.rs::upload_layer_runtime_for`（`.zcode/fixed_alloc_probe_patch.py`，
结果 `fixedalloc-probe-1step.stderr`）：在真实 `LayerDeviceWeights::new_layer` 之后按同样字节做 20 次
`cuMemAlloc_v2`/`cuMemFree_v2`。整 token（43 层）：`build = 847.8 ms`，其中**分配 259.5 + 释放 154.9
= 414.4 ms（49%）**、拷贝 `433.3 ms`（51%）。按层型：window-only 99 MB → 16.2 ms、Hca128 107 MB → 17.1、
Csa4 127 MB → 20.9（alloc 5.4–6.2、free 3.2–3.7）。

**老开关是死的**：`COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=8` 跑 8 步结果一行没变
（`devcache-8step.jsonl`：fixed 仍 0.98–1.03 s/token、h2d 仍 6708.9 MB/token、峰值显存同值，
逐层 cache_status 只有 `host_resident_hit`/`prefetched_host`/`miss`）。
原因：默认翻成主机常驻后，Flash 解码走 `generate.rs` 里新的 `fixed_host_cache` + `resolve_host_layer`，
**老 `FixedLayerCache::access` 根本没被调用**，"换个环境变量拿 0.85 s"这条捷径不存在。

**价码（量出来的）**：一层大小的可复用设备槽位（跨层 restage）= 拿回 `414 ms/token`、显存 +137 MB；
整模型常驻显存 = 拿回 `848 ms/token`、显存 +5.9 GB（峰值 1.5 → 7.4 GB），且必须把设备缓存接进新流程。
注意 restage 自身有成本：`DeviceBuffer::copy_from_host` 每次调用都 `stream.synchronize()`，
per-tensor 同步会吃掉一部分收益，得走异步接口。

## 槽位实测（2026-09-17 同日，探针树）

按 `ExpertSlot::new_or_restage` 的范式写了 `LayerDeviceWeights::restage_from_host`
（`.zcode/fixed_slot_patch.py`），槽位由 Flash B1 请求驱动持有。**结论：`fixed` 稳态 `1.02 → 0.71 s/token`
（`-0.31 s`，token 的 `-8%`），峰值显存 `+216 MB`，8 个 digest + 8 条层链 + argmax 全部逐位不变。**

- **坑：Csa4 与 Hca128 逐层交替，一个槽位每 token 只能复用一次**（第一版就是错的）。
  必须按层型分槽位（window-only / Hca128 / Csa4）或按几何建池。
- restage 自己 `13.6 ms/层`（≈10 GB/s），**非常驻路线上已接近地板**（机器可达 12.4 GB/s）；
  剩下那 `0.585 s/token` 只有整模型常驻显存能拿。
- **已落地（2026-09-17 晚，`7cdd0b1`）**：交付分支默认开、`COLIBRI_V4_FLASH_FIXED_SLOT=0` 关。
  交付帧复核：`fixed 1.02 → 0.73 s/token`、8 个 digest + 8 条层链 + argmax 逐位不变、峰值
  `1500053504 → 1716060160`；两个跑序里开着那臂都更快。见
  `docs/experiments/2026-09-17-sm120-fixed-layer-slot-delivery.md`。
- **一处更正**：先前说"账本要改成记真实搬运字节（现在会虚报 6.7 GB/token）"**不成立**——
  restage 拷的是**整张张量表**，`num_bytes()` 量的就是它，所以两臂的 `fixed_h2d_bytes` 相同、
  本来就没有虚报。真正只能靠墙看的只有 `fixed_h2d_ns`（restage 分支里它只剩 shard 打开那一步）。
- 探针证据：`/srv/localmoe/targets/slotkind-8step.jsonl`（+ 第一版单槽的错误样本 `slot-probe-8step.jsonl`）。

## 为什么仓库里"没有复用已有函数"

`ExpertSlot::new_or_restage`/`restage_weights`、`StrictCompressorArena` 各是一份**为单个调用点写的**复用范式；
`LayerDeviceWeights` 上**没有** restage（graph 确认只有 new_l0/new_layer/new_layer_with_wo_a）。
唯一一份通用的设备缓存实现（`FixedLayerCache`，有 `device_hit`）在默认翻转后被旁路。
根因是 `DeviceBuffer::from_host` 把"分配"和"拷贝"绑在一起，最顺手；要用复用就得显式回答
"这块显存归谁、活多久"——仓库里每个地方各自回答了一遍。这次是把第三个调用点接进同一个范式。

## 两条固定层缓存的分工（2026-09-17 查证，含一处更正）

**更正**：先前记的"老设备缓存开关被主机常驻旁路"说法不准——Flash 面**从来没有**接过老那份：
`git log -S "FixedLayerCache" -- generate.rs` 一条提交都没有；`generate.rs` 从 `29009b9`
（引入 `fixed_host_cache`）起就用新的那份，`598958e` 只是把"宿主常驻"变成默认。
在那之前 Flash 面没有固定层缓存（每层每 token 从分片重建宿主负载）。

| | `fixed_host_cache`（新） | `FixedLayerCache`（老） |
| --- | --- | --- |
| 服务面 | **Flash 面**：CLI 基线、serve、DSpark | full61 B1、常驻 prefill、attention 门禁 |
| 接入点 | `generate.rs:448` 建、`:573`/`:596` 取 | `full61_b1.rs:224`/`:290`；`layer_stationary_prefill.rs:248`/`:2389`；`attention_gates.rs:989`/`:1400` |
| 入口 | `main.rs:258 run_real_v4_flash_base` | `main.rs:389 run_real_v4_full61_b1`（`COLIBRI_V4_FULL61_B1`） |
| 开关 | `COLIBRI_V4_FIXED_HOST_RESIDENT_GIB`（默认 8 GiB，`=0` 关） | `COLIBRI_V4_DEVICE_FIXED_CACHE_GIB`、`COLIBRI_V4_HOST_FIXED_CACHE_GIB`、`COLIBRI_V4_STATIONARY_RAW_*` |
| 能力 | **只有宿主常驻**：下个 token 不读盘、不解码，但每层每 token 仍要重新 H2D `137.4 MB` | 有**设备常驻**（`device_hit` 整层不上传）与 raw 变体 |

后果：**"设备常驻"这半能力只写在老那份里**，而它绑着 full61/stationary 的假设（raw wo_a、
pinned 站台、`validate_stationary_*`）。要拿槽位剩下的那 `0.585 s/token`，等于把设备侧搬到 Flash 面，
或者按槽位范式在 Flash 面自己补一层——这次的槽位就是这层的第一半（去掉分配，保留 restage 拷贝）。
这也解释了为什么在 Flash 面设 `COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=8` 一行都不变。

基建：这台机器的 `gcc` 是 15，`nvcc` 解析不了它的头文件（`__is_pointer`、`cospi/rsqrt` 冲突），
`-allow-unsupported-compiler` 与 `-ccbin gcc-14` 都救不了——**小基准直接 ctypes 调 `libcudart`**。
