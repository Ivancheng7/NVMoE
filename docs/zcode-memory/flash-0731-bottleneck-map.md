---
name: flash-0731-bottleneck-map
description: "用实测硬件上限标出的瓶颈地图：PCIe 14.27 GB/s 是结构下限（712 ms/token），哪些到顶哪些还能压"
metadata:
  node_type: memory
  type: project
  originSessionId: sess_e71e7b77-4ac9-47ac-8cf0-b0f4d3e91485
---

2026-09-18 现场量了三个硬件上限，给每个稳态相位标出"到顶还是有余量"：

| 资源 | 实测 | 工具 |
| --- | --- | --- |
| PCIe H2D（锁页→设备，Gen4 **x8**） | **14.27 GB/s** | `.zcode/io-ceiling/h2d_rate6g.py` + `h2d_probe.py` 两次独立一致 |
| 盘（routed 读，6 线程） | 3.79 GB/s | 2026-09-16 disk-ceiling 隔离回放 |
| 主机内存展开 | 9.4 GB/s | `bandwidth_probe.py` |

**结构性下限**：每 token 必须过 PCIe 的是 `fixed` 6.71 GB + `routed` 3.45 GB =
**10.16 GB → 712 ms/token**（@14.27）。稳态 e2e 2847 ms/token，**传输占 25% 且不可能更快**。

**到顶（别在这抠）：** routed 读盘（3.45 GB/token，已贴盘）、routed H2D（14.0 GB/s=链路）、
**shared 的 MLP 计算**（探针显示 shared 窗口 15.3 ms ≈ worker_max 14.9，而 mlp 只 4.0——
**MLP 被读线程藏住了**，优化算力拿不到收益，除非 routed 读先变快）。

**还能压（杠杆从大到小）：**
1. **fixed 6.71 GB/token 的 H2D → 设备常驻**：峰值显存 3.96 GB / 卡 16 GB，装得下 5.6 GB
   fixed 模型。做成后 PCIe 总量 10.16→3.45 GB/token，理论上限 712→242 ms/token。
2. **routed 跨 token 复用专家**：相邻 token 路由高度重叠，"少读"而非"读快"。
3. **compressor/indexer 计算搬设备侧**（attention 18 步已搬完；这两个是那次明说"没动"的）。
4. **fixed 相位那 ~6 ms 余量**：先查清下面的记账 bug。

## ⚠ fixed.h2d_ns 曾经不可信（2026-09-20 已查清并修好）

`host_resident_hit` 层报 156 MB / 5.44 ms = **28.7 GB/s，是链路实测 14.27 的两倍**，
物理上不可能。**根因不是字节算两遍，是漏了一段计时**：固定层设备槽位的 restage（把整层
固定权重拷进复用设备缓冲）发生在 loader 自己的 H2D 计时器开始之前，所以那一格只记到那笔
很小的 raw `wo_a` 上传，restage 的时间落到了 `window.residual_ns` 里。

**2026-09-20 修好（交付分支 `f5b991b`）**：在 `generate.rs::load_flash_b1_layer_from_host`
里把 `take_restaged` + loader 一起夹进一个计时器，用跨度覆盖 `fixed.h2d_ns`。实测同一
二进制、同一字节：槽位开从 `24.12`（超链路）落到 `8.67 GB/s`，`window.residual` 从 `4.05 s`
回到 `1 ms`，数值逐位不变。记录
`docs/experiments/2026-09-20-sm120-fixed-h2d-restage-accounting.md`。
**所以 2026-09-20 之前所有帧里槽位开着的 `fixed.h2d_ns` 都偏高**，引用要打折。

**Why:** 用户问"哪些到瓶颈、哪些能继续压"。**不能靠感觉说"到顶了"**，必须拿实测上限
和相位账对齐——这次一量就发现引擎自报的数超了物理链路 2 倍，差点据此下错结论。

**How to apply:** 报"到顶"必须给出该资源的实测上限和用量；报"能压"要给杠杆排序。
[[flash-0731-fixed-resident-vram]] 已把"fixed 显存常驻"这条最大杠杆做完（稳态 −0.72 s/token）。
相关 [[flash-0731-steady-state-vs-first-token]]、[[flash-0731-per-token-floor]]、
[[flash-0731-routed-read-ceiling]]、[[flash-0731-fixed-h2d-bandwidth]]。
