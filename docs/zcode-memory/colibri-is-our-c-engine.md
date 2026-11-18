---
name: colibri-is-our-c-engine
description: 仓库根 c/ 就是上游 JustVugg/colibri 引擎快照，DeepSeek-V4 路径是另写的 Rust；colibri 快的机制清单
metadata:
  node_type: memory
  type: project
  originSessionId: sess_3385957a-4ff3-4744-99f0-ef2c4f67a315
---

仓库根的 `c/` 目录**就是上游 JustVugg/colibri 引擎的一份快照**，不是我们自研的 C 参考实现：
`c/colibri.c` 是 GLM-5.2 引擎（注释意大利语），还有 `olmoe.c` 和共享头 `st.h`/`quant.h`/`tier.h`。
git 历史含上游提交（`Dual-SSD streaming: COLI_MODEL_MIRROR...`、`Merge pull request #506 from JustVugg/...`），
最后一次改动是 2026-07-22 `engine: keep the 107 KB GrDraft struct out of .data`。
**但 `c/deepseek_v4.c` 从没进过我们的树**（`git log --all -- c/deepseek_v4.c` 为空）——DeepSeek-V4-Flash
这条路是 `rust/colibri-cuda/` 里另写的一套。所以"colibri 怎么写的"可以直接读这一份，不用猜。

colibri 快的原因（其 docs 自述，非我们的量测）：decode 约 **75% 就是 routed 专家读盘**，整套设计围着少读/早读/边读边算：

- `PIPE=1` 异步读池（默认 8 worker，盘服务 −18%）、`PILOT=1` 路由前瞻预取（一层前可预测 71.6%）
- `O_DIRECT`（Strix Halo 上 +65%）、双 SSD 镜像（带宽 ×2）、一个专家三个矩阵盘上相邻、**一次 pread**
- 学到的热专家缓存 `.coli_usage` + pin 进 RAM/VRAM；dense 常驻 RAM（int4 ~9.9 GB）；MLA 压缩 KV（小 57×）

⚠ 别把它的头条速度当成同机对比：它 DeepSeek-V4 的 1.5–1.6 tok/s（≈0.64 s/token）是
**RTX 5080 + 两块真 NVMe + 32 GB RAM**；我们 2.305 s/token 是 **16GB 单卡 + 一块 WSL 虚拟盘 + 15 GiB**。
差里约一半是盘/内存环境（3.66 vs ~6 GB/s），另一半是可搬的技术：读盘重叠、前瞻预取、少搬字节、
attention 在 GPU 上。对照 [[flash-0731-bottleneck-map]] 与 [[disk-speed-two-tier]]。

我们**已经有**的对应物（别重复发明）：expert slot 跨层复用（`expert_slot.rs`）、固定权正常驻显存、
attention 设备链、固定参数预取线程。**还没有**：读盘池化重叠、跨层路由前瞻预取、单次 pread 布局。
