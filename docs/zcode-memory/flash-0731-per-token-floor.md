---
name: flash-0731-per-token-floor
description: 一个 token 的账——交付线 2026-10-01 实测稳态 1.998 s/token（单 token 形状、三次均值）；下面是各历史形状，别当现状
metadata:
  node_type: memory
  type: project
  originSessionId: localmoe-clean-a1f0d1dadf0989bf
---

**2026-10-01 实测（要看这个）**：交付线 `0554419`，单 token prompt 形状（`TOKEN_IDS=0`、8 步、
每次丢页缓存、默认命令——fan-out 已默认开），三次稳态 **`2.037 / 1.971 / 1.987`，均值 `1.998 s/token`，
离散 `0.028`**（比历史那批的 `0.18–0.27` 紧很多）。

- 首 token `14.2–14.9 s`；稳态相位（s/token）：`routed 0.60–0.64`、`attention 0.57`、
  `shared 0.44–0.47`、`router 0.11`、`head 0.10`、**`fixed 0.000`**
- 峰值显存 `8242397184`（8.24 GB，整模型常驻已开）
- 与交付线自己记的 09-30 默认面值（`1.986 / 1.977`）吻合在 `0.02` 以内 → 这一批可对照
- **`fixed` 稳态是 0**：整模型常驻显存把这一项（09-17 表里的 `0.72 s`）直接抹平了
- 当天盘速 **`3.42 GB/s`**（两次复量 `3.4254 / 3.4243`）：低于记录的快档带 `3.54–3.75`，
  远高于慢档 `2.69–2.80`。不是作废档，但记住这个偏差；WSL 虚拟机当天反复重启（`uptime` 每次都 `0 min`）
- 结果文件在 `/srv/localmoe/targets/delivery-floor-2026-10-01/`（`step8-r1..r3.jsonl`）

**一个缺口：`真 32 token prompt` 那组的 token ids 找不到。** 仓库和 `.zcode/` 里只有
`TOKEN_IDS=0` 和 5-token 那组（`58572,42549,42549,109865,42549`）。也就是说 `1.581` / `1.943`
这些被反复引用的数字**输入不可复现**——要再引用得先把那 32 个 id 找回来钉进记录。

**2026-09-21 现状（历史）**：真 32 token prompt 稳态 `1.943 s/token`（交付默认，常驻显存开）、
开 `COLIBRI_V4_FLASH_ROUTED_FANOUT=1` 后 `1.581`；单 token prompt 形态 `2.240 → 1.805`。
整模型常驻显存也让 `fixed` 归零。**下面这张 09-17 的表只剩历史价值**，别拿它当现状；
现在排第一的还是专家路径（routed），收尾方式见 [[flash-0731-routed-fanout]]。

**2026-09-18 更新（三件改动当天已合进交付分支 `codex/business-restart`，见
[[flash-0731-three-arms-merged]]）：**
上面这张表是 2026-09-17 的形状。之后三项已变且**都已进交付树**：`restage` 的口子已补（读盘直接进锁页，
`routed.alloc` 归零）；`attention` 的 16 个小步已搬设备侧（16.99→13.33 ms/层）；
`final head` 已并行（468→110 ms/token）。
**"地板 1.6–1.8 s/token"这句也作废**：现在稳态 2.85 s/token，而结构性下限是
PCIe 的 712 ms/token（见 [[flash-0731-bottleneck-map]]）。读这张表要看日期，别当现状。

**2026-09-17 晚第三次重记**（`arenaab-arena-fix-b` 池子开那一臂，稳态后 4 token）。
**整 token `3.70 s` = 层循环 `3.25` + **层循环外** `final head 0.45`。**
排名按真实归属（不是按相位表那五行）：

| # | 项 | 每 token | 占 | 里面 |
| --- | --- | --- | --- | --- |
| 1 | **专家路径** = `routed 0.97` + 它触发的读盘 `0.43` | **1.40 s** | 38% | 读盘 0.43（贴盘上限）、restage 0.44、kernel 0.46、H2D 0.25 藏在 kernel 底下 |
| 2 | `attention` | 0.79 s | 21% | compressor 0.20 + indexer 0.08 + **16 个小步 0.50** |
| 3 | `fixed` | 0.72 s | 19% | 约 0.59 是 restage 的拷贝 |
| 4 | **`final head`（层循环外）** | **0.45 s** | 12% | 盘读只有 0.07，其余 0.35 是 CPU 流式 |
| 5 | shared MLP 自己 | 0.18 s | 5% | 4.2 ms/层 |
| 6 | `router` | 0.11 s | 3% | |
| 7 | `wo_a_decode_gpu_sync` | 0.02 s | 1% | 这是层内那点未归账 gap，0.52 ms/层 |

**⚠ `final head` 以前被记成 `~0.13 s`，那是错的**（把 `attention_share.py` 打的 `head_read`——只有盘读那
`0.07`——当成了整步）。它又是唯一在层循环外面的大项，所以任何"层内五项相加"的表都看不见它。
整步的墙在 `phase_profile.final_heads[].outer.wall_ns`，实测稳态 `0.42–0.51 s`。详见
[[flash-0731-phase-profile-reading]]。

- **`routed 0.97` 的"贴盘上限"判词作废。** 旧的 `3.45 GB ÷ 3.79 GB/s = 0.91 s` 是把读盘算在它头上得出的，
  而读盘发生在 `shared` 窗口里。合成第 1 项之后，**真正贴顶的是读盘 0.43（别碰）和 H2D 0.25（别碰）**；
  中间那 **0.44 s 的 restage 是纯多出来的**：每 token 同样的 3.3 GB 被搬两遍（先读进普通宿主缓冲，
  再拷进锁页暂存区）。这是这一项里的口子。
- **`fixed 0.72`**：槽位已落地（`7cdd0b1`）。剩下约 0.59 是 restage 自己的拷贝；再往下只有整模型常驻显存
  （`+5.9 GB`）。锁页只值 10%，别试。
- **`attention 0.79`**：SGEMM 只占 0.2–0.3%，算子优化没空间；compressor 已经 4.9 ms/次。
  **该动的是那 16 个小步（0.50 s，比 compressor+indexer 加起来还大）**，见 [[flash-0731-attention-compressor]]。
- **`final head 0.45` 里那 0.38 s CPU 流式**（1.01 GiB/token，单线程标量约 3 GFLOP/s）**是剩下最便宜的一块**：
  归约是每个神经元自己的 `fold`，神经元互相独立，按输出神经元并行**逐位相同是构造上成立的**，
  不需要靠测去证明。别去切 `gpu_exact` 臂（3 步峰值 3.72 GB）。详见 [[flash-0731-final-head-cost]]。
- **`shared 0.62` 已拆**：四分之一是 shared MLP，四分之三是 routed 读盘（已归到第 1 项）。

**整请求的账**（8 步）：进程墙 `43.3 s` = 请求 `40.24` + 启动 `3.1`；请求 = 层循环 `35.31`
+ `final head 4.05` + 其余 `0.87`。**token 0 一个人就占 `11.25 s`**（付冷页缓存里 6.7 GB 固定权重的读）。
**服务面同批实测 `3.94 s/字` 对命令行 `4.04 s/token`——一样快**（`servevscli-svc-1`），
以前那个 `4.81` 只是提交旧了。服务仍不写相位账。见 [[flash-0731-serve-path-speed]]。

**服务面现在有默认关的相位落盘口了**（2026-09-20，`fd4a864`），见 [[flash-0731-serve-phase-sink]]。

这个形状的地板约 `1.6–1.8 s/token`：每 token 必须从盘上搬 3.45 GB 专家字节，这是设计本身。
能改体验的仍然是三件结构性的：输入上限 `prompt + 生成 − 1 ≤ 128`、整模型常驻显存、以及 DSpark
（唯一"少跑几趟"的路，[[flash-0731-dspark-arm-state]]）。**整模型常驻显存 2026-09-20 已做完**：
`e75d804`，稳态 `3070.6 → 2353.9 ms/token`（−0.72 s / −23%）、峰值 +6.35 GB、数值逐位不变，
默认关、待拍板。见 [[flash-0731-fixed-resident-vram]]。

相关：[[flash-0731-routed-read-ceiling]]、[[flash-0731-fixed-h2d-bandwidth]]、[[flash-0731-attention-compressor]]、[[flash-0731-phase-profile-reading]]
