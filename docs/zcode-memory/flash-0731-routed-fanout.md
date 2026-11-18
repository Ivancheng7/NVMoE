---
name: flash-0731-routed-fanout
description: routed 六专家并行下发（默认关）实测 −17~19%；新的收尾是最慢那条 lane + 被顶起来的 MLP，诊断不能再相加
metadata:
  node_type: memory
  type: project
  originSessionId: sess_d5b114a4-bb7b-40e4-acb6-6ebfbd11808d
---

2026-09-21，交付分支 `codex/business-restart`，代码 `0d12b0f`，记录
`docs/experiments/2026-09-21-sm120-routed-fanout.md`。

- 开关 `COLIBRI_V4_FLASH_ROUTED_FANOUT=1`，**默认关**（交付面行为一个字没动）。六个路由槽位各一个
  线程、一条流、一套常驻槽位：每条 lane 自己读盘 → `upload_lane` → 在自己流上跑内核，六条互相重叠。
- 稳态：单 token prompt `2.240 → 1.805 s/token`、真 32 token prompt `1.913 → 1.581`（−17~19%）；
  8 步请求 `79.8 → 65.5 s`。两批 10 次 off/on 比对数值**逐位相同**；峰值 `8175288320 → 8242397184`（+64 MiB）。
- **新的收尾**：一层的墙 = 最慢那条 lane 的读→上传→内核（`routed.fanout_lane_max_ns`，真 prompt
  `19.5 ms/层`，其中读占 9~16）。共享 MLP 的墙也被顶起来（`shared.main 4.18 → 14.47 ms/层`）。
- **2026-09-23 消融量清（`/srv/localmoe/targets/a1-mlpabl/`，同二进制三臂交错）**：MLP 那 `+9.9 ms`
  ≈ **上传 4.5**（六条 lane 和 MLP 抢同一条 PCIe）**+ 内核 5.4**（抢 GPU）；lane 的**读**不背这个账
  （关的那臂在同一窗口也在读，MLP 仍是 4.46）。但 **MLP `14.38` < `lane_max 19.59`，它不卡人**——
  层的墙由最慢那条 lane 决定，**"先修 MLP"是错的排序**（那条链降到 14 ms 以下 MLP 才会变成绑定项）。
- 所以下一步的真把手是**把读从 lane 的关键路径上摘掉**：路由前瞻预取（colibri `PILOT=1` 量到猜中
  71.6%；我们只有固定权重的提前一层，自己也**没量过路由可预测性**）；其次是 lane 内 H2D 与内核
  双缓冲（`−2.4 ms/层` 量级）。见 [[colibri-is-our-c-engine]]。
- **诊断含义变了，禁止相加**：routed 的 `setup/h2d/kernel` 现在是六条重叠 lane 的墙相加
  （串行 `5.70` → 并行和值 `16.90 ms/层`），不再落在 `routed.wall` 里；routed 行新增 `fanout` 与
  `fanout_lane_max_ns`，嵌套计时守门条件对 fan-out 改成"最慢 lane ≤ 层墙"（原来的"和值 ≤ 墙"会误报，
  第一次跑就是这么崩的：`nested phase timing exceeds`）。
- **A2（专家身份缓存）叠在 A1 之上只再省 `0.07 s/token`**：三臂同批 `off 1.943` / `fanout 1.619` /
  `fanout + cache258 1.549`。缓存砍得掉其它五条 lane、砍不掉那条 miss 的，所以省的是六条并发读的
  竞争，不是读盘量（磁盘流量确实降了：真 prompt 每层 `6.0 → 1.0` 次载入、`3.21 → 0.86 GiB/token`）。
  A2 记录里"读盘挪出 shared 窗口后收益全额上墙"的预期**作废一半**，已在原地更正。
- **整条请求的数是两组，别混**（2026-09-22 用户当场抓到"8 步请求 79.8 s"这个错数）：单 token 形状
  8 步 e2e `34.11 → 30.76 s`（就是文档里一直引的 `33.9–36.8 s`），真 32 token prompt 因为每层要多跑
  32 行预填充，同样 8 步 e2e `81.87 → 67.57 s`、其中 token 0 `67.66 → 55.04 s`。写表时**必须标清
  哪种 prompt、哪个度量**：`79.8` 是层墙之和不是 e2e（真 e2e `83.7`），而且属于 32 token 形状。
- **缓存在 8 步之内是白干的**：整条请求 `fanout 68.64` 对 `fanout+cache 69.09 s`——稳态每 token 省的
  `0.07 s` 被第一个 token 的 `+1.2 s`（多锁页那一堆 lane、第一遍位置全 miss）抵掉。判缓存的收益要看
  更长的生成，8 步看不出来。
- **2026-10-01 "压 lane 争用"线已关闭**（`exp/lane-contention` `4b0c6dc`+`097d1d6`，记录
  `docs/experiments/2026-10-01-sm120-fanout-gpu-window.md`）：两笔账都到顶。①MLP 顶起那
  `0.26 s/token`（`main_ns 9.9` 对关臂 `3.9 ms/层`）**不卡墙**——层墙 = attention + router +
  max(MLP, lane 链)，MLP 藏在 lane 链 `24–27 ms/层` 下面，"看 main_ns 掉不掉"会假成功；
  ②给六条 lane 的上传+内核段加计数闸门 `COLIBRI_V4_FLASH_FANOUT_GPU_WINDOW=<K>`（默认不设），
  三臂交错 K=3 与不开逐项打平、K=2 略亏，数值逐位同。层的临界路径是**最慢 lane 的读（~20 ms）**，
  减并发/错峰/降优先级一族杠杆全部作用于不绑定的 GPU 小尾巴。还活的杠杆：把读挪出 lane 链
  （A2 缓存是已落地前置件、路由前瞻预取未做）。
- 两个默认开/关都还**待拍板**（见仓库 `现状.md` §五；A1 已于 2026-09-30 拍成默认开 `9799a0c`）。

相关：[[flash-0731-per-token-floor]]、[[flash-0731-routed-read-ceiling]]、[[flash-0731-expert-slot-cost]]、
[[flash-0731-bottleneck-map]]、[[resident-on-is-the-comparison-baseline]]
