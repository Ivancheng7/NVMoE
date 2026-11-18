---
name: disk-speed-two-tier
description: 本机盘速在 2.7 / 3.65 GB/s 两档之间晃约 25%，是宿主机侧竞争，客户机查不到；跑性能数字前先量一次
metadata:
  node_type: memory
  type: project
  originSessionId: sess_8de0637a-69e5-49ad-ae80-3b165c228c38
---

`/dev/sdd`（WSL 虚拟盘，底层是 Windows 上的 VHDX）的冷读速率会在两档之间晃：

- **快档 `3.54–3.75 GB/s`**（20 次冷跑 mean `3.658`、stdev `0.067`）——正常
- **慢档 `2.69–2.80 GB/s`**——晃约 25%

**根因最可能是宿主机（Windows）对同一块物理设备的 I/O 竞争**，客户机完全看不见。实测对照：
在同一块盘上多加一个独立读者，探针立刻从 `3.62` 掉到 `2.46`（`−32%`），幅度吻合。旁证：
WSL VM 会在会话之间反复重启（`boot_id` 变、`uptime` 常是 "up 2 min"、`dmesg` 里 sdd 反复
mount/unmount），宿主机本身没重启。

**已排除**：plan 形状、客户机内存压力、`read_ahead_kb`、`O_DIRECT`、客户机进程抢盘。

**Why:** 2026-09-20 我因此写错了一篇记录——把一次 `2.75` 的量测当成"旧记录 `3.79` 是虚的"
的证据，断言是 plan 的重复区域造成的。其实两份 plan 都是 `dup 1.000x`，几小时后原样命令
重跑 `3.79` 完整复现。真凶是环境窗口，不是数据。

**How to apply:** 跑性能/S 级数字前先跑一次
`disk_ceiling /srv/localmoe/targets/io-ceiling/plan-expert.tsv reuse 0 6`，
`wall_gbps` 落在 `2.7` 档就判本批作废、换时间重跑，**不要**拿它跟历史数字比、更不要据此
修改已有结论。规程已写进 skill `localmoe-wsl-run` 的 0.3 节和 `localmoe-experiment-record`
的硬规则 3。相关：[[flash-0731-routed-read-ceiling]]（那份的 3.79 是有效的，
但 `plan-real-routed.tsv` 的 `2.215x` 只是探针输入，不代表引擎行为——引擎稳态读聚合
`3.9–4.4 GB/s`，贴着冷盘）、[[flash-0731-bottleneck-map]]。
