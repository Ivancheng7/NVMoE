---
name: hardware-instant-power-loss
description: 开发机（技嘉 B650M + 7500F + RTX 5060 Ti）会无预警硬件级瞬时断电，平均两三周一次，未提交工作和长任务有丢失风险
metadata:
  node_type: memory
  type: project
  originSessionId: sess_b11d5275-1619-4139-a9d7-1b0ffa2ec667
---

开发机是台式机：技嘉 B650M GAMING WIFI（BIOS F3，2025-07-16）+ 锐龙 7500F + 32G DDR5-6000(EXPO) + RTX 5060 Ti 16G，系统盘致态 TiPlus7100 2TB。

截至 2026-09-16，从 2025-12-31 起共发生 **22 次瞬时断电**（Kernel-Power 41 全部字段为空、无 BugcheckCode、无 WHEA、无 Minidump/MEMORY.DMP）。已排除：Windows 蓝屏、显卡驱动、硬盘、CPU/内存硬件报错。断电时刻一半集中在下午 13:00–15:59，另一块在 20:00–04:10。

**Why:** 这是硬件级断电而不是系统崩溃，发生前没有任何征兆，任何正在运行的长任务（模型推理、cargo 编译、WSL 里的 run）都会被拦腰砍断，未提交的改动有丢失风险。

**How to apply:** 跑长任务前先 commit/push；不要把一次完整 run 当成必然能跑完；汇报时如果发现结果异常中断，先怀疑是断电而不是代码问题。不要再把这类现象当成 Windows 崩溃去查日志。

相关：[[wsl-run-environment]]
