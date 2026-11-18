---
name: flash-0731-final-head-parallel
description: final head 按输出 token 切 6 线程已实测 468→110 ms/token（省 358
  ms），与串行及交付原版逐位一致；代码在 exp/final-head-parallel（42d94e0），还没合并交付
metadata:
  node_type: memory
  type: project
  originSessionId: sess_90ec6a1e-35aa-41b3-8fd8-cca5fd8cefa3
---

2026-09-17 晚做完的实验（记录在仓库 `docs/experiments/2026-09-17-sm120-final-head-parallel.md`，
文档提交在交付分支 `0e732e4`）：

- **改动**：`v4_state.rs` 的 `final_checkpoint_head_cpu_timed` 把 505 个权重 chunk 按
  `COLIBRI_V4_FINAL_HEAD_THREADS`（默认 6）切成连续段，`thread::scope` 各自 `pread` 各自算，
  logits 各写各的私有段，join 后主线程拼回 + 原版同款升序严格 `>` 扫 argmax。
  **逐位一致是构造性的**（每个神经元自己的 fold，顺序一个字没变），且 t1 vs t6 vs
  交付分支原版串行（`baseline-nonresident-8step.jsonl`）三方 344 层链 + 全部 digest 一致。
- **数字**：稳态 final head 468→110 ms/token，8 步整请求 45.9→36.7 s；峰值显存不变
  （`1823014912`），`final_head_source_bytes` 不变（1.01 GiB/token）。t1 帧里 token 0
  853 ms 是页缓存冷，别算进改动收益。
- **状态：代码还没进交付分支**——按 [[experiment-branch-hygiene]] 留在
  `exp/final-head-parallel`（代码提交 `42d94e0`），合并前要 cherry-pick + 复跑基线钉死
  四值。合并后预期整 token 3.70→3.34 s。
- 没动的：`gpu_exact` 臂（显存原因，见 [[flash-0731-final-head-cost]]）、DSpark 臂用的
  `final_checkpoint_logits_from_normalized_cpu`（同款循环，将来上交付可照抄切法）。
- A/B 方法和"同一批内背靠背"的规矩见 [[optimize-the-optimization-loop]]；读盘与算已有
  重叠（两轮 read_ns 50–200 ms 抖），线程数 6 是拍的没扫。

相关：[[flash-0731-per-token-floor]]、[[flash-0731-phase-profile-reading]]
