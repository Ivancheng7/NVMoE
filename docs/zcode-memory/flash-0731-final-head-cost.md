---
name: flash-0731-final-head-cost
description: final head 是 0.45 s/token（第 4 大项、在层循环外），其中 0.38 s 是单线程标量 CPU 流式；按输出神经元并行逐位不变，是剩下最便宜的一块
metadata:
  node_type: memory
  type: project
  originSessionId: localmoe-clean-a1f0d1dadf0989bf
---

**2026-09-17 晚查清。** 默认臂是 `final_head = canonical_cpu_streamed_bf16`
（`COLIBRI_V4_FLASH_FINAL_HEAD_ARM` 不设即 CPU）。实测稳态 **`0.42–0.51 s/token`**，
占一个 `3.70 s` token 的 **12%**——**第四大项，不是零头**。

- **墙在 `phase_profile.final_heads[].outer.wall_ns`**，它在**层循环外面**。旧记录里的 `~0.13 s` 是错的：
  那是 `attention_share.py` 打的 `head_read`，**只有盘读**。稳态分解：盘读 `0.07 s`（1.01 GiB 走页缓存，
  约 13 GB/s），**其余 `0.38 s` 是 CPU**。所以任何"层内五项相加"的表都看不见它，这也解释了
  为什么它一直没进优化清单。读法见 [[flash-0731-phase-profile-reading]]。
- **那 0.38 s 是什么**：`v4_state::final_checkpoint_head_cpu_timed` 里一个单线程标量循环——
  505 个 chunk（`CHUNK_ROWS = 256`）× 每个 256 个神经元 × 4096 次 `f32` 顺序累加，
  外加每个元素一次 bf16 拆包（`u16::from_le_bytes`）。129280 神经元 × 4096 = 529M MAC，
  实测约 **3 GFLOP/s**，就是标量单线程的量级。
- **它天然可并行而且不用靠测证明逐位相同**：`canonical_final_head_row_dot` 是
  `fold(0.0, |sum, (i, w)| sum + bf16(i) * bf16(w))`——**每个神经元自己的加法顺序是固定的，
  神经元之间完全独立**。按输出神经元（`token_offset`）切块，每个神经元的归约顺序一个字都不变，
  逐位相同是构造上成立的。这台机器 6 个物理核，估计那 0.38 s 掉到 0.10–0.15，
  **省 0.25–0.3 s/token（7–8%）**。
- **别去切 `COLIBRI_V4_FLASH_FINAL_HEAD_ARM=gpu_exact`**。那个臂是实现了的、也确实"exact"，
  但 2026-08-28 量过 **3 步峰值显存 3.72 GB**（`docs/experiments/2026-08-28-sm120-flash-0731-adaptation.md`），
  默认臂之所以是 CPU 就是因为它。**在 CPU 这条路里按行并行**才是便宜的解。
- 顺带：`gpu_exact` 也是 2026-08-28 那条"3 步峰值 3.72 GB"的旧印象的来源，那不是每步泄漏——
  默认 CPU final head 下 1 步和 8 步的峰值一模一样（见 `2026-09-16-sm120-multitoken-and-serve.md`）。

在 token 预算里的位置与排名见 [[flash-0731-per-token-floor]]；`attention_share.py` 的 `head_read`
不等于这一步的墙，见 [[flash-0731-phase-profile-reading]]。
