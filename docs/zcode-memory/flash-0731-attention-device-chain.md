---
name: flash-0731-attention-device-chain
description: attention 那 18 个同步点已搬设备侧（exp/attention-device-chain），数值逐位不变，相位 19.54→13.34 ms/层
metadata:
  node_type: memory
  type: project
  originSessionId: sess_05af1aca-e68c-4bc0-b4d1-2594310bf351
---

2026-09-18：交付解码路径 attention 的 18 步从"每步 H2D+D2H+两次同步"改成设备侧串行入队
（中间结果留在显存）。代码在分支 `exp/attention-device-chain`（基于交付 `codex/business-restart`
的 `0e732e4`），**还没合并进交付分支**。开关 `COLIBRI_V4_ATTENTION_DEVICE_CHAIN`（默认开，`=0` 走旧路）。

实测（8 步真实 prompt，同二进制开关对比）：

- **数值逐位不变**：`generated_ids = [55222, 33832, 42549, 42549, 42549, 42549, 42549, 127384]`，
  8 个 token 的 `collapsed_bf16_digest` / `logits_f32_digest` / `normalized_bf16_digest` 两臂全等。
- **attention 相位 19.54 → 13.34 ms/层**（344 层-趟求和 `6.72 → 4.59 s`，带遥测）。
- 墙钟噪声大（三次配对 +6.28/+3.49/−1.57 s），**判据用相位账不要用墙钟**。

**不是**给设备 attention 解锁：`run_device_attention_row` 只接在层静止/full61 面，交付 CLI 的
`run_b2_layer` 永远走主机路径——所以现状页那句"设备 attention 存在但被门槛挡着"是错的，已改。
这是移植（换 `*_device_into` 版本），没写新 kernel。

**还没动**：compressor（5.20 ms/层）+ indexer（2.01 ms/层）仍是主机实现，设备链里还有一次 D2H
去喂它们——它们是 attention 里现在最大的一块。

记录：`docs/experiments/2026-09-18-sm120-attention-device-chain.md`。
相关：[[flash-0731-attention-compressor]]、[[flash-0731-phase-profile-reading]]、
[[experiment-branch-hygiene]]、[[cuda-anchor-symbol-requires-run-identified]]。
