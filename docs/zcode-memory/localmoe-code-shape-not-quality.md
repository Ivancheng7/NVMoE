---
name: localmoe-code-shape-not-quality
description: localmoe-clean 引擎代码是「形状失控」而不是「水平低」——用户说它被落后 AI 污染时要用证据纠正，别顺着清理
metadata:
  node_type: memory
  type: project
  originSessionId: sess_a9e0716d-7d0a-462a-bfe0-b5dd06f16f66
---

2026-09-16 用户说仓库里的代码"被之前落后的 ai 污染过……不符合工程规范"。**这个归因不成立**，我核过：

- `main.rs:45-52` 的 `tilelang_sum4!` 宏用 `core::ptr::read_volatile` 阻止 LLVM 把浮点归约重结合，因为 TileLang 的四路归约是 `(0+2)+(1+3)` 而不是左折叠。
- `main.rs:58-60` 把 FP4 的 E2M1→E4M3 查找表压进两个 `u64` 立即数，注释写明目的是让它留在寄存器、不掉进 thread-local memory。
- 59,471 行里 **0 个 TODO / FIXME / HACK**，只有 1 处 `#[allow(dead_code)]`。

这种代码只有长期死磕数值一致性的人才写得出来。真正的问题是**形状**：`main.rs` 曾达 45,540 行 / 464 个自由函数；用 107 个不同的 `COLIBRI_*` 环境变量当接口、没有 CLI；单个函数最长 3,361 行（`run_real_v4_layer_stationary_prefill`）；函数名最长 61 字符（`stationary_mtp1_capacity_preflight_rejects_execution_flags`）。

**Why:** 顺着"这是烂代码"去清理，最容易误删上面那类只有在追逐位不一致时才写得出来的关键细节——而那种细节删掉之后是看不出来的，等下次数值对不上才会发现。用户的感受是真的（读起来确实难受），归因是错的。

**How to apply:** 用户再提"AI 污染"时，先把具体证据摆出来，区分"水平"和"形状"两件事，再谈怎么改。形状问题已经有专门的重构会话在按块搬运处理（见 [[concurrent-agent-sessions]]），不要另起一个跟它抢。

相关：[[docs-plain-register]]、[[concurrent-agent-sessions]]
