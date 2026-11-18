---
name: run-step-naming-convention
description: 顶层函数前缀规约——run_ = 可单独跑的程序，step_ = 模型执行里的一步；51 个函数已按此改名
metadata:
  node_type: memory
  type: project
  originSessionId: sess_906a0364-cd9c-4934-9515-35f286fd700a
---

2026-10-01 之前，`rust/colibri-cuda/src/` 里 140 个顶层函数全叫 `run_*`，但混着两种完全不同的东西：
**引擎步骤**（被执行路径调用，85 个）和**可单独跑的程序**（只被 `main.rs` / 测试调用，55 个）。
读代码时无法区分，AI 会话每次都要重新猜。

已按调用者分类，把 51 个引擎步骤改名为 `step_*`（`8536807`，基于旧底座；cherry-pick 到交付线 `6c442a7`）。
剩下 89 个 `run_*` 是程序。

**判定规则（新函数照这个走）**：看调用者，不看名字。只有 `main.rs` 或测试调它 → 程序，用 `run_*`；
被执行路径（`layer_rows`、`layer_ops`、`generate`、`device_attention`、`attention_device_chain`、
`qd4`、`layer_stationary_prefill` 等）调 → 步骤，用 `step_*`。
`validate_*` / `*_gate` / `*_preflight` / `*_audit` / `*_bench` / `*_sweep` **一律算程序那一侧**，
即使被执行路径调用——审计不是一步计算。

刻意保留 `run_*` 的四个例子：`run_generic_device_state_audit_mirror`、`run_serial_routed_replay`、
`run_sparse_attention_reference_state_device_into`、`run_targeted_attention_audit`。

规则正文写在仓库的 `rust/colibri-cuda/MODULE_LAYOUT.md`（「命名规约」一节）。

**验证改名有没有动数值的办法**（改名本身是机械变换，但这是能拿出来的证据）：
改名前后的提交各构建一次、各跑一次同一个单步推理，比 `argmax` + `hidden_digest` + 警告集合。
2026-10-01 实测两者逐位相同，警告集合逐条相同（62 条唯一）。

关联 [[main-rs-split-handoff]]（模块地图与拆分约定）、[[flash-0731-baseline-run-recipe]]（单步推理的钉死值）。
