---
name: delivery-binary-embeds-experiment-modes
description: 交付二进制里编着约 30 个实验模式、129 个开关；19 个模块交付文件零引用但混着共享 helper，所以清理必须按函数切不是按文件搬
metadata:
  node_type: memory
  type: project
  originSessionId: sess_a26b211a-15e2-4c1f-bb64-8b297534f5e0
---

2026-10-09 在 Windows worktree（分支 exp/dspark-draft-probe）实测的仓库形状，用来回答"要不要瘦身"：

- `rust/colibri-cuda/src` 61,605 行 / 60 个文件，**单一个 bin `colibri-cuda`**，Cargo.toml 没有 `[features]`、没有第二个 `[[bin]]`——所有实验入口都编进交付二进制。
- `main.rs` 只有 563 行，是一长串 `if <环境变量> { 跑模式; return; }`：30 个不同的 `COLIBRI_V4_*` 名字在这里做路由，全 crate 共出现 129 个不同的 `COLIBRI_V4_*`。
- **TODO/FIXME 在 rust/ 里是 0 条**。"临时"不是用标记写的，是用"环境开关 + 模式"写的；唯一一条生产待办是 `main.rs:236` 那句散文注释（`ponytail: ... when production dispatch lands`）。
- 19 个模块在交付文件（generate/layer_rows/layer_ops/device_attention/v4_serve/b2_*/fixed_*/final_head 等）里 `mod::` 引用数为 0，合计 **19,189 行（31%）**：qd4/qd4_gates/qd4_routed、attention_gates、state_audit、slice_checks、expert_slices、l0_bench、argmax_records、full61_b1、layer_stationary_prefill、fixed_layer_cache、v4_federated/v4_loopback/loopback_wire、stationary_lifecycle、host_provenance、mtp1_l0/mtp1_resources。

**为什么不能照这个清单整块删**：main.rs 用 `use X::*` 重新导入移出的模块（见 [[main-rs-split-handoff]]），交付代码可以用**短名**调到实验模块里的函数，`mod::` 引用检查会漏。堵洞复查（按短名 grep 交付文件）后确认这些是活的：
- `attention_gates::commit_layer_attention_state` → `layer_rows.rs:467`（层执行主路径）
- `slice_checks::load_layer_hash_ids` → `generate.rs:40,197`、`b2_runtime.rs:404,494`
- `stationary_lifecycle::{greedy_argmax_decision, b1_needs_final_head, greedy_forward_tokens}` → `generate.rs:811,896`
- `l0_bench::fnv1a_{u16,f32,bytes,i32}` → 交付面到处在用（哈希工具）
- `state_audit::check_compressed_boundary_bf16`、`qd4::maybe_run_routed_qd4_benchmark` → 被调用但内部走环境开关（默认不执行）

所以这是"纠缠"不是"分层"：想切得按函数切。

**其它实测**：`v4_state.rs` 7,851 行里 374/376 处断言在末尾的测试模块（`#[cfg(test)]` 在 5220 行），运行时只有 2 处——别按"运行时断言多"指责它，它的问题是 Pro 几何常量当相等判据（见 [[v4-state-hardcoded-geometry]]）。挂载点检查是活的：`v4_state.rs:4755 validate_model_dir_for` 拒 DrvFS/符号链接/非 pin 版本，经 `safetensors.rs`、`artifacts.rs` 进交付路径。

**How to apply**：要瘦身就 (1) 把 main.rs 的 30 条路由按 delivery/experiment 分栏，experiment 的移到 feature 或第二个 bin；(2) 老 FixedLayerCache(1,264) 与新 `fixed_host_cache.rs`(141)/`fixed_layer_slot.rs`(65) 并存，但 `fixed_host_cache.rs` 又引用 `fixed_prefetch`，删前先理链；(3) 纯入口模块里 full61_b1(993)+layer_stationary_prefill(3,375)+qd4_gates(331)+expert_slices(347)+argmax_records(732)+loopback 三件(2,895)≈ 8,700 行是候选。不要动断言、组合校验、切片证书（slice_checks 是数值逐位不变的判据来源）。
