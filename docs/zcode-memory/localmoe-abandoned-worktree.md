---
name: localmoe-abandoned-worktree
description: Desktop\localmoe 主仓库工作区里有一坨从未提交的废弃工作（stage 阶梯 + manifest/oracle
  大教堂），钉错版本 60d8d707，交付分支一行都没要
metadata:
  node_type: memory
  type: project
  originSessionId: sess_2a7cea18-e5df-4782-9233-190bd287df89
---

`C:\Users\27236\Desktop\localmoe` 是**主仓库**（`localmoe-clean` 等是它的 worktree，共用 `.git`）。它的检出 `main` 停在 2026-07-29，**落后交付分支 `codex/business-restart` 135 个提交**，而且**不含上游 `origin/main` 的 `dcdee6f`**（`Implement exact Flash-0731 inference and serving`）——交付线是从 `dcdee6f` 长出来的，localmoe 停在它前面一步。

工作区里有一大坨**从未提交过**的废弃工作（2026-07-29 ~ 08-10）：

- `rust/colibri-cuda/src/main.rs` 脏版本 **48,926 行**（blob `913e0aa5…`，`git log --all --find-object` 搜遍全库任何提交都没有）；交付分支早已拆成 563 行 + 59 个模块。
- 9 个孤儿模块共 **15,451 行**：`flash_stage1..5`、`flash_manifest`、`flash_runtime`、`flash_variation`、`flash_hca128_l5p2_fixture`。这些路径**在任何提交里都不存在**，也没被 `.gitignore` 挡（只是从没 `git add`）。
- 制品：`rust/colibri-cuda/manifests/` 26 个 json（25 MB，含 3 版 full-runtime acceptance + shard/header inventory）、`fixtures/` 56 个文件（16 MB）、31 个 Python oracle 脚本共 16,448 行。
- 3 篇未跟踪实验文档（`docs/experiments/flash-stage5-*`、`flash-stage6-hca128-l5p2-*`）+ `docs/learning/` 课程。

**性质**：一个"逐 stage 冻结 oracle"的正确性验证阶梯——实现 stage N 的数学 → 把输出冻成 fixture → 证明 stage N+1 匹配。所以有 190 个测试、26 个 manifest、`FlashAcceptanceManifest`/`ShardManifest`/`HeaderInventory` 一堆概念。里面**确实有真实前向数学**（`flash_stage2_mhc_sinkhorn_bf16`、`flash_stage3_sparse_softmax_f32`、`inverse_rope`、gate logits、wo_a…），不是纯门禁；但外面裹了一层巨大的验收/证据外壳。

**为什么算无用功**：`AGENTS.md` 后来自纸黑字加了 "不加门禁、manifest、oracle 或实验框架"，正是冲着这坨写的。更要命的是它**钉死 `60d8d707…`**（另一份 checkpoint），整棵工作区搜不到交付目标 `9e165c30…`。交付分支验证过：**59 个模块里没有一个 stage/manifest/oracle/acceptance 文件**，概念全丢，只留了原仓库就有的 `make_v4_*_oracle.py` / `v4_full61_oracle.py` 等；31 个新 oracle 脚本**0 个**进交付。

**索引须知**：`index_repository` 直接指 `Desktop\localmoe` 会 abort（22,246 个文件 / 6.6 GB，被 `web/node_modules` 和 `rust/target` 撑爆）。要分析就把 `rust/colibri-cuda/src/*.rs` + `Cargo.toml` 拷到 scratch 树再索引（实测 5,341 nodes / 25,302 edges / 20 文件）。

相关：[[flash-0731-is-the-delivery-target]]、[[wsl-run-environment]]、[[main-rs-split-handoff]]
