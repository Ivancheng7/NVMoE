---
name: cargo-check-ice-wrapper
description: localmoe-clean 里普通 cargo check 会在钉住的 nightly 上 ICE（原始树也一样），必须用
  rust/rustc-human-wrapper.sh 才能当编译门槛
metadata:
  node_type: memory
  type: project
  originSessionId: sess_24caf8ff-f2e0-48f8-bebb-91b720903e59
---

在 `rust/` 下直接跑 `cargo check --locked -p colibri-cuda` 会让钉住的
nightly-2026-04-03 **编译器恐慌**（`mir_borrowck` 的 ICE，rustc 退出码 101，
提示 "the compiler unexpectedly panicked"）。

这个 ICE 与代码改动无关：2026-09-16 我把 `mod tests` 搬到独立文件时复现了它，
随后用**未改动的原始 main.rs**（45540 行）在干净目标目录里重跑，同样 ICE。
两边加上 wrapper 都通过，且都是 59 个警告。所以它是环境/工具链问题。

正确的编译门槛：

```bash
cd /srv/localmoe/build/localmoe-clean-cuda128/rust
RUSTC_WRAPPER=$PWD/rustc-human-wrapper.sh \
CARGO_TARGET_DIR=/srv/localmoe/targets/<新目录> \
cargo check --locked -p colibri-cuda
```

wrapper 的作用是把手里的 `--error-format=json` 换成 `short`，绕开「渲染诊断时崩溃」
这条路径。它早已在仓库里（`13ceb59` 加入），但 CI、文档、Cargo 配置都没有引用它。

2026-09-16 补上两处（`af52d35`、`b5104b4`）：

- `run-identified.sh` 现在**自己设** `RUSTC_WRAPPER`（绝对路径），所以走脚本的构建/运行
  不用再手动 export；直接敲 `cargo oxide` 或 `cargo check` 绕开脚本时仍需自己设。
- `rustc-human-wrapper.sh` 在 git 里曾是 `100644`（无执行位），clone 到 ext4 后权限真实
  生效，于是每个新 clone 都得手动 `chmod +x` 一次。已改成 `100755`，不再需要。

两个坑：

- `RUSTC_WRAPPER` 必须写成**绝对路径**（或至少含 `/`）。cargo 是按**当前工作目录**解析
  wrapper 的，纯文件名会被当成命令名直接执行失败（`could not execute process`）。
- 排查这个 ICE 时**已经证伪**两个假设，别再花时间：`main.rs` 里那个唯一的非 ASCII 字符
  （doc 注释里的 `×`）不是诱因，CRLF 行尾也不是。用钉死的 nightly 做最小复现（多字节
  字符、高亮跨越多字节字符、CRLF）三种情况都不崩。根因是 cuda-oxide 后端吐出的一条
  坐标颠倒的 span（起点 17、终点 15）撞上 nightly 的 `annotate_snippets` 渲染器没有兜底。

同一门禁下另一件事：**`cargo fmt --check` 在 HEAD 上本来就不通过**，main.rs 有
两处（重构前是 3474 行和 37281 行）rustfmt 想重排的多行条件。所以「fmt 必须干净」
不能当门槛；可行的替代是「不引入新的 fmt 差异」：新文件保持 fmt-clean，main.rs
的可重排位置集合与改动前一致。CI（`rust-preflight.yml` 只做 fmt + check + --version）
因此在本机环境里是跑不通的。

编译门槛过了并不等于搬运正确；测试模块被 `cfg(test)` 挡住，要另外用
`cargo check --locked -p colibri-cuda --tests` 或真机 `./run-identified.sh test --arch sm_120`。

**2026-10-13 补充（动删除前必读）**：`--tests` 那次编译曾经**长期是坏的**。`93423d8`
删 DSpark 臂时，把 `v4_state.rs` 测试模块里两处引用（`DSPARK_STAGES`、
`DsparkB5PersistentState`、`flash_dspark_b5_topk`…）留在了原地：`cargo check --tests`
26 个 error、191 个测试整个跑不了，连带没人再看见 bin 那 57 条警告。已在
`cleanup/conservative`（`216c243`）修掉；警告数随之变成 **54（bin）/ 16（test）**。

判「谁没人用」**必须带 `--tests`**：bin 构建报 57 条 `never used`，带 `--tests` 只剩 19 条，
差的那 38 个不是死代码，是被 191 个测试当 oracle 用的 CPU 参考实现（`route_layer_cpu`、
`indexer_scores_cpu`、`csa_compress_topk_cpu`、`compressor_initial_matches`、`run_cpu`…）。
照 bin 那份清单删，测试构建立刻碎。另外 `kernels_module.rs:10` 的
`cuda_oxide_device_extern_*` 是 cuda-oxide 后端生成的，永远别删。细节见
[[delivery-binary-embeds-experiment-modes]]。

相关：[[wsl-run-environment]]、[[flash-0731-baseline-run-recipe]]
