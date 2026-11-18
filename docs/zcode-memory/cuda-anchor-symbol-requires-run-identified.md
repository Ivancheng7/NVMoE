---
name: cuda-anchor-symbol-requires-run-identified
description: 直接 cargo build 会链接失败报 cuda_oxide_artifact_anchor undefined，必须走
  run-identified.sh
metadata:
  node_type: memory
  type: reference
  originSessionId: sess_05af1aca-e68c-4bc0-b4d1-2594310bf351
---

在 localmoe 里构建 Rust 二进制**必须**走 `rust/run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`。

直接用 `cargo build --release`（哪怕带了 `RUSTC_WRAPPER=rustc-human-wrapper.sh`）会链接失败：

```
rust-lld: error: undefined symbol: cuda_oxide_artifact_anchor_246e25db_colibri_cuda_0_0_0
>>> referenced by .../colibri_cuda::kernels::load
```

根因：`#[cuda_module]` 的 artifact anchor 由 cuda-oxide 后端（`rustc-codegen-cuda`）在编译期写一个 `.o`
并提供该符号；后端只在 `-Zcodegen-backend` 生效时才跑，而那个 flag 由 `run-identified.sh` 设上。
`cargo build` 不带它 → anchor 对象不生成 → `link()` 引用悬空。

**两个迷惑点**：rustc 只报 `linking with cc failed` 把 cc 的真实 stderr 吞了；要看到真错误得自己包一层
cc 记录 stderr（`-C linker=<脚本>`），或对已知能出二进制的树复现——**在干净树上也报同样的错**，
所以别怀疑自己的改动。这个坑 2026-09-18 花了近一小时，与 [[cargo-check-ice-wrapper]] 是两回事
（那个是前端 ICE，这个是后端没跑）。

相关：真实构建/跑推理规程见 skill `localmoe-wsl-run` 第 1 步。
