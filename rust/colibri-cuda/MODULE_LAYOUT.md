# Rust 侧模块划分（2026-09-16 拆分记录）

## 现状

`rust/colibri-cuda/src/main.rs` 原来是一个 45,540 行的单文件。已按「只搬文件、不动函数体」
的方式拆到 25,381 行，搬出的代码放在同目录的独立文件里：

| 文件 | 内容 | 行数 |
| --- | --- | --- |
| `tests.rs` / `tests_continued.rs` | 两个 `#[cfg(test)]` 模块（124 个测试） | 4,539 / 456 |
| `layer_stationary_prefill.rs` | `run_real_v4_layer_stationary_prefill`（原 3,361 行单函数） | 3,363 |
| `artifacts.rs` | 结果文件/provenance 工具，含 `sha256_file` | 508 |
| `final_head.rs` | Flash final-head 采集、候选门禁、GPU exact workspace | 1,145 |
| `env_flags.rs` | 环境变量解析、校验、full61 preflight | 357 |
| `slice_checks.rs` | 各格式原语自检入口（`run_real_v4_*_slice_if_available`） | 1,328 |
| `generate.rs` | B2/B4 生成、Flash base/serve/dspark、stationary 基准入口 | 2,518 |
| `layer_ops.rs` | 单层算子：mHC pre、RMSNorm、RoPE、compressor、sparse attention、router、shared expert | 2,848 |
| `device_attention.rs` | device 侧 attention/state 行执行与请求 scratch | 3,078 |

## 三条必须遵守的约定

1. **`mod kernels` 必须留在 main.rs 内联。** cuda-oxide 的 `#[cuda_module]` 明确拒绝
   `mod foo;` 这种独立文件模块（报错 `cuda_module requires an inline module so kernel
   signatures are visible`），设备代码搬不走。

2. **新模块的 `mod` 声明要写在 `tilelang_sum4` 宏之后。** `macro_rules!` 是文本作用域，
   声明在宏之前的模块看不到它。所以 main.rs 里专门有一段声明块，位于宏定义下方。

3. **搬出的模块用 `use X::*;` 重新导入，调用点保持原样。** 这些簇有上百个调用点，逐个改成
   `X::foo(...)` 会把改动撒进各个函数体；改为在 main.rs 加一行 glob 之后，调用点与重构前
   逐字节相同。代价是搬出的项需要 `pub(crate)`（否则父模块看不见子模块的私有项）。
   唯一例外是 `layer_stationary_prefill`：只有一处调用，glob 会触发 unused 警告，所以由
   glob 提供非限定名，调用点维持原文。

## 再拆一批的做法

1. 用 `python3 -c` 或脚本按行号整块切出，**切口必须落在顶层项的边界上，且不能把属性行
   （`#[derive]`/`#[allow]`）与其项拆开**——这是唯一踩过的坑，拆开会导致整段解析失败。
2. 新文件头加 `use super::*;`，内容原样保留（这些代码本来就在第 0 列，不需要重新缩进）。
3. 给搬出的顶层项、固有 impl 方法、结构体字段加 `pub(crate)`；
   trait impl 的方法和 enum 变体不能加（编译错误）。
   注意 `unsafe fn`、`const fn`、`extern "C" fn` 这些声明形式。
4. 在 main.rs 的声明块里加 `mod X;` 与 `use X::*;`。
5. 校验：`rustc` 的 token 流比对（去掉空白与尾随逗号后应完全一致）能证明搬运没有改语义。

## 验证门槛（本机环境）

```bash
cd <源码目录>/rust          # 例如 /srv/localmoe/build/localmoe-clean-git/rust
RUSTC_WRAPPER=$PWD/rustc-human-wrapper.sh \
CARGO_TARGET_DIR=/srv/localmoe/targets/<新目录> \
cargo check --locked -p colibri-cuda            # 要再加 --tests 才算覆盖测试模块
# 走 run-identified.sh 时它会自己 export RUSTC_WRAPPER：
./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda
```

- 直接跑 `cargo check` 会在钉住的 nightly 上触发编译器 ICE，**未改动的原始树同样如此**；
  必须套 `rustc-human-wrapper.sh`，它把 `--error-format=json` 换成 `short`，绕开那条渲染路径。
- 编译警告数必须保持 59（bin）/ 22（test），这是搬运未改变语义的旁证。
- `cargo fmt --check` 在 HEAD 上本来就不通过（main.rs 有两处 rustfmt 想重排的位置），
  所以门槛是「不新增格式差异」：新文件保持 fmt-clean，main.rs 的可重排位置集合不变。
- 真机验证：跑一次 Flash base 单步推理，要求 `argmax=55222`、`hidden_digest` 含
  `5ed8bf88beae3cfe`，且全部 257 个 digest 字段与基线一致。
- `./run-identified.sh test --arch sm_120` 有 8 个失败是既有环境问题（4 个缺
  `rust/colibri-cuda/results/...` 的 cutpoint 产物、2 个缺 `COLIBRI_CUBLAS_EXACT_LIB`、
  2 个是 `v4_state.rs` 里 V4-Pro 时代的过期断言）。判据是失败集合与改动前一致，而不是全绿。

## 还没搬的

main.rs 里仍有约 2.5 万行，最大的几块：`run_real_v4_full61_b1`（990 行）、
`impl FixedLayerCache`（976+253 行）、`run_real_layer_rows`（791 行）、
`struct ExpertSlot`（751 行）、`prepare_generic_device_attention_row`（681 行）、
`run_real_layer_attention_rows`（623 行）、`run_v4_b2_generate_arm`（451 行）、
`impl DeviceRequestScratch`（434 行）、`run_real_v4_b4_layer_stationary`（409 行），
以及 31,000 行以后的一批 `run_mtp1_*`／`run_real_l0_*` 入口。按上面的做法继续即可，
每批搬完按同一套门槛验证。
