# Rust 侧模块划分（2026-09-16 拆分记录）

## 现状

`rust/colibri-cuda/src/main.rs` 原来是一个 45,540 行的单文件。已按「只搬文件、不动函数体」
的方式拆到 14,708 行，搬出的代码放在同目录的独立文件里：

| 文件 | 内容 | 行数 |
| --- | --- | --- |
| `tests.rs` / `tests_continued.rs` | 两个 `#[cfg(test)]` 模块（124 个测试） | 4,539 / 456 |
| `layer_stationary_prefill.rs` | `run_real_v4_layer_stationary_prefill`（原 3,361 行单函数） | 3,363 |
| `device_attention.rs` | device 侧 attention/state 行执行与请求 scratch | 3,078 |
| `layer_ops.rs` | 单层算子：mHC pre、RMSNorm、RoPE、compressor、sparse attention、router、shared expert | 2,848 |
| `generate.rs` | B2/B4 生成、Flash base/serve/dspark、stationary 基准入口 | 2,518 |
| `attention_gates.rs` | attention 门禁与审计入口：topk sweep、wavefront、device token-state 门禁、cache-off | 2,279 |
| `slice_checks.rs` | 各格式原语自检入口（`run_real_v4_*_slice_if_available`） | 1,328 |
| `mtp1_l0.rs` | mtp1 容量预检/草稿候选，L0 行与 compressor 检查 | 1,605 |
| `layer_rows.rs` | `run_real_layer_rows` 与 `run_real_layer_attention_rows` | 1,417 |
| `qd4.rs` | QD4 常驻缓存、routed 复放与 stationary 门禁 | 1,284 |
| `fixed_layer_cache.rs` | `FixedLayerCache` 结构体与两个 impl（residency、raw 快照、访问路径） | 1,264 |
| `final_head.rs` | Flash final-head 采集、候选门禁、GPU exact workspace | 1,145 |
| `v4_loopback.rs` | 双 worker loopback、副本恢复、恢复端点与认证客户端 | 1,105 |
| `full61_b1.rs` | `run_real_v4_full61_b1` 单步入口 | 991 |
| `expert_slot.rs` | `ExpertSlot` 结构体与其 impl | 752 |
| `artifacts.rs` | 结果文件/provenance 工具，含 `sha256_file` | 508 |
| `env_flags.rs` | 环境变量解析、校验、full61 preflight | 357 |

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
   一个文件可以装多个不相邻的块：`fixed_layer_cache.rs` 是「结构体 + 跨 1,000 行的两个
   impl」，`qd4.rs`、`attention_gates.rs` 都是把分散的入口函数收进同一个文件。
2. 新文件头加 `use super::*;`，内容原样保留（这些代码本来就在第 0 列，不需要重新缩进）。
3. 给搬出的顶层项、固有 impl 方法、结构体字段加 `pub(crate)`；
   trait impl 的方法和 enum 变体不能加（编译错误）。
   注意 `unsafe fn`、`const fn`、`extern "C" fn` 这些声明形式。
4. 在 main.rs 的声明块里加 `mod X;` 与 `use X::*;`（按字母序插进现有列表）。
5. 校验：除了 `rustc` 的 token 流比对（去掉空白与尾随逗号后应完全一致），
   还可以做更强的**字符级恒等式**——把搬走的块去掉文件头和注入的 `pub(crate)` 之后，
   加上新 main.rs 应等于原文件「逐字节」相等（`cut_batch*.py` 里的比对就是这条，
   它同时能抓住被误删的空行和逗号）。

## 怎么验证（本机环境）

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
- 编译警告数必须保持 59（bin）/ 22（test）。搬走的死代码会让警告「换文件」——例如
  `run_real_l0_rows` 那条从 `main.rs` 挪到 `mtp1_l0.rs`——所以判据是**警告集合去掉行号后
  逐条相同**，不只是总数相同。
- `cargo fmt --check` 在 HEAD 上本来就不通过，而且偏差多数是原码自带的：搬走的代码会把它们
  原样带走。2026-09-16 用 `rustfmt --edition 2021 --check` 逐文件量，整棵树 54 处
  （`v4_state.rs` 19、`tests.rs` 13、`layer_ops.rs` 4 等），**main.rs 反而是 0 处**——
  原来「main.rs 有两处想重排的位置」就在被搬走的代码里。所以判据是「不新增格式差异」：
  新文件里的每一处偏差都要能在搬运前的原文里找到。唯一已知的例外是 `pub(crate)` 注入——
  它会把个别接近 100 列的签名顶过线（当时 1 处，`fixed_layer_cache.rs` 的
  `validate_stationary_key`），不改语义，可以接受；要消掉就得动签名行，那会破坏
  「原文逐字节」这条更强的保证，不值得。
- 真机验证：跑一次 Flash base 单步推理，要求 `argmax=55222`、`hidden_digest` 含
  `5ed8bf88beae3cfe`，且结果 JSON 里全部 261 个 16 位十六进制 digest 字段与
  `/srv/localmoe/targets/0731-sm120-clean-cuda128-default-block32-r2.jsonl` 逐一相同
  （`layers`/`fp4_accum`/`final_head`/`peak_device_used_bytes` 也要相同）。
- `./run-identified.sh test --arch sm_120` 有 8 个失败是既有环境问题（4 个缺
  `rust/colibri-cuda/results/...` 的 cutpoint 产物、2 个缺 `COLIBRI_CUBLAS_EXACT_LIB`、
  2 个是 `v4_state.rs` 里 V4-Pro 时代的过期断言）。判据是失败集合与改动前一致，而不是全绿。

## 还没搬的

大块已经搬完了，剩下三类：

- **搬不动的**：`mod kernels`（约 2,850 行、52 个 kernel）和 `fn main()`（391 行）。
  想让 kernels 离开 main.rs 只能试 `include!` 把整个内联模块文本粘进来——**没验证过**，
  要做得先做 spike（搬完跑 cargo check + 一次真机推理对比 digest）。
- **还能按「整块搬一个函数」处理的中等块**（每个 110–330 行，合计约 1,900 行）：
  `run_real_v4_expert0_slice_if_available`、`run_v4_block_certificate_experiment`、
  `impl LayerDeviceWeights`、`pinned_h2d_service_gate`、`upload_pinned_layer_runtime_raw`、
  `stationary_layer_phase_profile_json`、`impl DeviceRequestScratchGeometry`、
  `run_real_v4_final_head_bench`、`run_streamed_lm_head_gpu_normalized`、
  `run_certified_argmax_p0`、`flash_b1_layer_phase_profile_json`、`fixed_prepare_dual_accounting`。
- **长尾**：剩下的约 6,000 行由几百个小于 110 行的项（结构体、类型别名、常量、小函数）构成。
  再往下压行数就得按**主题聚类**搬（一组相关的小项凑够一个文件），而不是「一个函数一刀」；
  每次聚类同样按上面的四条门槛验证。
