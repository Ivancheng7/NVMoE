# Rust 侧模块划分（2026-09-16 拆分完成记录）

> 表里的行数是 2026-09-16 的快照，之后一直在漂；名字和职责是准的，行数只当量级看。

## 命名规约（2026-10-01）

顶层函数的前缀区分两件**不同**的事，不要再混：

- **`run_*` = 一个可以单独跑的程序。** 它自己解析输入、跑完打印或写结果文件，入口挂在
  `main.rs` 的 if 链上（受 `COLIBRI_V4_*` 开关控制）。
- **`step_*` = 模型执行里的一步。** 它被执行路径调用，不自己解析参数。

2026-10-01 之前两者都叫 `run_*`（140 个），引擎步骤和实验程序在名字上无法区分，
读代码只能靠猜。改名只动标识符，不动行为。

判断一个新函数该用哪边，**看调用者，不看名字**：只有 `main.rs` 或测试调它 → 程序；
被执行路径（`layer_rows.rs`、`layer_ops.rs`、`generate.rs`、`device_attention.rs`、
`attention_device_chain.rs`、`qd4.rs`、`layer_stationary_prefill.rs` 等）调它 → 步骤。
`validate_*`、`*_gate`、`*_preflight`、`*_audit`、`*_bench`、`*_sweep` 一律算程序那一侧，
**即使它被执行路径调用**（审计不是一步计算）。

## 现状

`rust/colibri-cuda/src/main.rs` 原来是一个 45,540 行的单文件，现在 **559 行**：只剩 crate 属性、
顶层 `use`、LUT 常量、两个 `const fn`、`tilelang_sum4` 宏、40 行模块声明块、`include!` 一行和
`fn main()`（391 行）。业务代码全部搬进同目录的独立文件（下表按行数降序）：

| 文件 | 内容 | 行数 |
| --- | --- | --- |
| `tests.rs` / `tests_continued.rs` | 两个 `#[cfg(test)]` 模块（124 个测试） | 4,539 / 456 |
| `layer_stationary_prefill.rs` | `run_real_v4_layer_stationary_prefill`（原 3,361 行单函数） | 3,363 |
| `device_attention.rs` | device 侧 attention/state 行执行与请求 scratch | 3,078 |
| `kernels_module.rs` | 设备侧 `#[cuda_module] mod kernels`（由 main.rs `include!` 粘回） | 2,853 |
| `layer_ops.rs` | 单层算子：mHC pre、RMSNorm、RoPE、compressor、sparse attention、router、shared expert | 2,848 |
| `generate.rs` | B2/B4 生成、Flash base/serve/dspark、stationary 基准入口 | 2,518 |
| `attention_gates.rs` | attention 门禁与审计入口：topk sweep、wavefront、device token-state 门禁、cache-off | 2,279 |
| `mtp1_l0.rs` | mtp1 容量预检/草稿候选，L0 行与 compressor 检查 | 1,605 |
| `layer_rows.rs` | `step_real_layer_rows` 与 `step_real_layer_attention_rows` | 1,417 |
| `slice_checks.rs` | 各格式原语自检入口（`run_real_v4_*_slice_if_available`） | 1,328 |
| `qd4.rs` | QD4 常驻缓存、routed 复放与 stationary 门禁 | 1,284 |
| `fixed_layer_cache.rs` | `FixedLayerCache` 结构体与两个 impl（residency、raw 快照、访问路径） | 1,264 |
| `fixed_layer_planning.rs` | 预算解析、promotion 计划、层选择、capacity plan、bounded readahead | 1,154 |
| `final_head.rs` | Flash final-head 采集、候选门禁、GPU exact workspace | 1,145 |
| `v4_loopback.rs` | 双 worker loopback、副本恢复、恢复端点与认证客户端 | 1,105 |
| `full61_b1.rs` | `run_real_v4_full61_b1` 单步入口 | 991 |
| `state_audit.rs` | 持久状态/attention 审计、逐级 bit 对比与诊断输出 | 908 |
| `loopback_wire.rs` | loopback/recovery 的帧、样本、重放与 wire 统计 | 778 |
| `expert_slot.rs` | `ExpertSlot` 结构体与其 impl | 752 |
| `argmax_records.rs` | final-head 基准、certified argmax 与 id-only 记录 | 732 |
| `device_geometry.rs` | device 侧字节台账常量、scratch 几何与 device state 结构体 | 655 |
| `b2_generate.rs` | B2 生成请求解析、parity stop、B2 physical 与 JSON 记录 | 640 |
| `mtp1_resources.rs` | MTP1 草稿字节常量、容量探针与 scratch 预检 | 606 |
| `flash_arms.rs` | mHC/router/FP4 各 arm 的解析、cuBLAS 会话与 provenance | 593 |
| `artifacts.rs` | 结果文件/provenance 工具，含 `sha256_file` | 508 |
| `fixed_prefetch.rs` | 固定权重的预取任务、租约与 promotion residency | 447 |
| `run_records.rs` | 运行记录结构体、critical-path 时间线与 telemetry | 425 |
| `stationary_lifecycle.rs` | 常驻生命周期记账、贪婪决策与三个静态检查 | 418 |
| `layer_loading.rs` | 层运行时装载与上传（含 pinned raw 上传） | 414 |
| `phase_profiles.rs` | 双路记账与 stationary 各相位 profile JSON | 402 |
| `device_weights.rs` | `LayerDeviceWeights` 与其 impl、draft 权重上传、dspark 阶段装载 | 401 |
| `l0_bench.rs` | L0 摘要与 buffer 统计、三个 FP8/KV benchmark | 395 |
| `launch_wrappers.rs` | `launch_v4_*` kernel 启动包装与 expert 误差断言 | 363 |
| `env_flags.rs` | 环境变量解析、校验、full61 preflight | 357 |
| `expert_slices.rs` | expert-0 slice 与 block certificate 实验入口 | 347 |
| `b2_runtime.rs` | B2 raw runtime 装载、单层执行与双 token 前缀准备 | 338 |
| `qd4_gates.rs` | QD4 lane 预留校验与 stationary/immutable-cache 门禁 | 329 |
| `mtp1_draft.rs` | MTP1 草稿状态推进、对齐与 QD4 lane 借用 | 309 |
| `dspark.rs` | device attention 行执行与 Flash DSpark 各阶段 | 289 |
| `qd4_routed.rs` | routed 复放/时间线结构体与序列化 | 225 |
| `host_provenance.rs` | 内存快照、挂载来源与 cache-off 基准 | 202 |

（`v4_chat.rs`、`v4_expert.rs`、`v4_federated.rs`、`v4_state.rs`、`safetensors.rs`、
`cublas_exact.rs` 等 `v4_*` 文件在这次拆分之前就存在，没有动过。）

## 四条必须遵守的约定

1. **设备模块用 `include!` 粘回 main.rs。** cuda-oxide 的 `#[cuda_module]` 只接受内联模块
   （报错 `cuda_module requires an inline module so kernel signatures are visible`），
   但 `include!("kernels_module.rs");` 里的 token 在属性宏展开前就已经展开成内联模块，
   所以 2,853 行的设备代码可以放在独立文件里。2026-09-16 实测：编译 0 error、59 条警告，
   真机 261 个 digest 与基线全同。被包含的文件**不能有 `use super::*;` 文件头**，它是被粘贴的
   片段，不是模块。

2. **新模块的 `mod` 声明要写在 `tilelang_sum4` 宏之后。** `macro_rules!` 是文本作用域，
   声明在宏之前的模块看不到它。所以 main.rs 里专门有一段声明块，位于宏定义下方。

3. **搬出的模块用 `use X::*;` 重新导入，调用点保持原样。** 这些簇有几百个调用点，逐个改成
   `X::foo(...)` 会把改动撒进各个函数体；改为在 main.rs 加一行 glob 之后，调用点与重构前
   逐字节相同。代价是搬出的项需要 `pub(crate)`（否则父模块看不见子模块的私有项）。

4. **切口必须互不重叠。** 删除是按行号从后往前做的，如果两段区间重叠，同一段会被抽进两个文件
   （报 `ambiguous name`），而且第二次删除会错位、多删一段。脚本里加
   `assert` 校验区间互不重叠后再动手。

## 再拆一批的做法

1. 用 `python3 -c` 或脚本按行号整块切出，**切口必须落在顶层项的边界上，且不能把属性行
   （`#[derive]`/`#[allow]`）与其项拆开**——拆开会导致整段解析失败。
   一个文件可以装多个不相邻的块：`device_geometry.rs`（字节台账 + scratch 几何）、
   `qd4_gates.rs`（lane 校验 + 门禁）都是这种。
2. 新文件头加 `use super::*;`，内容原样保留（这些代码本来就在第 0 列，不需要重新缩进）。
3. 给搬出的顶层项、固有 impl 方法、结构体字段加 `pub(crate)`；
   trait impl 的方法和 enum 变体不能加（编译错误）。注意 `unsafe fn`、`const fn`、
   `extern "C" fn` 这些声明形式。
4. 在 main.rs 的声明块里加 `mod X;` 与 `use X::*;`（按字母序插进现有列表）。
5. 校验：除了 `rustc` 的 token 流比对（去掉空白与尾随逗号后应完全一致），还做更强的
   **字符级恒等式**——把搬走的块去掉文件头和注入的 `pub(crate)` 之后，加上新 main.rs 应等于
   原文件「逐字节」相等；声明块本身按设计被重写，所以要把它排除在外再比。
   最后一条用 `difflib` 独立实现，要求所有差异只能是空行（接缝处会少一个空行）。

## 怎么验证（本机环境）

```bash
cd <源码目录>/rust          # 例如 /srv/localmoe/build/localmoe-clean-git/rust
RUSTC_WRAPPER=$PWD/rustc-human-wrapper.sh \
CARGO_TARGET_DIR=/srv/localmoe/targets/<目录> \
cargo check --locked -p colibri-cuda            # 要再加 --tests 才算覆盖测试模块
./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda
```

- 直接跑 `cargo check` 会在钉住的 nightly 上触发编译器 ICE，**未改动的原始树同样如此**；
  必须套 `rustc-human-wrapper.sh`。`run-identified.sh` 已经自己设好，但 `cargo check`
  不会自动带上，要手动写。
- **`wsl.exe -e bash -c` 里没有 `cargo`**：跑构建要用 `-lc`，否则报 `exit=127 command not found`。
- 编译警告数（2026-10-13 实测，交付线 `fa61748`）：**54（bin）/ 16（test）**。搬走的死代码会让
  警告「换文件」——例如 `run_real_l0_rows` 那条从 `main.rs` 挪到 `mtp1_l0.rs`——所以判据是
  **警告集合去掉行号后逐条相同**，不只是总数相同。
- **判「谁没人用」必须带 `--tests`**：不带时 bin 构建报 57 条 `never used`，带上之后只剩 19 条。
  差的那 38 个不是死代码，是被 191 个测试当 oracle 用的 CPU 参考实现（`route_layer_cpu`、
  `indexer_scores_cpu`、`csa_compress_topk_cpu`、`compressor_initial_matches`、`run_cpu` 等）。
  照 bin 那份清单删，测试构建立刻碎。**两个构建里都报 `never used` 的，才是真死代码。**
- 2026-10-13 修好过一次测试构建：`93423d8` 删 DSpark 臂时，把 `v4_state.rs` 测试模块里两处引用
  （`DSPARK_STAGES`、`DsparkB5PersistentState`、`flash_dspark_b5_topk` 等）留在了原地，
  `cargo check --tests` 26 个 error、191 个测试整个跑不了，也就没人再看见 bin 那 57 条警告。
  **合分支后跑一次 `--tests`**，它比跑一次推理便宜。
- `cargo fmt --check` 在 HEAD 上本来就不通过，偏差多数是原码自带的。2026-09-16 逐文件量，
  整棵树 95 处（`v4_state.rs` 19、`tests.rs` 13、`mtp1_resources.rs` 8、`b2_generate.rs` 7、
  `state_audit.rs` 7 等），**main.rs 是 0 处**。判据是「不新增格式差异」：新文件里的偏差要么
  能在搬运前的原文里找到，要么来自 `pub(crate)` 注入把接近 100 列的签名顶过线，
  要么来自单独成文件后 rustfmt 换了排版判断（`kernels_module.rs` 有 1 处这种，
  一行 `if cond { a } else { b }` 被要求展开）。都不改语义。
- 真机验证：跑一次 Flash base 单步推理，要求 `argmax=55222`、`hidden_digest` 含
  `5ed8bf88beae3cfe`，且结果 JSON 里全部 261 个 16 位十六进制 digest 字段与
  `/srv/localmoe/targets/0731-sm120-clean-cuda128-default-block32-r2.jsonl` 逐一相同
  （`layers`/`fp4_accum`/`final_head`/`peak_device_used_bytes` 也要相同）。
- `./run-identified.sh test --arch sm_120` 有 8 个失败是既有环境问题（4 个缺
  `rust/colibri-cuda/results/...` 的 cutpoint 产物、2 个缺 `COLIBRI_CUBLAS_EXACT_LIB`、
  2 个是 `v4_state.rs` 里 V4-Pro 时代的过期断言）。判据是失败集合与改动前一致，而不是全绿。

## 还剩什么

main.rs 里只剩 `fn main()`（391 行，二进制入口必须在 crate 根）和约 160 行的声明/常量底座。
想再压只能动 `main()` 本身——那是拆函数，不是搬文件，得另立规矩（并且要重新设计验证口径，
因为「逐字节相同」这条保证会被打破）。
