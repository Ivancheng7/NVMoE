# Rust 侧模块划分说明

`rust/colibri-cuda/src/` 采用“轻量入口 + 独立职责模块 + 内联设备核模块”的组织方式。

## 命名规约

顶层函数前缀区分两类职责：

- **`run_*` = 可独立运行的入口程序**：自己读取配置、执行任务并打印或写出结果文件（如 `run_real_v4_flash_base`、`run_real_v4_flash_serve`）。
- **`step_*` = 模型前向执行中的计算步骤**：由层循环或生成主路径调用，不直接解析命令行/环境参数（如 `step_real_layer_rows`）。

## 核心模块地图

| 文件 | 职责说明 |
| --- | --- |
| `main.rs` | 二进制入口：挂载子模块、内联设备 PTX 模块，并分发 `FLASH_SERVE`、`FLASH_BASE` 与默认自检 |
| `kernels_module.rs` | `#[cuda_module] mod kernels` 设备端 CUDA 核函数实现（由 `main.rs` 通过 `include!` 引入） |
| `generate.rs` | Flash-0731 CLI 贪婪生成与本地服务请求驱动主循环 |
| `v4_serve.rs` / `v4_chat.rs` / `v4_tokenizer.rs` | 本地服务帧协议、对话模板拼装与分词器加载 |
| `b2_runtime.rs` / `b2_generate.rs` | `B2RawRuntime` 请求期资源管理、整模型固定权重显存常驻与结果记录 |
| `layer_rows.rs` | 单层前向主干编排（注意力相位、路由、Shared MLP 与 6 路并发专家 fan-out） |
| `layer_ops.rs` | 层内算子封装：mHC、RMSNorm、RoPE、压缩器（含复用池）、稀疏注意力、路由与 Shared Expert |
| `attention_device_chain.rs` / `device_attention.rs` | 设备侧注意力链执行、KV/压缩器边界提交（`commit_layer_attention_state`）与设备 scratch |
| `expert_slot.rs` / `v4_expert.rs` | 路由专家锁页读槽、异步 H2D 上传与 FP4 块缩放解码 |
| `fixed_host_cache.rs` / `fixed_layer_slot.rs` / `fixed_prefetch.rs` | 固定权重主机缓存、跨层设备槽位复用与后台预取线程 |
| `final_head.rs` | 流式 BF16 词表投影头（默认 6 线程按输出神经元并行归约） |
| `v4_state.rs` / `v4_fp4.rs` / `v4_router.rs` / `v4_spec.rs` | 模型几何契约、存储路径安全校验、E2M1 FP4 布局与精确路由归约 |
| `cublas_exact.rs` / `flash_arms.rs` | 指定版本 cuBLAS 动态加载与算子后端（cublas / libdevice / block32）配置 |
| `safetensors.rs` / `artifacts.rs` | Safetensors 分片按需范围读取（buffered / direct）与 SHA-256 校验工具 |
| `phase_profiles.rs` / `run_records.rs` | 逐层逐相位耗时统计与 JSONL 结果序列化 |
| `slice_checks.rs` / `tests.rs` / `tests_continued.rs` | 算子切片自检与单元测试套件 |

## 结构约定

1. **设备模块通过 `include!("kernels_module.rs");` 粘回 `main.rs`**：`cuda-oxide` 的 `#[cuda_module]` 宏要求内联模块以可见化核函数签名，`kernels_module.rs` 作为纯内联片段维护，文件头部不加 `use super::*;`。
2. **`mod` 声明位于 `tilelang_sum4!` 宏定义之后**：保证文本作用域宏对所有子模块可见。
3. **编译与检查需携带 `rustc-human-wrapper.sh`**：
   ```bash
   cd rust
   RUSTC_WRAPPER="$PWD/rustc-human-wrapper.sh" cargo check --locked -p colibri-cuda --all-targets
   ./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda
   ```
   `run-identified.sh` 已自动注入该 wrapper。
