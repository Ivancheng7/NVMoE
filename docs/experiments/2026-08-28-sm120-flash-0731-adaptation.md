# Flash-0731 `sm_120` 适配记录（2026-08-28）

## 结论

新下载的 ModelScope `deepseek-ai/DeepSeek-V4-Flash-0731` 已能在 RTX 5060 Ti（`sm_120`）上完成 43 层真实 base-only 推理，并执行 CUDA 12.8/cuBLAS GPU exact final-head。`localmoe-clean` 现已把 Flash-0731 的默认 FP4 专家臂切到确定性的 `block32` 路径；同一环境连续两次完整运行均得到 token `55222`，hidden digest 均为 `5ed8bf88beae3cfe`。

官方 TileLang FP4 核在当前 `sm_120` 环境不是稳定 oracle：已保存的官方 full-43 运行分别出现 token `67900`（v4）、`55222`（v5）和 token `0`（r9），对应 hidden SHA-256 也不同。官方 L0 routed FP4 中间量重复运行同样发生漂移。因此不能把单次 `67900` 或某个随机中间快照当成唯一数值真值；当前可复现的基线是 Rust `block32` 路径，尚未声称 FP4 逐元素 `PASS_EXACT`。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- safetensors：48 个，权重总字节数 `166886535336`
- `config.json` SHA-256：`6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256：`98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- `inference/model.py` SHA-256：`c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f`
- `inference/kernel.py` SHA-256：`59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`

## 运行证据

构建使用 `cargo oxide build --arch sm_120 -- --release`，运行时固定使用 CUDA 12.8 的 `libcublas.so.12.8.5.5`，实际 `cublasGetVersion=120805`。

单步 GPU exact 结果：

- 43/43 层完成
- final-head：`gpu_exact`
- token：`55222`
- 端到端约 `14.70 s`
- 峰值设备内存约 `1.53 GB`

连续 3 步 GPU exact 结果文件：`/srv/localmoe/targets/0731-sm120-gpu-exact-3step2/result.jsonl`

- 输入/生成序列：`[0, 55222, 33832, 42549]`（生成部分为 `[55222, 33832, 42549]`）
- 3 步总耗时约 `41.52 s`
- 峰值设备内存约 `3.72 GB`

## 代码变更

- `cublas_exact.rs` 的 Flash-0731 ABI 身份已固定到 CUDA 12.8/cuBLAS 12.8.5.5，避免误加载 CUDA 13.x。
- hash-router layer 0 的 parity 采集不再错误要求 learned-router RMSNorm trace，能够完整采集 routed expert 边界。
- Flash-0731 未显式指定臂时默认使用 `block32`；`fp8_mma` 与 TileLang 缓存臂保留为显式实验项，待目标卡上有稳定证据后再进入性能结论。
- `run_real_v4_flash_base` 在 GPU exact 模式下创建并复用 final-head workspace，避免多 token 时重复写同一路径的 trace artifact 而崩溃。
- `sm_120` final-head capability gate、默认构建架构和 CUDA/cuBLAS 运行说明已同步。
- 官方 final-head 比对仍需显式设置 `COLIBRI_V4_FLASH_FINAL_HEAD_OFFICIAL_DIR` 才开启；没有稳定独立 oracle 时只报告 runtime GPU exact，不伪造 exact parity。

## 还没做完的

1. 修复或替换 `sm_120` 上不稳定的官方 TileLang FP4 参考核，先拿到可重复的 FP4 专家 oracle；在此之前不做 routed FP4 逐元素精确声明。
2. 在稳定 oracle 之上，再按层比较首个分歧点；当前已确认 mHC/router 的 CUDA 12.8 cuBLAS 路径和 layer 0 非 FP4 边界可复现，剩余差异集中在 routed FP4 GEMM。
3. 完成真实 serve 协议的多请求/取消/持续吞吐测试后，再更新性能结论。
