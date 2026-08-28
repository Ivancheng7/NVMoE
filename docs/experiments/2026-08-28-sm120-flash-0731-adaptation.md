# Flash-0731 `sm_120` 适配记录（2026-08-28）

## 结论

新下载的 ModelScope `deepseek-ai/DeepSeek-V4-Flash-0731` 已能在 RTX 5060 Ti（`sm_120`）上完成 43 层真实 base-only 推理，并执行 CUDA 13.1/cuBLAS GPU exact final-head。连续 3 个 greedy step 可以稳定完成，说明适配已经越过“一次性 smoke”阶段。

本次结果尚不能宣称通过仓库原有的官方数值 gate：仓库 README 记录的 token `33832` 对 `token_ids=0` 的首步参考，与当前新模型首步 `55222` 不一致。模型元数据与官方 0731 固定文件哈希一致，因此需要单独重新生成/复核 0731 的 full-43 官方 oracle，不能把这次运行标记为 `PASS_EXACT`。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- safetensors：48 个，权重总字节数 `166886535336`
- `config.json` SHA-256：`6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256：`98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- `inference/model.py` SHA-256：`c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f`
- `inference/kernel.py` SHA-256：`59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`

## 运行证据

构建使用 `cargo oxide build --arch sm_120 -- --release`，运行时使用 CUDA 13.1 的 `libcublas.so.13.2.1.1`，实际 `cublasGetVersion=130201`。

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

- `run_real_v4_flash_base` 现在在 GPU exact 模式下创建并复用一个 final-head workspace，避免多 token 时重复写同一路径的 trace artifact 而崩溃。
- `sm_120` final-head capability gate、默认构建架构和 CUDA/cuBLAS 运行说明已同步。
- 官方 final-head 比对改为显式设置 `COLIBRI_V4_FLASH_FINAL_HEAD_OFFICIAL_DIR` 才开启；没有独立 oracle 时只报告 runtime GPU exact，不伪造 exact parity。

## 未决事项

1. 安装或提供与固定 0731 revision 匹配的 PyTorch/CUDA 环境，重新生成 full-43 官方 oracle。
2. 用 oracle 按层比较首个分歧点，确认 `55222` 与 README 的 `33832` 是参考语义漂移还是实现数值差异。
3. 完成真实 serve 协议的多请求/取消/持续吞吐测试后，再更新性能结论。
