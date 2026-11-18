---
name: kernel-landscape-2026
description: 与 colibri-cuda 直接相关的 2026 年算子库/参考实现/工具链外部指针（TileKernels、sm120
  SGLang 补丁、CUDA Rust 两条路）
metadata:
  node_type: memory
  type: reference
  originSessionId: sess_7dbc2b1c-2529-434e-9559-d04f54033332
---

2026-09 查到的、对本项目有直接用处的外部资源：

- **deepseek-ai/TileKernels**（2026-04-22 开源，MIT，TileLang 写的）：MoE gating /
  token-to-expert 映射、FP8·FP4·E5M6 的 per-token/per-block/per-channel 量化（含
  SwiGLU+量化融合）、**Engram gating**、**Manifold HyperConnection (mHC)**（Sinkhorn
  归一化 + mix split/apply）。本仓库 `run_l0_mhc_pre`、`COLIBRI_V4_FLASH_MHC_LINEAR_ARM`
  是同一个机制，这是目前唯一一份贴着硬件极限写的公开参考实现。**要求 SM90/SM100，
  CUDA 13.1+，PyTorch 2.10+，不含 sm_120**——只能读，不能装。
- **deepseek-ai/DeepGEMM**：已扩成一库多算子——FP8/FP4/BF16 GEMM、融合通信的 fused
  MoE（Mega MoE）、lightning indexer 的 MQA scoring、HyperConnection，全部 DeepJIT 运行时编译。
- **0xSero/deepseek-v4-flash-sm120**：给 SGLang 打补丁让 DeepSeek-V4-Flash FP8 在 SM120 上跑
  （FlashMLA sparse decode 只有 SM90/SM100，否则报 `Unsupported architecture for sparse
  decode fwd`）。可作为**进度标尺**：4×RTX PRO 6000、8K 37.6 tok/s、300K 15.1 tok/s，
  作者自述还差 split-KV/多 CTA sparse decode 与 SM120 tiling。
- **CUDA Rust 两条路**（NVIDIA 2026-09 宣布）：本项目用的 `cuda-oxide` 是 SIMT 那条
  （rustc codegen backend → Pliron IR → LLVM → PTX，pinned nightly，官方仍标 early alpha）；
  另一条 `cutile-rs` 是 tile 路线（stable Rust 1.89+、CUDA 13.3、CUDA Tile IR JIT），
  已被 HuggingFace Grout 和 mistral.rs 使用。
- **sm_120 硬事实**：无 tcgen05/TMEM、无 wgmma、TMA 仅单 CTA（无 cluster multicast）、
  无 2-SM cooperative、无 cvt.rs。给 sm_120 编 `.cluster_dim 2,1,1` 能编能装，但启动时
  被静默降为 (1,1,1)，kernel 里用 `cluster.sync` 或 cluster 共享 SMEM 即死锁/读垃圾。
  FP4 只有 warp 级 `mma.sync`（mxf8f6f4）一条路。另有 RTX 50 上 ptxas 13.1.115 在
  QMMA.E2M1 的 codegen 数值异常报告——与 [[flash-0731-official-oracle-unstable]] 记录的
  漂移是否同源，值得对一次。

**2026-10-06 补：DeepSeek 2026-09-30 开源昇腾基础设施。** 官宣三件（架在华为 CANN
（Compute Architecture for Neural Networks）栈上，目标是 Ascend 950 一代）：
**TileLang-Ascend**（TileLang 编译器后端指向昇腾 NPU）、**DeepGEMM-Ascend**（昇腾算子
基础库，报道 Dense GEMM 达硬件理论上限 99.8%、MegaMoE 约 98%）、配套通信库。对我们的
价值：localMoE oracle 钉的 tilelang 与 TileLang-Ascend 是**同一前端语言换硬件后端**，
是「同一算子、两种硬件」的现成对照材料（昇腾读法：DaVinci Cube 单元≈Tensor Core、
Ascend C 分块+队列风格更像 Triton）。

**2026-09-16 补：sm_120 的"坏"在生态里有明确分工——别把不同 bug 当成一个。**
"sm_120 上原生 FP4 MoE 出垃圾"在公开材料里至少是三类**互不相同**的问题，混起来会
推错结论：

1. **CUTLASS/FlashInfer 的 arch 后缀问题**（NVIDIA/cutlass#3096、vLLM#41063、
   TRT-LLM#14154）：`compute_120` 不带后缀 → MMA 被禁用、报 "Arch conditional MMA"；
   `compute_120a` → TMA warp-specialized grouped GEMM tactic 初始化失败；
   `compute_120f`（CUDA 13.0+）才完整。**dense FP4 GEMM 是好的，只有 grouped MoE 坏。**
   修法是给能力检查加 120 家族、FlashInfer 的 `supported_major_versions` 加 12、JIT 缓存
   要清。生产上他们推荐先用 **Marlin W4A16**（把 FP4 权重解到 FP16 再算）——即"放弃原生
   FP4 换正确性"。
2. **DeepGEMM 的 mHC 断言**：`hyperconnection.hpp:56` 的 `static_assert` 只白名单
   SM90/SM100/101，SM120 直接编译期拒绝（SM100 要 TMEM/UMMA/cluster multicast，SM90 要
   WGMMA，SM120 只有 warp 级 `mma.sync` + 99KB smem 上限）。这一条对我们有直接影响，
   因为 `run_l0_mhc_pre` / `COLIBRI_V4_FLASH_MHC_LINEAR_ARM` 是同一机制。
3. **TileLang 自己的 lowering bug**：tile-ai/tilelang#2699 是 sm_120 上 ThreadSync 没把
   `st.bulk` 记成共享内存写、少插 `__syncthreads`，症状是"能编能跑、结果静默错误且
   **非确定**"（他们的复现是 20/20 次与参考不同），0.1.12 上仍存在，PR#2700 修。
   注意 `compute-sanitizer --tool racecheck` **查不出**这个竞争。0.1.13（2026-08-03）
   才加进 `gemm_mma_sm120` / `GemmMMASm120BlockScaled`（`cuda.mma.blockscaled`）这条
   warp 级 block-scaled MMA 路径。

**Hopper/Blackwell 上的 NVFP4 block-scaled 硬件路径本身是存在的**（Colfax 的
SM12x 教程：`mma.sync` 的 mxf8f6f4 加 block scale，99KB smem 上限），所以"sm_120 不能做
FP4"是错的——能做，是若干**具体的 kernel 模板/编译路径**坏的。

**2026-09-18 实测更正：别把"无 tcgen05/wgmma"字面读成"只能手写 kernel / 算不了 FP4"。**
查过代码与诊断文档后，这条硬事实在本项目里只逼出了一处手写，且 FP4 的错跟它无关：

- **唯一真被硬件逼到的手写是稀疏注意力**：官方 TileLang `sparse_attn` 在 sm_120 上要
  `141312` 字节动态 SMEM（sm_89 上 `344064`），**两者都超消费级上限，驱动拒绝启动**
  （证据保留在废弃工作 `main.rs:9576`）。交付线的 `device_attention.rs` /
  `attention_device_chain.rs` 是自己写的。
- **官方 FP4 GEMM 在 sm_120 上算错，根因不是缺张量核**：官方 `fp4_gemm` 根本不用 FP4 张量核，
  是把 FP4 转 FP8 再做 `mma.sync m16n8k32`——**sm_120 合法指令**。所以"没 tcgen05 所以算错"
  **已被实测证伪**（`determine_target("auto")` 自己给 `-arch=sm_120a`，"按 sm_100 生成代码"
  也证伪）。坏在搬运/同步：权重整块设常量、scale 精确 1.0 时结果**仍每次不同**，是漏
  `__syncthreads`（上游 tile-ai/tilelang#2699 同类）。**换 tilelang 0.1.8→0.1.14 就修好**
  （三组对照：0.1.8 全错；0.1.14 两档 torch 都逐位稳定且数字相同——唯一变量是 tilelang）。
- **交付线干脆不用张量核**：默认臂 `block32`（`kernels_module.rs:1428`）= 寄存器内解 FP4→E4M3 +
  `shuffle_xor` warp 归约 + 严格 fp32 累加序，整树 `git grep mxf8f6f4|blockscaled|tcgen05|wgmma`
  **零命中**；`fp8_mma` 臂用的是 FP8 `mma.sync`。选它的理由是**可复现**（`flash_arms.rs:130`），
  不是硬件不能做。细节见 [[flash-0731-official-oracle-unstable]]。

工具面收在 `nvdisasm` / `cuobjdump -sass`、`nsys`、CUDA event 计时三件套即可；
`ncu` 在 WSL2 取不到硬件计数器（见 [[wsl-run-environment]]）。
