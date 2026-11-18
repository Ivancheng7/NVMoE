---
name: ai-infra-layer-map
description: AI Infra 六层地图 + 两处「用户仓库里已经有」的挂钩（MTP 就是他的 DSpark/mtp1_*、NVFP4 ≈ v4_fp4.rs 的 E2M1 block32）+ 社区量化命名两套体系与读陌生引擎的五部件法
metadata:
  node_type: memory
  type: project
  originSessionId: sess_7df38066-8b46-4d73-bb93-e1c5c31669ac
---

2026-10-03 用户「AI Infra 好像不只是算子优化吧」发问后落的文档：
Windows `C:\Users\27236\Desktop\ai-infra-lab\AI-Infra-地图.md`（与 [[nano-vllm-lab-wsl]] 记的
`学习路线.md` 并列：那份讲怎么动手，这份讲整块地在哪）。

## 六层（我用这张图讲，别另起一套）

1. **算子层** — FlashAttention / CUTLASS / Triton / FlashInfer ↔ 他的 `kernels_module.rs`
2. **数值格式与量化层** — FP8 / NVFP4 / MXFP4 / INT4 / GGUF / AWQ / GPTQ ↔ `v4_fp4.rs`
3. **显存与缓存管理层** — 分页 KV、prefix caching、offload、锁页、预取 ↔ `fixed_layer_*.rs`、`expert_slot.rs`
4. **调度与执行层** — 连续批处理、chunked prefill、抢占、CUDA graph、PD 分离 ↔ `GenerationScheduler`（只做了排队）
5. **模型结构层** — MoE 路由、GQA/MLA/GDN/QSA、MTP、多模态塔 ↔ `v4_router.rs`、`mtp1_*.rs`
6. **服务层** — OpenAI 兼容 API、请求队列、限流 ↔ `c/openai_server.py`

**横切的一层不属于上面任何一层：性能测量**（roofline、profiling、A/B）。没有它上面六层全是瞎猜；
他项目里的相位账就在这一层。

**关键论点**：AI Infra ≠ 算子优化。第 1 层最成熟、最容易拿来就用（CUTLASS/FlashInfer），
真正拉开差距的是 **2/3/4 层**——因为它们跟「手上什么硬件、跑什么形状的模型」强绑定，没有通用答案。
「哪层重要」的判据是**墙**不是层，见 [[bandwidth-wall-decides-what-optimization-pays]]。

## 两处挂钩（用户不知道「这东西我自己已经在写」）

- **MTP = Multi-Token Prediction**，训练时加额外预测头学会预测下下个 token，推理时当**投机解码**用
  （一趟提多个候选、下一趟一起验证，认下来的白赚）。**是训练进权重的能力，不是运行时技巧**——
  所以 Qwen3.8 权重自带 MTP 头，llama.cpp 2026-09 支持后 27B 快约一倍，这就是「48–50 tok/s」里的一部分。
  **用户说「MTP 不太懂」，但他仓库里的 `mtp1_draft.rs` / `mtp1_l0.rs` / `mtp1_resources.rs` /
  `dspark.rs` 就是这东西的两个试验臂**（见 [[flash-0731-dspark-arm-state]]，接受率至今 0）。
- **NVFP4 ≈ 他已经会的大半**。FP4 只有一个元素格式 **E2M1**（1 符号+2 指数+1 尾数，只能表示
  `0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6` 这 16 个数），动态范围差，必须配块缩放。各家只差三个参数：

  | 格式 | 块大小 | 缩放系数 |
  | --- | --- | --- |
  | NVFP4（NVIDIA） | 16 | FP8 (E4M3) + 整张量 FP32 二级缩放 |
  | MXFP4（OCP，gpt-oss 用） | 32 | FP8 (E8M0)，即 2 的整数次幂 |
  | **他仓库里的 FP4** | 32 | FP8 缩放 |

  即他已在做「E2M1 + 块缩放」，**只差「块 16」和「二级缩放」**。

## 社区量化命名：两套不通用，别放在一起比

- **llama.cpp 私有体系**：`Q*`/`IQ*`/`UD-`；`K`=k-quant（块内混合精度）、`I`=importance matrix
  （先算哪些权重重要）、`_S/_M/_L`=尺寸档、`_XS/_XXS`=extreme small、`UD-`=Unsloth 逐层配方。
- **学术/GPU 体系**：AWQ（保护重要通道）、GPTQ（逐层补偿误差）、NF4（16 个格子放在正态分位点上）、
  FP8 / NVFP4 / SmoothQuant（权重+激活都压）。

## 读任何陌生小引擎的五部件法

不管 1450 行还是 300 行，都只回答五个问题：①权重怎么加载、什么格式 ②KV cache 怎么管
③一次前向走哪些算子 ④这一步算哪些请求、怎么组批 ⑤采样循环怎么收尾。
**读陌生引擎就是找这五块**，不必逐行读。（nano-vLLM 的对应位置见 [[nano-vllm-lab-wsl]] 里那份对照表。）

## 用户当前的知识状态（2026-10-03）

自认不懂：**MTP、社区各类量化版本、自己写的小框架**。原先的默认假设是「AI Infra = 算子优化」，
自己纠正过来了。**判断是我们讲过的都已经落进上面两份文档，下次别再从头讲一遍，直接从文档接着往下。**

**Why:** 用户是在建自己的知识骨架，不是要术语清单；他最容易接受的方式是**先把新概念挂到他已有的
东西上**（MTP 挂 DSpark、NVFP4 挂 v4_fp4.rs），再讲抽象层。六层图和这两个挂钩是这套讲法的骨架。

**How to apply:** 他再问某个 AI Infra 概念时，先归层、再找挂钩、最后才讲定义；术语首次出现给英文原词
（[[terms-glossary-industry-alignment]]）。走 [[plain-language-reporting]] 的大白话。
[[ai-infra-learning-direction]] [[flash-0731-vs-vllm-five-stage-map]]
