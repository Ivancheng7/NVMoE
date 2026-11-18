---
name: deepseek-line-scope-decision
description: 2026-10-03 拍板：DeepSeek 线优化收尾；CUDA graph 对我们无用（~0.1%，且录不进去）但在
  nano-vLLM 实验室值 4.3×——不矛盾，差别是「贴在哪面墙」
metadata:
  node_type: memory
  type: project
  originSessionId: sess_1694f5e1-f4b7-4970-b96c-c9093430f909
---

2026-10-03 用户判断 DeepSeek-V4-Flash 这条线「基本做得够多了」，给出三条取舍。我核对代码后
全部同意，并把 CUDA graph 那条的表面矛盾解开了。

## 三条裁定

- **第 4 条 调度器 + continuous batching：不做。** 每个请求都在本地、同一时刻基本只有一个用户
  发起，`GenerationScheduler` 的 `capacity=1` 就是对的（它的 docstring 原话是 "independent KV
  contexts"）。理由同 [[bandwidth-wall-decides-what-optimization-pays]]：连续批处理要「多请求拼
  一批摊薄权重读取」，单用户并发 1、batch 恒为 1，价值归零。只有把「学 AI infra」设成目标时
  才值得当练习搭——而那个需求已经交给 nano-vLLM 实验室（[[nano-vllm-lab-wsl]]）。

- **第 5 条 CUDA graph：对我们不做，但别推广成「CUDA graph 没用」。** ⚠️ 本次最重要的对照：
  - **对我们（V4-Flash）**：~0.1% 量级站得住。raw 上界约 0.3–0.6%（每 token 约 1300–2200 次
    启动 × 几微秒），但我们是磁盘+PCIe 带宽墙，CPU 跑在 GPU 前面，启动延迟本来就被藏掉。
    **更硬的理由是录不进去**：compressor 仍走主机 cuBLAS（`layer_ops::run_compressor_official_cublas_host`），
    层循环里每层一次 D2H 同步喂它，而 graph capture 不容许捕获区内有主机同步；另外 MoE 路由每
    token 变指针，`coli_cuda_expert_group_issue/_take`（`c/backend_cuda.h:74`）已经在做手工异步下发
    ——那部分已经是「穷人的 graph」。
  - **对 nano-vLLM 实验室（Qwen3-0.6B，32 请求批处理）**：`enforce_eager` 关→开 =
    **1083 → 4676 tok/s = 4.3×**（`/srv/localmoe/lab/run_first.py`，`EAGER=1|0`）。
  - **两者不矛盾，差别就是「贴在哪面墙」**：0.6B 全量常驻 + 多请求 ⇒ 权重一批只读一遍、每步成本
    被 kernel 启动开销主导 ⇒ 启动墙；V4-Flash 每 token 从盘上搬 3.45 GB ⇒ 带宽墙，启动开销被藏。
    **同一个开关，一面墙上值 4.3×，另一面墙上值 0.1%**——这就是
    [[bandwidth-wall-decides-what-optimization-pays]] 那张表的实证。

- **第 5 条 prefix caching：唯一还有价值的。** 用法具体是**多轮对话复用上文**——第 N 轮 prompt
  含前 N−1 轮全部内容，而 `GenerationScheduler` 现在给每个请求独立的 KV context、用完就丢，
  等于每轮重新预填充整个历史。MLA 压缩 KV 让要缓存的东西很小，正好合适。

## 唯一不属于「优化」的遗留

`CONTEXT_WINDOW_EXCEEDED` 还在服务路径（`rust/colibri-cuda/src/generate.rs:1269`）、128 token
输入上限、首字 ~23.9 s。「优化够了」不覆盖这条——它属于「能不能用」。

**How to apply:** 以后别再把 CUDA graph / continuous batching 列进 DeepSeek 线的候选。被问起
CUDA graph 时先问「哪面墙」，再引上面那两个数字。相关：[[ai-infra-learning-direction]]、
[[flash-0731-vs-vllm-five-stage-map]]、[[flash-0731-per-token-floor]]、[[flash-0731-attention-compressor]]
