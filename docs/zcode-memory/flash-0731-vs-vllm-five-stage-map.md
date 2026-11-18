---
name: flash-0731-vs-vllm-five-stage-map
description: 我们的引擎对着 vLLM 五步学习路线各做到哪——前向/KV/prefill-decode 做透、调度只做排队、prefix
  caching 与 CUDA graph 全无
metadata:
  node_type: memory
  type: project
  originSessionId: sess_1694f5e1-f4b7-4970-b96c-c9093430f909
---

2026-10-03 用户给出 vLLM 式五步 AI infra 学习路线（由内向外），要求对照我们实际做了什么。
结论：**五条里做透 3 条，第 4 条半做，第 5 条全无。**

| 路线图那条 | 我们 | 证据 |
| --- | --- | --- |
| 1 模型前向（RMSNorm/RoPE/GQA/MLP） | 做透，**但是 MLA 不是 GQA** | RMSNorm：`rmsnorm_v4_bf16` / `run_l0_rmsnorm(_device_into)` / `mhc_committed_rsqrt`；RoPE：`run_rope_device_into` / `compressed_rope_freqs` / `DeviceRequestRopeLut`；attention：`attention_absorb_batch_kernel`，头文件注释原文 "Decode-only MLA weight-absorption core for one token"；MLP 是 MoE：`moe` + `coli_cuda_expert_group` |
| 2 KV cache 管理 | 做了「写死分配」那半 | `V4KvCache`(v4_state.rs) / `.commit` / `commit_kv_row` / `preflight_kv_cache_commit` / `kv_alloc`(colibri.c) / `run_l0_kv_quantize_device_into`；存的是 `compressed_kv_bf16`（MLA latent）。**没有 PagedAttention / block table** |
| 3 prefill/decode 分离 | 主战场，做透 | 两条代码路径：`attn_pipe_prefill` vs decode-only `attention_absorb`；`layer_stationary_prefill`；实测预填充 393.69→334.65 ms/层（算力/行）vs decode 的带宽墙 |
| 4 调度器 + 连续批处理 | **只做了排队，没做批处理** | `GenerationScheduler`(c/openai_server.py:95)，docstring 原文 "Bounded FIFO admission for the engine's independent KV contexts"，`capacity` 默认 1；`run_serve_mux`(colibri.c)。有准入/有界队列/超时/背压(429/503)，**没有把多请求拼进同一 batch 跑一步再换出** |
| 5 prefix caching、CUDA graph | 都没有 | 两者搜索 0 命中（只有 cuda-bindings 的 FFI 符号；`v4_flash_0731_prefix_oracle.py` 是 model id 前缀，不是 prompt 前缀复用）。远亲是 spec decode（`spec_decode` / DSpark / `mtp1_*`），但那是「少跑几趟」不是「免费午餐」 |

**2026-10-06 已核实的 KV 位置事实**：生成路径每层 KV 历史是**显存**上的 128 行环形窗口（`DeviceBuffer`，device_attention.rs:1519，`KV_WINDOW=128`/v4_state.rs:29，512 维 bf16 ≈128 KB/层×61 层≈7.8 MB）；主机那份 `Vec<u16>` 的 `V4KvCache`（v4_state.rs:4258）只是 `preflight_kv_cache_commit`（v4_state.rs:4721）用的对账镜子，服务路径不碰；新 token 的 KV 行在显存上算完直接进环，不过 PCIe。所以「给我们的 KV 上 k8v4 式量化」判词是**没东西可压**（比「能做没钱」更硬），用户印象里的内存↔显存大通信是 routed 专家权重，不是 KV。环的机制（同日已核）：`window_topk_cpu`（v4_state.rs:3141）读的就是整个环（`absolute % window` 绕圈、写满覆盖最老）= 最近 128 token 滑窗；更老的历史靠压缩摘要行，每 4 token 攒 1 行（`csa_compressed_rows_after=(p+1)/4`，device_geometry.rs:24），容量常量 KV_COMPRESSED=32（原文+摘要共 160 行，v4_state.rs:29-31）；日常 8~32 token 测试连环比都填不满。

两个坑：

- 路线图写 GQA，我们做的是 **MLA**。查资料要查 MLA / weight absorption / DeepSeek attention，查 GQA 对不上。
- 「页缓存命中」（OS 文件缓存）**≠** prefix caching（跨请求复用相同 prompt 前缀的 KV），我们只有前者。

路线图把第 5 条叫「免费午餐」，**对我们不成立**：CUDA graph 治 kernel 启动开销，而我们的墙是带宽/读盘；
prefix caching 要多请求共享前缀，我们 capacity=1。所以第 5 条得等第 4 条做完才有意义。

**Why:** 用户要分清哪些是项目已经替他练过的、哪些还是空白，好决定下一步学什么；第 4 条的
`GenerationScheduler` 是现成脚手架（抬 capacity + 每步拼 batch 就是最小连续批处理）。

**How to apply:** 用户问「我们离 vLLM 差什么」「下一步学什么」时直接用这张表，别重查一遍代码。
细则见 `docs/术语表.md`。[[ai-infra-learning-direction]] [[terms-glossary-industry-alignment]]
