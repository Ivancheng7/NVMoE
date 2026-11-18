---
name: deepseek-v41-flash-facts
description: DeepSeek-V4.1-Flash（CED 架构）已核实的事实、两处我曾说错的纠正、官方参考怎么拿，以及 2026-09-16 决定暂不移植
metadata:
  node_type: memory
  type: project
  originSessionId: sess_a9e0716d-7d0a-462a-bfe0-b5dd06f16f66
---

官方 `deepseek-ai/DeepSeek-V4.1-Flash`，HF revision `dba1be0a40aa45a94ad051997016db3960a90277`，MIT，2026-09-10 发布，公告在 `api-docs.deepseek.com/news/news260910`。**已逐条核实**：`model_type` 从 `deepseek_v4` 变成 `deepseek_v41`；config 从扁平变嵌套 `text_config`/`vision_config`；CED——40 层拆成 20 层 causal encoder + 20 层 decoder；`compress_ratios=[0,0,2×18,1×20,0,0,0]`；`rms_norm_eps` 1e-6→1e-20；FP8 块 128×128→32×32；`num_hash_layers` 删掉换成 Engram（第 [1,14] 层）；候选池 `candidate_source_layer=20`；FP4 主 KV；`hc_mult=4` 的 `pre_mix` 跨层传递；DeepSeek-ViT + Aligner。权重 510.3 GB / 48 分片。

**两处我说错过、要记住的纠正：**
1. 本地**从来没有**下载过官方参考实现，`%LOCALAPPDATA%\Temp\dsv4\` 根本不存在。
2. decoder 不是"从 4 个源层投影 KV"。参考实现里是一个共享槽位、**最后写入者生效**：只有 `kv_source_layers=[2,8,14,20]` 会写，第 21–39 层读的是第 20 层写的那份。

**拿参考时的三个坑：** 上游 `transformers` **没有** `deepseek_v41`；官方 `inference/model.py` 的 `python model.py` 自测是**形状 smoke test**（权重未初始化），**不能当数值 oracle**；真 oracle 仍得走 prefix-oracle 那套技术配全量 checkpoint。

**2026-09-16 决定：暂不移植**，先补 Flash-0731 在 sm_120 上的证据链。理由：README 招牌证书在 sm_120 上站不住（见 [[flash-0731-official-oracle-unstable]]），没有可靠靶子时写 v4.1 kernel 是盲写。注意 CED 的设计思路（有序层、每个共享张量一个槽位、生产者先写消费者后读、请求间不重置）和现有持久化层状态与 lease 规则是同一套。

相关：[[v4-state-hardcoded-geometry]]、[[flash-0731-baseline-run-recipe]]
