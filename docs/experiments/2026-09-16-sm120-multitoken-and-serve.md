# 连续 8 token 与服务路径首次真机端到端（2026-09-16）

## 结论

这条链今天往前走了一步半。

**连续 8 个 token 从 CLI 出。** 当前 HEAD 的二进制跑 `COLIBRI_V4_FLASH_BASE=1` + `COLIBRI_V4_GREEDY_STEPS=8`，
生成 `[55222, 33832, 42549, 42549, 42549, 42549, 42549, 127384]`，整个请求 `107.9 s`，
峰值显存 `1466499072` 字节（1.466 GB）。**峰值显存和单步跑完全一样**，不随步数增长；第一颗 token 的
`hidden_bf16_digest` = `5ed8bf88beae3cfe`，与单步基线逐位相同。单步耗时 12.2–15.0 s（第一颗含预填充）。

顺带纠正一条旧印象：2026-08-28 记录里"3 步峰值 3.72 GB"是那次用 `gpu_exact` final head 的开销，
不是每步泄漏——默认 CPU final head 下 1 步和 8 步的峰值一模一样。

**服务路径（stdio 协议）第一次真机端到端跑通。** 真实 prompt 文本进（`你好，你是谁？`），
走 tokenizer + chat 模板 → 生成 8 个 token → 解码文本按 `DATA` 帧流式返回，最后 `DONE` 带统计。
同一个进程里连着两次相同请求，返回的 token id 完全一致；另一个进程用同一 prompt 再跑，首 token 也一致。

**但它慢，而且慢在一个固定开销上。** 第一个 token 要 95–157 s——`--max-tokens 1` 也要 95 s，
说明和生成长度无关、**每次请求都付**；之后 16.6 s/token，一次 8-token 请求约 4.5 分钟。
同一条链的 CLI base 路径单步只要约 13 s。**这个开销在哪，没查清。**

没声称的部分：**没走 OpenAI HTTP 网关**（`c/openai_server.py` 的 `Engine` 类是给 C 引擎写的，
按 `SNAP`/`SERVE`/`NGEN` 那套环境变量和额外握手行工作，能不能驱动 Rust 引擎没验证）；
没测并发、取消、多请求吞吐；输出文本本身不成句（id-only 贪心、base 路径），**本次不评价语义质量**，
只记录链路通不通。

## 固定输入与模型

- 模型目录 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 与 `2026-09-16-sm120-first-layer-divergence.md` 是同一份权重、同一台卡（RTX 5060 Ti 16 GB，`sm_120`）

## 运行证据

CLI 8 步：

- 二进制 `/srv/localmoe/targets/delivery-8step-target/release/colibri-cuda`
  （SHA-256 `a08bb7c331ea31e7acdcbd22e2e8e770260176492b377053f0f4c2ce997c7495`）
- 结果 `/srv/localmoe/targets/0731-sm120-8step.jsonl`
- 环境：`COLIBRI_V4_FLASH_BASE=1`、`COLIBRI_V4_FLASH_TOKEN_IDS=0`、`COLIBRI_V4_GREEDY_STEPS=8`、
  `COLIBRI_V4_STREAM_READ_MODE=buffered`、`COLIBRI_V4_FLASH_PREFETCH=1`、
  `COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas`、`COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas`、
  `COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice`、
  `COLIBRI_CUBLAS_EXACT_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`。
  `COLIBRI_V4_FLASH_FP4_ACCUM_ARM` 和 `COLIBRI_V4_FLASH_FINAL_HEAD_ARM` **故意不设**，
  用来顺带确认默认臂仍是 `block32` / `canonical_cpu_streamed_bf16`
- 单步对照 `/srv/localmoe/targets/0731-sm120-1step-head.jsonl`：43 层、峰值同样 `1466499072`

服务（stdio 协议，`COLIBRI_V4_FLASH_SERVE=1`）：

- 客户端 `rust/colibri-cuda/tools/flash_serve_client.py`（本次新增，仓库内）
- 探测日志在 WSL 侧 `/srv/localmoe/targets/flash-serve-probe.log`、`-probe2.log`、`-probe3.log`
  （`*.log` 被 `.gitignore` 忽略，不入库，所以原始 `DONE` 行抄在下面）

四次探测的原始 `DONE` 行：

```text
# flash-serve-probe.log，prompt「你好」，8 token
DONE 1 STAT 8 0.039513 0 0 5 1 86485.697 16568.025 max_new_tokens ids=118142,128839,128838,87337,109405,109405,109405,87337
# flash-serve-probe2.log，prompt「你好，你是谁？」，同进程两次请求 → 同一组 id
DONE 1 STAT 8 0.029144 0 0 8 1 157331.189 16738.583 max_new_tokens ids=58572,42549,42549,109865,42549,42549,42549,42549
DONE 2 STAT 8 0.031706 0 0 8 1 136216.308 16585.980 max_new_tokens ids=58572,42549,42549,109865,42549,42549,42549,42549
# flash-serve-probe3.log，另一个进程、prompt「你好」、max-tokens 1
DONE 1 STAT 1 0.010496 0 0 5 1 95272.053 0.000 max_new_tokens ids=118142
```

第一行和第四行是不同进程、同一个 prompt（`你好`），首 token 都是 `118142`；第三行是同进程重发同一
prompt，id 逐个相同。所以输出只由 prompt 决定，这四次都稳定。

`STAT` 后面的字段顺序：`completion_tokens / tokens_per_second / cache_hit / rss / prompt_tokens /
length_limited / ttft_ms / itl_ms / stop_reason / ids`。`tokens_per_second` 是
`completion_tokens / 整个请求耗时`（8 / 274.5 = 0.0291），不是 ITL 的倒数。

客户端侧的墙钟：第一个 8-token 请求 `274.5 s`（其中 TTFT `157.3 s`，ITL `16.59 s` × 7 ≈ 116 s），
第二个 `252.3 s`；`--max-tokens 1` 那次 `95.3 s`。

复现命令（WSL 内）：

```bash
python3 <repo>/rust/colibri-cuda/tools/flash_serve_client.py \
  --binary /srv/localmoe/targets/delivery-8step-target/release/colibri-cuda \
  --model-dir /srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --cublas-lib /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5 \
  --prompt-json '"\u4f60\u597d\uff0c\u4f60\u662f\u8c01\uff1f"' --max-tokens 8 --repeat 2
```

（中文 prompt 走 Git Bash → `wsl.exe` 容易被转码，用 `--prompt-json` 传 `\uXXXX` 转义最稳。）

## 代码变更

`tools/flash_serve_client.py`（新增）：驱动这条 stdio 协议的最小客户端。仓库里原本只有协议本身的单元测试
（`src/v4_serve.rs` 里的帧格式与拒绝用例），没有任何能真跑一次的客户端，所以只能现写。
它负责起进程、等 `READY`、发 `SUBMIT`、按长度读 `DATA` 帧、收 `DONE`、把统计打出来。

## 未决事项

1. **服务路径首个 token 的固定开销（95–157 s）没定位。** 它是每次请求都付，且与 `max_tokens` 无关。
   下一步该做的是拿它对照 base 路径同一相位：base 路径的第一步只要约 15 s，两边都调用同一个
   `run_v4_b2_generate_arm`，但传的臂参数不同（base 传 `record_final_head_loader=true, capture_layer_trace=true`，
   serve 传 `false, false`）。**这也正是该拆的那个函数**（450 行）。
2. **OpenAI HTTP 网关能不能驱动 Rust 引擎，没验证。** `c/openai_server.py` 的 `Engine` 起的是 C 引擎、
   用 C 引擎的环境变量，还解析 `TIERS`/`hwinfo` 之类的握手行；要对上得先确认 Rust 侧发不发这些。
3. **取消、并发、多请求没测。** 协议里有 `CANCEL`，`validate_submit` 只允许 `cache_slot=0`，
   也就是单槽串行；15 分钟一次请求的吞吐对"服务"这个目标来说还不够。
4. **多 token 的正确性只有第一颗锚住了**（与单步基线逐位相同），第 2–8 颗只做了同一运行内的自比对，
   没有独立参考。
