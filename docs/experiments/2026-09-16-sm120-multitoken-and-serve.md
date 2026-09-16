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

**但它慢，而且慢的地方不在显卡。** 第一个 token 要 95–157 s；`--max-tokens 1` 也要 95 s，
说明和生成长度无关。真正的原因是**没有批量预填充**：耗时按 prompt 长度线性涨，约 19 s 一个 prompt token
（四个数据点见下节），而一次完整的 43 层单步是 12–15 s。CLI 那条之所以看着"只是 13 s/步"，
是因为它的 prompt 只有 1 个 token，看不见这一段。**13 s/步是这张卡 + 每步流 167 GB 权重的正常代价；
"prompt 有几个 token 就先等几个 13 秒"是循环写法问题，两件事要分开算。**

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
# flash-serve-probe4.log，prompt「hi」、max-tokens 1（chat 模板把它也撑到 5 个 token）
DONE 1 STAT 1 0.011045 0 0 5 1 90540.646 0.000 max_new_tokens ids=62268
```

第一行和第四行是不同进程、不同 prompt（`你好` / `hi`）但**同样的 `prompt_tokens=5`**，首 token 前耗时
`86.5 s` / `90.5 s`；第三行是同进程重发同一 prompt，id 逐个相同。所以输出只由 prompt 决定，
这四次都稳定。

### 首个 token 的耗时是 prompt 长度乘出来的

把上面几行和 CLI 单步那组摆在一起，`prompt_tokens` 和"第一个 token 出来之前花了多久"是一条直线：

| `prompt_tokens` | 首 token 前耗时 | 来源 |
| --- | --- | --- |
| 1 | 15.0 s | CLI base，`COLIBRI_V4_FLASH_TOKEN_IDS=0` |
| 5 | 86.5 / 90.5 / 95.3 s | serve，`你好` 与 `hi` |
| 8 | 136.2 / 157.3 s | serve，`你好，你是谁？` |

斜率约 **19 s 一个 prompt token**，而一次完整的 43 层单步是 12–15 s——同一个量级。

代码上对得上：`generate.rs` 的生成循环（第 390 行起 `while !b2_generate_active_indices(...).is_empty()`）
每个迭代只从 `input_ids[position]` 取**一个** token，prompt 的 token 和后面生成的 token 走同一个循环，
**没有批量预填充**。prompt 每多一个 token，就多一整遍 43 层。

对照一次：CLI 那条 9 个 token（1 prompt + 8 生成）用了 107.9 s，约 12.0 s/迭代；
serve 那条 13 个 token（5 prompt + 8 生成）用了 274.5 s，约 21.1 s/迭代。两个每迭代成本**不一样**，
serve 那次高约 1.75 倍，这一部分**没有解释**（可能是更长的前缀、每步多一次 decode，或 serve 关掉了
那两个 trace 臂之外的其他差异）。

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

1. **没有批量预填充：prompt 每多一个 token 就多一整遍模型。** 证据在上面那节（四个数据点连成斜率
   约 19 s/token 的直线，循环每迭代只吃一个 token）。后果是：5 个 token 的 prompt 忍得了，
   200 个 token 的真实对话要在第一个字之前等约一小时。下一步不是"优化循环"，而是把 serve 路径接到
   已有的预填充实现上——仓库里本来就有 `layer_stationary_prefill.rs` 那条路。
2. **serve 每迭代比 CLI 贵约 1.75 倍（21.1 s vs 12.0 s），没解释。** 两个路径调的是同一个
   `run_v4_b2_generate_arm`，差别只有 prompt 长度和 `record_final_head_loader` /
   `capture_layer_trace` 两个臂参数（CLI 传 `true, true`，serve 传 `false, false`）。
3. **OpenAI HTTP 网关能不能驱动 Rust 引擎，没验证。** `c/openai_server.py` 的 `Engine` 起的是 C 引擎、
   用 C 引擎的环境变量，还解析 `TIERS`/`hwinfo` 之类的握手行；要对上得先确认 Rust 侧发不发这些。
4. **取消、并发、多请求没测。** 协议里有 `CANCEL`，`validate_submit` 只允许 `cache_slot=0`，
   也就是单槽串行。
5. **多 token 的正确性只有第一颗锚住了**（与单步基线逐位相同），第 2–8 颗只做了同一运行内的自比对，
   没有独立参考。
