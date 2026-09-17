# 连续 8 token 与服务路径首次真机端到端（2026-09-16）

## 结论

这条链今天往前走了一步半。

**连续 8 个 token 从 CLI 出。** 当前 HEAD 的二进制跑 `COLIBRI_V4_FLASH_BASE=1` + `COLIBRI_V4_GREEDY_STEPS=8`，
生成 `[55222, 33832, 42549, 42549, 42549, 42549, 42549, 127384]`，整个请求 `107.9 s`，
峰值显存 `1466499072` 字节（1.466 GB）。**峰值显存和单步跑完全一样**，不随步数增长；第一颗 token 的
`hidden_bf16_digest` = `5ed8bf88beae3cfe`，与单步基线逐位相同。单步耗时 12.2–15.0 s（第一颗含预填充）。

顺带纠正一条旧印象：2026-08-28 记录里"3 步峰值 3.72 GB"是那次用 `gpu_exact` final head 的开销，
不是每步泄漏——默认 CPU final head 下 1 步和 8 步的峰值一模一样。

### 和 README 招牌数字的关系

README 的 12GB 笔记本成绩是 **`0.0762 tok/s`**，也就是 `13.126173804 s` 一个 one-token、43 层请求
（`docs/assets/readme/flash-0731-abba.csv` 里八行的最终 F 臂：`12.93 s` / `13.33 s`，其余臂
`31.5 / 32.5 / 34.3 / 34.9 / 93.2 / 93.5 s`）。

和我们这次在 5060 Ti 上的实测**基本一致**：单步 `12.9 s`、8 步里每个 token `12.2–13.4 s`，
即 `0.075–0.082 tok/s`。峰值显存也一致（他们最终 `1.35 GiB`，我们 `1.37 GiB`）。
**换句话说：这台桌面卡并不比那台游戏本快，因为两边卡住的都是同一条权重流路径，不是显卡。**

README 自己把范围写清楚了（第 46 行）：那份 43/43 证书pin在"one prompt token, one greedy step"上，
long contexts / other prompts 需要各自的证据。它没说、也没有测的是：**成本是 prompt 长度的线性函数**——
README 的运行命令用的是 `COLIBRI_V4_FLASH_TOKEN_IDS=0`，prompt 就是一个 token，正好落在看不见这个问题的区间。

README 第 60–65 行还有一张标着"sizing estimates with no benchmark-run evidence yet"的表，
给 16GB desktop 估 `0.209 to 0.440 tok/s`，假设是 `fixed-resident`。我们实测 `0.078 tok/s`，
差 2.7–5.7 倍；差距的来源正是那条假设——**固定参数常驻显存**（见上面"12 秒去哪儿了"一节，
现在没常驻，每 token 重读 6.71 GB）。

### 12 秒去哪儿了：引擎自己记的账

8 步那次运行的结果文件里有引擎自己的字节台账和逐相位计时，把它们加总（8 个迭代 = 1 个 prompt
forward + 7 个生成 forward，344 次层访问）之后，一张 token 的成本是这样：

| 项 | 每次迭代 | 说明 |
| --- | --- | --- |
| 从 SSD 读固定参数 | **6.71 GB** | 43 层 × 156 MB（attention + shared + 各类 norm），每个 token 重读一遍 |
| 从 SSD 读 routed 专家 | **3.45 GB** | 每层 6 个专家 × 13.37 MB，只读被选中的 |
| 从 SSD 读 final head | **1.06 GB** | `129280 × 4096` 的 bf16 head，每个 token 读一次，而且在 CPU 上算 |
| 合计 | **11.2 GB/token** | 8 个迭代共 `89.75 GB`（`physical` 台账） |
| 峰值显存 | 1.466 GB | 1 步和 8 步**逐字节相同**：权重不在显存里，KV/状态是预分配的 |
| H2D 过 PCIe | ~10.2 GB/token | `fixed_h2d` 53.67 GB + `routed_h2d` 27.59 GB，与 SSD 读入字节数 1:1，没有复用 |

**有效读取速率约 `0.83 GB/s`**（89.75 GB / 107.92 s）。分项看：固定参数那支
`read_alloc_ns` 累计 55.96 s 读 53.67 GB（约 **0.96 GB/s**）；routed 那支整个 `wall` 49.88 s
读 27.59 GB（约 **0.55 GB/s**，其中 `setup_ns` 29.18 s 是专家槽位准备，比真正读的 14.97 s 还大）。

**计算不是瓶颈**：routed 专家的 kernel 全程 3.65 s（约 10.6 ms/层），attention 执行 10.47 s
（约 30 ms/层）。显卡大部分时间在等数据。

**同一台机器上顺序读能跑多快**：final head 那 1.06 GB，第一个 token 冷读用 642 ms（**1.65 GB/s**），
之后的 token 只要 47 ms（**22.5 GB/s**，显然命中了页缓存）。所以盘本身不止 0.83 GB/s——
**0.83 GB/s 是我们这条流式读路径跑出来的，不是盘的物理上限**。盘的上限还没测（仓库里有
`c/iobench.c` 和 `c/tools/v4_io_matrix.py` 可以直接用）。

**一个直接可动的杠杆**：固定参数总共 43 × 156 MB = **6.7 GB，16 GB 的卡装得下**。现在它们每个 token
都重读一遍；钉进显存的话，每 token 的 SSD 流量从 11.2 GB 掉到约 4.5 GB。`fixed_layer_cache.rs` /
`fixed_prefetch.rs` / QD4 那套机器就是干这个的，现在只有部分生效。

**服务路径（stdio 协议）第一次真机端到端跑通。** 真实 prompt 文本进（`你好，你是谁？`），
走 tokenizer + chat 模板 → 生成 8 个 token → 解码文本按 `DATA` 帧流式返回，最后 `DONE` 带统计。
同一个进程里连着两次相同请求，返回的 token id 完全一致；另一个进程用同一 prompt 再跑，首 token 也一致。

**但它慢，而且慢的地方在权重流，不在显卡也不在 KV cache。** 第一个 token 要 95–157 s；
`--max-tokens 1` 也要 95 s，说明和生成长度无关。真正的原因是**没有批量预填充**：耗时按 prompt 长度
线性涨，约 19 s 一个 prompt token（四个数据点见下节），而一次完整的 43 层单步是 12–15 s。
CLI 那条之所以看着"只是 13 s/步"，是因为它的 prompt 只有 1 个 token，看不见这一段。

单步的 12 秒也不是"读完 167 GB 权重"——引擎台账说每个 token 只读约 11.2 GB，
**有效速率约 0.83 GB/s**，而同一台机器顺序读能跑 22.5 GB/s。也就是说：既不是显卡算得慢
（routed 专家 kernel 全程 3.65 s），也不是 KV/状态爆显存（1 步和 8 步峰值逐字节相同），
**是这条"SSD → host RAM → PCIe → 显存"的流式读路径本身慢**，而且 prompt 里的每个 token 都要再走一遍。

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

1. **没有批量预填充：prompt 每多一个 token 就多读约 11.2 GB。** 证据在上面两节（四个耗时数据点连成
   斜率约 19 s/token 的直线；循环每迭代只吃一个 token）。后果：5 个 token 的 prompt 忍得了，
   200 个 token 的真实对话要在第一个字之前等约 2.2 TB 的读。下一步不是"优化循环"，而是把 serve
   路径接到已有的预填充实现上——仓库里本来就有 `layer_stationary_prefill.rs` 那条路。
2. **读路径的有效速率（0.83 GB/s）和盘的上限差多少，没测。** 已知的锚点：同一份 final head
   冷读 1.65 GB/s、页缓存命中 22.5 GB/s。要用 `c/iobench.c` / `c/tools/v4_io_matrix.py`
   在模型目录上跑一次原始读，才知道 0.83 是"盘就这速度"还是"读法有问题"。
3. **固定参数 6.7 GB 本可以常驻显存，现在是每个 token 重读。** 43 层 × 156 MB，16 GB 的卡装得下
   （现在峰值只用 1.466 GB）。这一刀能把每 token 的 SSD 流量砍掉约 60%。
4. **`routed.setup_ns` 29.18 s 是纯软件开销**（专家槽位准备，比真正读数据的 14.97 s 还大），
   占整个 8 步运行的 27%。没有定位到具体是谁在花。
5. **serve 每迭代比 CLI 贵约 1.75 倍（21.1 s vs 12.0 s），没解释。** 两个路径调的是同一个
   `run_v4_b2_generate_arm`，差别只有 prompt 长度和 `record_final_head_loader` /
   `capture_layer_trace` 两个臂参数（CLI 传 `true, true`，serve 传 `false, false`）。
   **2026-09-17 更正：已定位。** 是固定参数的后台预取线程没在服务路径上开——
   `flash_b1_prefetch_enabled` 要求 `COLIBRI_V4_FLASH_BASE`，而服务只设 `FLASH_SERVE`。
   实测也设上 `FLASH_BASE` 之后，serve 每个字 `9.87 → 4.52 s`、首字 `79.9 → 27.1 s`。
   见 `2026-09-16-sm120-prompt-chunk.md` 的"服务路径一直白白慢一倍"一节。
6. **OpenAI HTTP 网关能不能驱动 Rust 引擎，没验证。** `c/openai_server.py` 的 `Engine` 起的是 C 引擎、
   用 C 引擎的环境变量，还解析 `TIERS`/`hwinfo` 之类的握手行；要对上得先确认 Rust 侧发不发这些。
7. **取消、并发、多请求没测。** 协议里有 `CANCEL`，`validate_submit` 只允许 `cache_slot=0`，
   也就是单槽串行。
8. **多 token 的正确性只有第一颗锚住了**（与单步基线逐位相同），第 2–8 颗只做了同一运行内的自比对，
   没有独立参考。
