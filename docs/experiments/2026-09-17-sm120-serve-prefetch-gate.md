# 服务面的预取门卫：从只认 BASE 放开到 BASE 或 SERVE（2026-09-17）

## 结论

把 `flash_b1_prefetch_enabled` 的**面门槛**从"只认 `COLIBRI_V4_FLASH_BASE`"改成
"`BASE` 或 `SERVE` 都放行"，并把内层的 `COLIBRI_V4_FLASH_PREFETCH` 在这两个交付面上
**默认开**（`=0` 回到串行加载）。改的是一个函数（`b2_runtime.rs`），提交 `4cda82b`。

同一个二进制、同一批里的服务 A/B（prompt「你好，你是谁？」→ 8 个字）：

| 服务 | 首字 TTFT | 每字 ITL | 整请求 |
| --- | --- | --- | --- |
| 默认（预取开）`on` / `on2` | `30.40` / `25.71 s` | `4.815` / `4.562 s` | `66.1` / `59.6 s` |
| `COLIBRI_V4_FLASH_PREFETCH=0`：`off` / `off2` | `86.12` / `82.90 s` | `10.445` / `10.479 s` | `160.8` / `157.8 s` |

**四次运行的 token id 完全相同**：`58572,42549,42549,109865,42549,42549,42549,42549`，
和槽位复用之前那个二进制跑出来的也一样（跨二进制锚还在）。也就是说：每字约 `2.2 倍`、
首字约 `3.0 倍`，数值没动。

CLI 8 步（同一个二进制）也给了一组同批对照：默认 `47.04 s`、`=0` `96.07 s`（约 `2 倍`）。
两臂的 8 个 `hidden_bf16_digest`、8 条 `ordered_layer_digest_sha256` 链、8 个 argmax
都与参照 `defaults-pc-8step.jsonl` `IDENTICAL`，默认臂的峰值显存 `1500053504` 也和参照一致。

**没声称的部分**（别照着脑补）：

- **只放宽门槛是不够的**，交接说明里那句"门里 `BASE` 或 `SERVE` 都放行"漏了一半：
  门里还有 `optional_one_flag("COLIBRI_V4_FLASH_PREFETCH")`，而它在变量缺席时返回 `false`；
  `c/openai_server.py:1738` 只设 `FLASH_SERVE`、**不设 `PREFETCH`**（只有测试客户端
  `flash_serve_client.py:52` 设）。只改面门槛的话，交付服务路径仍然是慢的那一档。
  所以这次两件一起改，这正是"改法和 chunk 当初一样"。
- **服务是拿协议客户端跑的**（`flash_serve_client.py` 去掉强制 `PREFETCH=1` 那一行的副本），
  **不是 `c/openai_server.py`**：这台机器上 `c/` 编不出来（`nvcc` 解析不过 gcc 15，
  见 `2026-09-17-sm120-fixed-h2d-bandwidth.md` 记的那条弯路），`/srv/localmoe` 下
  没有任何 `c/colibri` 二进制，openai_server 的 flash 模式起不来。所以"交付服务入口"
  这次**只有代码保证**（不设也会开），没有真起过一次服务。
- **服务的绝对秒数不能跨批比较**：本批 `off` 是 `86.12 / 82.90 s` 首字、
  `10.44 / 10.48 s` 每字；交接说明里那批的同一配置是 `79.47 / 80.24 s` 与 `10.03 / 10.00 s`。
  同一个配置跨批差 6% 以上，所以这里只用**同批的开关之比**，别把 `86.12` 当成"旧默认值"。
- **`=0` 臂不是历史的逐位还原**：CLI 那跑 8 个 digest 全同，但峰值显存是
  `1466499072`，而默认臂和参照都是 `1500053504`。这 `33.6 MB` 的设备分配差在哪、
  为什么跟着这个开关动，**这次没查**（见未决事项 2）。
- **没有重测 `routed` 那条读盘的账**，也没有解释"预取一开就快一倍"里有多少来自并行读盘、
  多少来自固定层的宿主预取：两条路挂在同一个门后面（`layer_rows.rs:958` 的
  `flash_routed_prefetch` 与 `generate.rs:589` 的下一层预取），这次只量了门的开关，没有拆。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- `48` 个 safetensors，合计 `166886535336` 字节（整个目录 `166898773853` 字节）
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`

（与 `2026-09-17-sm120-attention-split-new-default.md` 是同一份权重，那篇里的字节数与
文件数就是从这里抄的。）

**身份说明**：这次也没有走 `rust/run-identified.sh`，结果文件里没有源码指纹字段。
可对上的身份是下面这组：源码树 + HEAD + 二进制 SHA-256 + 构建时间。

## 运行证据

- 源码树：`/srv/localmoe/build/merged-prompt-chunk`，HEAD `4cda82b`（detached，工作区干净）
- 构建：`cargo oxide build --arch sm_120 -- --release`，
  `CARGO_TARGET_DIR=/srv/localmoe/targets/merged-prompt-chunk-target`，
  `RUSTC_WRAPPER=rust/rustc-human-wrapper.sh`
- 二进制：`/srv/localmoe/targets/merged-prompt-chunk-target/release/colibri-cuda`
  （`16487968` 字节，构建于 `2026-09-17 13:55`，SHA-256
  `5d20f223c85327ca9471b2a7dda9238a454a7d85e4fc72df8790d626a4f24f9c`）
- cuBLAS：`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`
- 机器：WSL2、RTX 5060 Ti 16 GB（`sm_120`）、跑之前 `ps -eo comm | grep colibri-cuda` 为 `0`（没有并发引擎）

CLI 8 步（`default` 臂**故意不设** `COLIBRI_V4_FLASH_PREFETCH`，用来测新默认；
`prefetchoff` 臂设 `=0`）：

```bash
cd /srv/localmoe/build/merged-prompt-chunk
/usr/bin/time -v env \
  COLIBRI_V4_FLASH_BASE=1 \
  COLIBRI_V4_FLASH_TOKEN_IDS=0 \
  COLIBRI_V4_GREEDY_STEPS=8 \
  COLIBRI_V4_STREAM_READ_MODE=buffered \
  COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas \
  COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas \
  COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice \
  COLIBRI_DSV4_MODEL_DIR=/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  COLIBRI_CUBLAS_EXACT_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5 \
  COLIBRI_V4_RESULT_JSONL=/srv/localmoe/targets/prefetch-gate-cli/default.jsonl \
  /srv/localmoe/targets/merged-prompt-chunk-target/release/colibri-cuda
```

（`prefetchoff` 那一跑逐字相同，只多 `COLIBRI_V4_FLASH_PREFETCH=0`，结果文件换成
`prefetchoff.jsonl`。两个名字都是新的，没有覆盖任何旧文件。）

| CLI 臂 | `/usr/bin/time -v` wall | `request_e2e_ns` | 峰值显存 | Max RSS |
| --- | --- | --- | --- | --- |
| `default`（预取开） | `0:49.30` | `47.04 s` | `1500053504` | `7257320 kB` |
| `prefetchoff`（`=0`） | `1:38.67` | `96.07 s` | `1466499072` | `7134952 kB` |

服务 A/B（同一个二进制，客户端副本去掉了强制 `PREFETCH=1`）：

```bash
# .zcode/serve_ab.sh 里做的是：sed '/COLIBRI_V4_FLASH_PREFETCH/d' flash_serve_client.py > /tmp/serve_client_no_prefetch.py
python3 /tmp/serve_client_no_prefetch.py \
  --binary /srv/localmoe/targets/merged-prompt-chunk-target/release/colibri-cuda \
  --model-dir /srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --cublas-lib /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5 \
  --prompt-json '"\u4f60\u597d\uff0c\u4f60\u662f\u8c01\uff1f"' \
  --max-tokens 8 --repeat 1
```

四跑按 `on off off2 on2` 的顺序（两种顺序都排到了），每跑都是一个新引擎进程：

| 跑 | 环境 | STAT 里的 TTFT | STAT 里的 ITL | 客户端 wall |
| --- | --- | --- | --- | --- |
| `on` | 不设 `PREFETCH` | `30401.883 ms` | `4814.630 ms` | `66.1 s` |
| `off` | `=0` | `86124.629 ms` | `10444.525 ms` | `160.8 s` |
| `off2` | `=0` | `82902.558 ms` | `10478.900 ms` | `157.8 s` |
| `on2` | 不设 `PREFETCH` | `25710.152 ms` | `4562.399 ms` | `59.6 s` |

结果文件（都是 WSL 路径，不在仓库里）：

- `/srv/localmoe/targets/prefetch-gate-cli/default.jsonl`、`prefetchoff.jsonl`（+ 同名 `.stdout` / `.stderr`）
- `/srv/localmoe/targets/prefetch-gate-serve/serve-{on,off,off2,on2}.log`

判定用的脚本（`.zcode/`，未跟踪）：`build_delivery.sh`、`run_cli.sh`、`serve_ab.sh`、
`check_cli.sh`（包 `verify_baseline.py` 与 `outputs_match.py`）。
`verify_baseline.py` 对 `default.jsonl` 打 7 项全 `OK`；`outputs_match.py` 对
`defaults-pc-8step.jsonl` 判 `IDENTICAL`，对 `prefetchoff.jsonl` 判
`DIFFERS: peak device bytes`。

## 代码变更

只改了一个函数：`rust/colibri-cuda/src/b2_runtime.rs::flash_b1_prefetch_enabled`
（提交 `4cda82b flash: let the serve surface keep the prefetch worker`）。

改之前它要求 `spec == FLASH_0731 && requests == 1 && optional_one_flag(BASE)`，
然后返回 `optional_one_flag(PREFETCH)`——也就是**只有 `FLASH_BASE=1` 且显式 `PREFETCH=1`**
才有那个 worker。改之后：面门槛是 `BASE` **或** `SERVE`，`PREFETCH` 在这两个面上
不设即开、`=0` 关。形状照抄旁边的 `flash_b1_prompt_chunk_enabled`。

顺带核过、写在这里免得下次再查：`COLIBRI_V4_FLASH_BASE` 在服务路径上**只**管这一件事——
`strict_compressor_enabled`（`layer_ops.rs:1739`）本来就把 `SERVE` 算进去，
`main.rs:245` 的服务分支排在 base 分支之前，所以放开这个门能复现那次测量、没有第二种副作用。
`rust/colibri-cuda/src/tests.rs` 里没有任何用例读这个环境变量，不需要改测试。

## 未决事项

1. **服务的相位账还没有**：serve 不写 `phase_profile`，所以"默认每字 `4.7 s`"里
   `fixed / attention / routed / shared` 各占多少目前是黑的。上面第 1 件做完了，
   服务就是主要交付面，得让它有自己的账（要么让服务写 `phase_profile`，
   要么把 `flash_b1_phase` 已有的字段打出来）。
2. **`=0` 臂的峰值差没查**：CLI 两臂 digest 全同，但峰值 `1500053504`（默认）
   对 `1466499072`（`=0`），差 `33.6 MB` 的设备分配。怀疑与 `layer_rows.rs:958`
   那条"`flash_routed_prefetch = 预取开 && expert_slots.is_none()`"的两条分支有关
   （一条走线程并行读盘、一条走 routed 槽位），**但这只是怀疑，没有证据**。
   要和固定层设备槽位一起看，因为两者都动这块的显存。
3. **`c/openai_server.py` 那条路没有真跑过**：现在只有"环境里不设 `PREFETCH` 也会开"的
   代码保证。要真验得先有 `c/colibri`，而这台机器上 `c/` 编不出来。
4. **预取到底省了什么没拆**：门后挂着两件事（并行读盘、下一层宿主预取），
   这次只量了门的开关。要拆得在 `layer_rows.rs:958` 两个分支上分别计时。
5. 下一步的排期在 `.zcode/plans/plan-2026-09-17-next-optimizations.md`（未跟踪）：
   接着做固定层设备槽位移植（`−0.31 s/token`、`+216 MB`）和压缩器主机缓冲复用
   （`−51 ms/token`），再做 `shared` 的 0.65 s 到底是谁的那次测量。
