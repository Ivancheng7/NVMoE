# 新默认帧下的 attention 拆分（2026-09-17）

## 结论

在**新默认配置**（固定参数主机常驻 8 GiB 开、prompt chunk 开）下用命令行走了一次 8 步、真实
prompt（token 0）的运行，8 个 token 从 CLI 返回（`generated_ids = [55222, 33832, 42549, 42549,
42549, 42549, 42549, 127384]`，`stop_reason = max_new_tokens`），四个钉死值逐项复现。结论两条：

1. **attention 在新帧里是每 token `1.008 s`，占一层循环 `3.78 s` 的 `26.6%`**——和 `fixed`（`1.01 s`）、
   `routed`（`0.96 s`）并列第一，是这四项里唯一一直没人动过的。它的占比从交付帧的 `15–17%` 涨到
   `26.6%`，不是它变慢，是别的项被优化掉了。
2. **它的构成翻转了**：`compressor 0.288 s + indexer 0.131 s = 0.42 s`（占 attention 的 42%），
   而**另外 16 个小步合计 `0.59 s`（占 58%）**。同在遥测开着的条件下对比 9-16 那次非常驻帧
   （compressor `0.689 s`、indexer `0.219 s`、小步 `0.55 s`），掉的是 compressor/indexer 这两项
   （Csa4 层每次 `23.35 → 7.03 ms`，indexer `10.43 → 6.25 ms`），小步没动。

由此**重估** 9-16 那条"把 compressor 权重的 bf16→f32 展开挪到设备侧可省 0.9 s"：那个估计的分母
（compressor+indexer = 0.91 s）在常驻默认下只剩 `0.42 s`，所以它的上限掉到 `≤0.4 s`，
而 attention 里现在更该先动的是那 16 个小步的同步结构。**不是结论、只是记录**：这条改动还没做，
收益上限也没实测。

**这次没有声称的部分**（别照着脑补）：

- **没插桩**，所以"常驻为什么让它快 3 倍"只有推测（compressor 那次调用的主机侧展开/拷贝原来在和
  固定流的每层 H2D 抢主机带宽；常驻之后主机安静了），没有直接证据。
- **每个小步内部没拆**：只有 `sync_wall_ns ≈ gpu_event_ns`（例如 compressor `288.2` vs `284.9 ms`、
  mhc `70.7` vs `67.6`）说明每步都是主机等 GPU 的串行步，分不出 H2D / kernel / D2H / 同步各占多少。
- **本次带遥测**（`COLIBRI_V4_BENCH_TELEMETRY=1`），它会加同步点让计时略慢（历史量级约 1%），
  所以墙钟不能当基线数字，拆分比例可以用。不带遥测的同一配置墙钟另有记录（`defaults-pc-8step`，
  整请求 `42.41 s / 8 步`）。
- **服务路径没量**：serve 不写 `phase_profile`，所以"每字 `4.52 s`"那个帧里 attention 占多少不知道。
- **没有改任何推理代码**，所以没有新的数值行为，也没有新的正确性结论。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- `48` 个 safetensors，合计 `166886535336` 字节（整个目录 `166898773853` 字节）
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
  （其 metadata `total_size = 166878536440`）
- 源码树：`/srv/localmoe/build/merged-prompt-chunk`，HEAD `598958e`（工作区干净，无未提交改动）
- 二进制：`/srv/localmoe/targets/merged-prompt-chunk-target/release/colibri-cuda`
  （`16488976` 字节，构建于 `2026-09-17 09:46:53 +0800`，SHA-256
  `3a90da8abc6607b70329735cb65e126e846cb6a028c32bd733b5c03eac255b1c`）
- cuBLAS：`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`
- 机器：WSL2、RTX 5060 Ti 16 GB（`sm_120`）、本次运行前机器刚重启约 3 分钟、无其它引擎进程
  （`load average 0.34`）

**身份说明**：这次没有走 `rust/run-identified.sh`，所以结果文件里**没有源码指纹字段**
（`COLIBRI_SOURCE_IDENTITY`）。可对上的身份是上面那组：源码树 + HEAD + 二进制 SHA-256 + 构建时间。

## 运行证据

命令（与同配置的 `defaults-pc-8step` 逐字相同，**只多** `COLIBRI_V4_BENCH_TELEMETRY=1`）：

```bash
cd /srv/localmoe/build/merged-prompt-chunk
/usr/bin/time -v env \
  COLIBRI_V4_FLASH_BASE=1 \
  COLIBRI_V4_FLASH_TOKEN_IDS=0 \
  COLIBRI_V4_GREEDY_STEPS=8 \
  COLIBRI_V4_STREAM_READ_MODE=buffered \
  COLIBRI_V4_FLASH_PREFETCH=1 \
  COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas \
  COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas \
  COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice \
  COLIBRI_V4_BENCH_TELEMETRY=1 \
  COLIBRI_DSV4_MODEL_DIR=/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  COLIBRI_CUBLAS_EXACT_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5 \
  COLIBRI_V4_RESULT_JSONL=/srv/localmoe/targets/attn-split-pc-8step.jsonl \
  /srv/localmoe/targets/merged-prompt-chunk-target/release/colibri-cuda
```

结果文件（WSL 路径，不在仓库里）：`/srv/localmoe/targets/attn-split-pc-8step.jsonl`

`/usr/bin/time -v`：wall `0:46.87`、`Maximum resident set size = 7268616 kB`、
`File system inputs = 44322904`（512 B 块）、`Exit status 0`。

钉死值（`.zcode/verify_baseline.py`，逐项 OK）：`argmax = 55222`、`generated_ids[0] = 55222`、
`hidden digest = 5ed8bf88beae3cfe`、43 层、`fp4_accum = block32`、
`final_head = canonical_cpu_streamed_bf16`、`stop_reason = max_new_tokens`、
峰值显存 `1500053504`；`request_e2e_ns = 44370317250`。

顶层相位（`phase_profile.layers`，每个 token 一层循环的墙，单位 s；脚本 `.zcode/attention_share.py`）：

| token | outer | fixed | routed | shared | attention | router |
| --- | --- | --- | --- | --- | --- | --- |
| 0（分块 prefill 那一趟） | 13.35 | 9.46 | 1.02 | 1.15 | 1.57 | 0.12 |
| 1 | 3.92 | 1.07 | 0.95 | 0.85 | 0.89 | 0.11 |
| 2 | 3.79 | 1.00 | 0.95 | 0.80 | 0.89 | 0.11 |
| 3 | 3.81 | 1.00 | 0.96 | 0.77 | 0.94 | 0.11 |
| 4 | 3.64 | 1.03 | 0.96 | 0.51 | 0.97 | 0.12 |
| 5 | 3.69 | 1.03 | 0.96 | 0.56 | 0.97 | 0.12 |
| 6 | 3.60 | 1.02 | 0.96 | 0.47 | 0.97 | 0.13 |
| 7 | 3.78 | 1.01 | 0.96 | 0.65 | 1.01 | 0.12 |

末四个 token 的 attention 均值 `0.981 s`、占比均值 `26.7%`。

attention 子步（token 7，`.zcode/attention_split.py`；`sum of steps = 1.000 s` vs 墙 `1.008 s`）：

| 区域 | 子步 | 合计 ms | 每被调层 ms | 层数 |
| --- | --- | --- | --- | --- |
| 入口 | `mhc` | 70.7 | 1.64 | 43 |
| q 路 | `attn_norm` | 17.2 | 0.40 | 43 |
| | `q_a_projection` | 50.7 | 1.18 | 43 |
| | `q_a_norm` | 20.4 | 0.47 | 43 |
| | `q_b_projection` | 59.7 | 1.39 | 43 |
| | `q_b_norm` | 16.0 | 0.37 | 43 |
| | `q_rope` | 17.2 | 0.40 | 43 |
| kv 路 | `kv_projection` | 50.4 | 1.17 | 43 |
| | `kv_norm` | 14.8 | 0.34 | 43 |
| | `kv_rope_quantize` | 43.6 | 1.01 | 43 |
| **压缩 / 索引** | **`compressor`** | **288.2** | **7.03** | **41** |
| | **`indexer`** | **131.2** | **6.25** | **21** |
| 选择 / 提交 | `commit_topk` | 59.9 | 1.39 | 43 |
| | `sparse_score` | 29.7 | 0.69 | 43 |
| | `sparse_inverse_rope` | 15.9 | 0.37 | 43 |
| 输出路 | `wo_a` | 35.7 | 0.83 | 43 |
| | `wo_b` | 57.4 | 1.33 | 43 |
| | `hc_post` | 21.2 | 0.49 | 43 |

分区域汇总（同上一次运行的 token 7，合计 `999.9 ms`）：

| 区域 | 合计 ms | 占 attention |
| --- | --- | --- |
| 压缩 / 索引（compressor 288.2 + indexer 131.2） | 419.4 | 41.9% |
| q 路（6 步） | 181.2 | 18.1% |
| 输出路（wo_a / wo_b / hc_post） | 114.3 | 11.4% |
| kv 路（3 步） | 108.8 | 10.9% |
| 选择 / 提交（commit_topk / sparse_score / sparse_inverse_rope） | 105.5 | 10.6% |
| 入口 mhc | 70.7 | 7.1% |

对照：非常驻帧那次（`/srv/localmoe/targets/bench-telemetry-8step.jsonl`，9-16 深夜）的同两张表是
attention `1.462 s`、compressor `689.2 ms`（Csa4 层每次 `23.35 ms`、Hca128 `9.94 ms`）、
indexer `219.1 ms`（每次 `10.43 ms`）、其余 16 步 `547 ms`。两次都是遥测开，差别只在配置
（那次常驻关、无 chunk），所以这次对比里的 `23.35 → 7.03 ms` 归因于常驻默认。

层型分布（token 7）：layer 0/1 是 window-only（compressor/indexer 读数 0，整层 attention `10.9–13.4 ms`）；
带 indexer 的 21 层每层 `28–45 ms`；只有 compressor 的 20 层每层 `16–24 ms`。

## 代码变更

**没有改推理代码。** 本次为纯测量，新增两个读账脚本（未跟踪，不进库，从 WSL 侧用 `/mnt/c/...` 路径跑）：

- `.zcode/attention_split.py`：按 token 打 attention 子步表（合计、gpu_event、每被调层、层数、逐层明细）
- `.zcode/attention_share.py`：按 token 打顶层相位表与 attention 占比，并打印一个负载代理
  （最后一次 final head 的权重读时间）

## 未决事项

1. **常驻为什么让 compressor 快 3 倍（`23.35 → 7.03 ms/次`）。** 现在只有推测。要证实得在
   `run_layer_compressor` / `run_compressor_official_cublas_host` 内部插桩，把
   D2H 权重、主机 bf16→f32 展开、H2D 权重、SGEMM 四段分开计时（一次 1 步运行即可）。
2. **16 个小步的 `0.59 s/token`（占 attention 58%）怎么削。** 每步都是一次 H2D + 小 kernel + D2H + 同步，
   一层 18 个同步点。两条路：先量单步内部分解（同第 1 条的办法），或者直接试解码侧的设备 attention
   （`COLIBRI_V4_DEVICE_ONE_TOKEN_DECODE` + `COLIBRI_V4_DEVICE_ATTENTION_PRODUCTION`）——
   它现在被门槛挡着（只允许 2–5 步、≤9 行），且文档里没有任何实测数字。
3. **设备侧展开那个改动的收益上限重估到 `≤0.4 s`。** 动手前先在当前默认帧确认它还剩多少，
   别按 9-16 那条 `0.9 s` 的旧估计派活。
4. **服务路径的 attention 占比没量。** serve 不写 `phase_profile`；要这个数得先让服务把那块也写出来。
5. **新默认帧的正式墙钟基线**用不带遥测的 `defaults-pc-8step`（`42.41 s / 8 步`）；
   本记录只用它的拆分比例，别把 `44.37 s` 当基线。

## 追加：压缩器插桩 + 一个 A/B（同日，探针构建）

**插桩**：把 `layer_ops.rs` 的 `run_compressor_official_cublas_from_device`（权重 D2H）与
`run_compressor_official_cublas_host`（主机展开 / 权重 H2D / 工作区分配 / 逐行 / SGEMM / 尾部）分成
六段计时，每次调用打一行到 stderr。打在**另外复制的一份树**上，不碰任何会话的工作区：

- 源码 `/srv/localmoe/build/attn-probe`（`merged-prompt-chunk` 的副本，HEAD `598958e`）
- 目标目录 `/srv/localmoe/targets/attn-probe-target`
- 打补丁脚本 `.zcode/compressor_probe_patch.py`、`.zcode/compressor_probe_patch2.py`（未跟踪，不进库）
- **这份二进制是插桩构建，不是基线，别拿它出交付数字**

**六段分解**（1 步运行 `/srv/localmoe/targets/attn-probe-1step.jsonl`，62 次调用；中位数，单位 ms）：

| 层型 | 权重 D2H | 主机展开 | 权重 H2D | 工作区+零缓冲 | 逐行 | SGEMM | 单次合计 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Csa4（21 次） | 1.49 | **15.07** | 4.22 | 1.17 | 0.57 | **0.06** | 25.03 |
| Hca128（20 次） | 0.82 | 0.73 | 2.46 | 1.05 | 0.46 | 0.05 | 5.82 |
| indexer（21 次） | 0.50 | 0.35 | 1.58 | 1.00 | 0.45 | 0.05 | 3.98 |

三个读数：

- **算力不是项。** SGEMM 段中位 `0.05–0.06 ms`，占单次调用的 `0.2–0.3%`；而且这一段量的是**入队**时间，
  GPU 真正的执行时间藏在后面那次 D2H 同步里，逐行段（`0.45–0.57 ms`）是它的上界。所以
  1024×7168 的 FP32 GEMV 在这条路上最多占 `~3%`。**在 attention 这条路上"优化算子"没有空间。**
- **主机的重新申请才是大项。** 同一规模的工作在 21 次调用里从 `0.6 ms` 跳到 `13–17 ms`（双峰），
  典型的新页缺页指纹：`to_f32` 每次调 `Vec::collect` 申请两块全新的 29 MB（Csa4）/14.7 MB（Hca128），
  D2H 的 `to_host_vec` 同样。Csa4 的展开段 `min 1.27 / med 15.07 / max 26.44 ms`。
- 与 8 步帧的相位账对齐：探针的 62 次调用合计在 1 步（冷）是 `701 ms`，token 0 的相位账
  compressor+indexer 是 `927.7 ms`（含 16 个小步之外的同一步），量级一致。

**A/B（同一棵树，只把权重展开改成复用线程本地缓冲，其余不动）**：

| | 1 步（冷） | 8 步末四 token 平均 |
| --- | --- | --- |
| 每次新申请（原样） | `701.0 ms` | `420.5 ms` |
| 复用缓冲 | `444.3 ms`（`-257`） | `369.4 ms`（`-51`） |

**数值零变化**：8 步那跑与交付参照逐项 `IDENTICAL`（8 个 hidden digest、8 条 `ordered_layer_digest_sha256`
链、8 个 argmax、峰值显存 `1500053504`），用的是 `.zcode/outputs_match.py`
（`/srv/localmoe/targets/attn-probe2-8step.jsonl` 对 `defaults-pc-8step.jsonl`）。
1 步那次冷帧的大收益（`-257 ms`）主要来自"页还没碰过"，8 步热帧里那部分早被吃掉了，
所以**该引用的是 `-51 ms/token`（约 12%）**，别引 1 步那个数。

**修法优先级因此改序**（都是当前默认帧、每 token）：

1. **复用主机缓冲：`-51 ms`，已实测，数值逐位不变。** 仓库里现成的 `StrictCompressorArena`
   就是干这个的（它持有可复用的 `wkv_f32`/`wgate_f32`/`input_f32`/设备缓冲/工作区），
   只是交付解码路径（`full61_b1.rs`）给 `run_layer_compressor` 传的是 `None`。
   这次 A/B 用的是线程本地缓冲的探针版本，等价效果；正式改法应该走 arena。
2. **设备侧展开：上限是权重往返那部分拷贝**（Csa4 每次 D2H 1.5 + H2D 4.2 ms，
   三类合计约 `230 ms/token`）——比复用大，但要改数值路径之外的东西（新的 bf16→f32 kernel + 常驻 f32 缓冲），
   排在复用之后做。
3. **别动 SGEMM / 别用算子优化去换 attention 的收益**：实测算术只占 0.3%，换了也看不见。

**还没拆的**：`mhc`（`1.64 ms/层`、`70 ms/token`，是 16 个小步里最大的一步）走的是设备侧 cuBLAS 臂、
没有主机往返，却要 1.64 ms，里面是什么没看过；下一个探针应该打它。

