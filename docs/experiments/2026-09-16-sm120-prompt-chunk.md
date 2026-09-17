# 整个 prompt 共用一趟层扫描：数值逐位不变，固定参数少读 4 趟（2026-09-16 深夜）

## 结论

预填充的每个 token 不再各走一遍 43 层。开关 `COLIBRI_V4_FLASH_PROMPT_CHUNK=1`
（**默认关**），只在交付的两个 Flash 面上生效（`COLIBRI_V4_FLASH_BASE` / `COLIBRI_V4_FLASH_SERVE`）。

**数值逐位不变，这是硬结论。** 5 个 token 的 prompt + 8 步生成，同一个二进制只差这个开关：

- 12 个 `hidden_bf16_digest`、12 个 `ordered_layer_digest_sha256`、`argmax` 列表、
  `generated_ids`、`routed_source_bytes`、峰值显存——**全部逐个相同**。

**固定参数的账是逐字节精确的，比墙钟硬得多。** 同样的 5-token prompt：

| | 关 | 开 | 差 |
| --- | --- | --- | --- |
| 固定参数层扫描次数 | `12.0` | **`8.0`** | `−4` |
| 进 SSD 的固定参数字节 | `80,507,390,304` | **`53,671,593,536`** | `−26,835,796,768` |
| 过 PCIe 的固定参数 H2D | `80,506,370,400` | **`53,670,913,600`** | `−26,835,456,800` |
| `phase_profile.layers` 条数 | `516`（12 × 43） | **`344`**（8 × 43） | |

一次 43 层扫描是 `6,708,949,192` 字节（1 步那次运行的 `fixed_source_bytes`）。
`26,835,796,768 ÷ 6,708,949,192 = 4.0` **整数**，H2D 那栏同样整数。
所以不是"大约省了 27 GB"，是**正好少读少传 4 趟**。

公式：扫描次数从 `prompt + 生成 − 1` 降到 `生成`，省 `prompt − 1` 趟。

**真实聊天那条路（serve）也量了，首字从 `158.5 s` 降到 `86.1 s`：−72.4 s，降 `46%`（快 1.84 倍）。**
同一个二进制、交替顺序四跑（关-开-开-关），四跑的 token id 逐个相同。
解码的每步间隔（ITL）四条都在 `18.2–19.4 s`，没动——这一刀只砍 prompt 那一段。

prompt 经 chat 模板是 8 个 token、生成 8 个，所以扫描次数 `15 → 8`，省 7 趟
= `46,962,644,344` 字节（4 趟整数倍之外多 3 趟，仍然是精确的 7 × 6,708,949,192）。
省下的字节除以省下的时间 = `0.65 GB/s`，与冷缓存下单线程读的实测速率同量级。

## 怎么改的

`run_v4_b2_generate_arm` 的循环从"外层一个 token 位置、内层 43 层"改成在 prompt 段上
"外层层、内层位置"：一个 prompt 的所有位置共用一次层权重加载。decode 段仍然一个位置一趟
（下一个 token 要等上一个的 argmax，串行不可避免）。

**为什么逐位安全**，三条已经存在的契约：

1. 每层 KV 行要求严格递增且 `committed_kv_rows() == start_pos`
   （`layer_rows.rs:659-668`、`v4_state.rs:1909-1916`）。同层内按位置 0,1,2… 顺序调，天然满足。
2. 层与层之间在状态上完全独立：`V4SerialPersistentState::transaction`
   （`v4_state.rs:2122-2140`）没有顺序约束，测试 `serial_persistent_state_is_lazy_independent_and_transactional`
   显式乱序访问 layer 0 和 2 并断言各自计数独立。
3. 位层内位置 p 的注意力只看同层 0..p-1 的行；那些行是由"上一轮整趟"提交还是"本层前几步"
   提交，**值完全一样**，所以浮点结果逐位相同。

仓库里本来就有同形状的先例：`prepare_two_token_prefix`（`b2_runtime.rs:272-338`）就是同一层
连续两次调用、位置 0 然后 1。

**没动算子。** `run_b2_layer` 的单行签名、`layer_rows.rs` 的 `rows > 2` 上限、attention /
compressor / routed 读全部原样，只是被调用的次数和顺序变了。

## 固定输入与模型

- 模型 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 权重 revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- RTX 5060 Ti 16 GB（`sm_120`）、cuBLAS `12.8.5.5`
- 分支 `exp/prompt-chunk`，提交 `719b5f2`；构建 `21.32 s`、exit 0、**59 条警告**（与基线相同）

## 运行证据

二进制 `/srv/localmoe/targets/prompt-chunk-target/release/colibri-cuda`。
结果文件都在 `/srv/localmoe/targets/`，日志在 `/srv/localmoe/targets/prompt-chunk-runs/`。
环境变量与交付跑一致（`FLASH_BASE=1`、`PREFETCH=1`、`MHC/ROUTER_LINEAR_ARM=cublas`、
`ROUTER_SCORE_ARM=libdevice`、`stream=buffered`；`FP4_ACCUM_ARM`/`FINAL_HEAD_ARM` 不设），
只多一个开关。

### 冒烟：prompt 只有 1 个 token（chunk 长度必然是 1）

| 文件 | 步数 | 墙钟 | 钉死值 |
| --- | --- | --- | --- |
| `promptchunk-off-1step.jsonl` | 1 | `12.72 s` | 四个全中 |
| `promptchunk-on-1step.jsonl` | 1 | `11.09 s` | 四个全中 |
| `promptchunk-off-8step.jsonl` | 8 | `93.66 s` | 8 个 digest 逐个相同 |
| `promptchunk-on-8step.jsonl` | 8 | `92.28 s` | 8 个 digest 逐个相同 |

单步那四个值：`argmax=55222`、`hidden_bf16_digest=5ed8bf88beae3cfe`、43 层、
峰值显存 `1466499072`。关和开两跑全部一致，`fixed_source_bytes` 也逐字节相同。

**这四跑做的是完全相同的功**（prompt=1 时 chunk 长度恒为 1），所以它们的墙钟差就是噪声：
1 步那对差 `12.8%`，8 步那对差 `1.5%`。

### 真测试：5 个 token 的 prompt + 8 步

`COLIBRI_V4_FLASH_TOKEN_IDS=58572,42549,42549,109865,42549`、`GREEDY_STEPS=8`
（`forward_tokens = 12`，正好在 128 的窗口内）。

- `promptchunk-off-5tok-8step.jsonl`：`138.8 s`，`chunk_positions = 1`
- `promptchunk-on-5tok-8step.jsonl`：`113.81 s`，`chunk_positions = 5`

逐位相同项：12 个 `hidden_bf16_digest`、12 个 `ordered_layer_digest_sha256`、
`argmax`（`65804, 73288, 73288, 60846, 60846, 60846, 60846, 62130`）、
`generated_ids`、`routed_source_bytes`（`41,391,489,024`）、峰值显存（`1466499072`）。
不同的只有墙钟和固定参数那两栏——那正是这次要改的东西。

**机制在记录里看得见**：`phase_profile.layers[].chunk_positions` 从 `1` 变成 `5`，
条数从 `516` 降到 `344`。

### 服务路径：真实 prompt、交替顺序四跑

跑这批时机器是空的（`nvidia-smi` 无 compute 进程、load average `0.00`），
但 WSL 刚重启过，所以**页缓存是冷的**。prompt 「你好，你是谁？」经 chat 模板是 8 个 token，
`--max-tokens 8`，每个 label 是一个全新的引擎进程 + 一次 SUBMIT：

| label | 顺序 | TTFT | ITL | token id |
| --- | --- | --- | --- | --- |
| `serve-off1` | 第 1 跑（最冷） | `163217.510 ms` | `18626.407 ms` | 见下 |
| `serve-on1` | 第 2 跑 | `87705.367 ms` | `18688.817 ms` | 同 |
| `serve-on2` | 第 3 跑 | `84523.523 ms` | `19447.178 ms` | 同 |
| `serve-off2` | 第 4 跑（最热） | `153775.513 ms` | `18240.659 ms` | 同 |

关：`163.2 / 153.8` → 均值 `158.5 s`。开：`87.7 / 84.5` → 均值 `86.1 s`。
**反向顺序也复现，所以不是"第二跑占了页缓存的便宜"**：顺序带来的漂移是 `4–6%`
（关那两跑差 `9.4 s`、开那两跑差 `3.2 s`），而效应量是 `72.4 s`，是漂移的 8–20 倍。

四跑的 id 都是 `58572,42549,42549,109865,42549,42549,42549,42549`——**与
`2026-09-16-sm120-multitoken-and-serve.md` 里槽位复用之前那个二进制跑出来的完全相同**，
跨二进制的锚也在。

日志 `prompt-chunk-runs/serve-{off1,on1,on2,off2}.log`。

## 代码变更（`exp/prompt-chunk`，`719b5f2`）

- `b2_runtime.rs`：新增 `flash_b1_prompt_chunk_enabled`，照 `flash_b1_prefetch_enabled`
  的写法，默认关；只在 `FLASH_0731` + `REQUESTS == 1` + 交付两个面上返回真。
- `generate.rs`：新增 `b2_prompt_chunk_span`（本趟处理几个位置）；外层迭代先算 span，
  prompt 段 = 剩余 prompt 长度、其他恒为 1；`hidden` 从一份变成每个位置一份；
  层循环里按 `offset × rows + row` 顺序调 `run_b2_layer`；digest 按位置存；
  提交循环按位置走。`parity_stop_after_layer` 那条 bench 路径要求 chunk 长度必须是 1（显式报错）。
- `flash_arms.rs`：给 `LayerStageTelemetry` 加手写 `AddAssign`（约 50 个 u64/计数器字段），
  一层仍然一条 profile，数字是该层这一趟所有位置的和，`attributed ≤ outer_wall` 自然成立。
  `routed_prefetched` 取 AND：混了同步读的一趟会报"没预取"并在嵌套计时守卫上显式失败，
  不会假装有后台读。
- `b2_generate.rs`：per-layer profile 加 `chunk_positions` 字段（默认关时恒为 1，
  所以旧证据的读法不变）。

## 未声称的部分

1. **CLI 那批的墙钟被并发运行污染了，那批只剩账本可用。** 跑 CLI 那批时机器上还有另一个
   会话（`exp/routed-read-threads`）在跑 GPU 任务：两跑做**完全相同的功**时墙钟能差 `12.8%`，
   而 8 步基线在干净机器上是 `70.06 s`、那次是 `93.66 s`。所以 `138.8 → 113.81 s` 那个
   `18%` **只能当方向**；那批的结论由逐字节账本承担，不由墙钟承担。
   **这条双向成立**：那几跑同样可能污染了对面那一批数字，两个实验的记录都该留这句话。
   serve 那批是机器空下来以后重跑的（`load average 0.00`），不受这条影响。
2. **serve 的绝对秒数会随盘的状态漂移，别当常数。** 那四跑是在 WSL 刚重启、页缓存冷的状态下
   量的，同一个二进制在热缓存下会整体更快。可用的是**同一批里的关/开之比**，不是跨批的绝对值。
   **而且这个比值本身也跟缓存状态有关**：稳态（热）账里固定参数只占每 token 的 `34%`，
   按那个份额外推首字最多降三分之一；这次冷缓存下量到 `46%`，比份额大，
   因为被省掉的那 7 趟正好是**最冷的**那几趟，而解码的趟跑在后面、占的缓存更暖。
   引用 `46%` 时要连这个条件一起说。
3. **只测了 5 个 token 的 prompt 和 8 个 token 的 serve prompt。** 121 个 token 的 prompt
   （8 步下的上限）起跑过一组，跑到 23 分钟时被主动掐掉——同机器上另一个会话的引擎又开始跑了，
   再并下去两边都不干净。**没写结果文件**，盘上没有半成品证据。收益该按公式线性外推，不是实测。
4. **attenion / routed 读 / shared 那三块一个字没动。** 按槽位复用后的账它们是每 token
   `8.28 s` 里的 `66%`（routed `25%` + shared `24%` + attention `16%`）。这一步只拿固定参数
   那 `34%`，所以预期收益上限就是约三分之一。
5. **没试过多请求（`REQUESTS > 1`）**，开关在那儿直接返回关。多行时 span 必须一致，
   代码里有显式报错兜着。
6. 层的 `e2e_ns` 在 chunk 内是共享的：同一趟里的位置报同一个趟墙钟。逐 token 的
   `hidden_bf16_digest` 才是锚，`e2e_ns` 在 prompt 位置上的含义变了。
7. **ITL 那四条里，开的那两跑平均比关的高约 `3%`（`19.07` vs `18.44 s`）。**
   解码路径按构造没被碰，而且"开"落在中间两跑、"关"落在最外两跑，跑次位置与开关共线，
   所以这个差**分不出是效应还是跑次**。没有把它当结论。

## 未决事项

1. **松开 `KV_WINDOW = 128` 的输入上限。** 交付这条路现在硬拒 `prompt + 生成 − 1 > 128`
   （`b2_generate.rs:91-99`；serve 返回 `CONTEXT_WINDOW_EXCEEDED`，`generate.rs:1055-1057`），
   所以 8 步生成时 prompt 最多 121 个 token。这一步的收益就卡在这个范围里。
   松开会让位置超过 128，走进 128 行 KV 环的绕回 + 两级压缩缓存那个**从没跑过**的区间
   ——独立的下一步，有自己的风险等级。
2. **121 token 的 prompt 实跑一次**，验证收益按公式外推成立，顺便压一下长 T 的边界。
3. **要不要合并进交付分支。** 数值零变化（12 个 digest 做证）+ 账本逐字节可核，按
   "交付分支只收验证过的东西"是可以进的；但和 `exp/slot-reuse` 一样，合并会让基线数字作废，
   所以是显式决定。注意它和 `exp/routed-read-threads` 不冲突：那个改的是读的并发度，
   这个改的是读的次数。
4. 多行（`REQUESTS > 1`）的 prompt chunk 没做；要做的话 span 对齐和
   `b2_router_metadata_row` 的行号布局都已经就位（`row = offset * rows + row`）。
