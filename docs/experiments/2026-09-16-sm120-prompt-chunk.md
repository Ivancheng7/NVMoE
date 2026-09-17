# 整个 prompt 共用一趟层扫描：数值逐位不变，固定参数少读 4 趟（2026-09-16 深夜）

## 结论

预填充的每个 token 不再各走一遍 43 层。开关 `COLIBRI_V4_FLASH_PROMPT_CHUNK`
（**2026-09-17 起默认开**：不设或 `=1` 是开，`=0` 回到逐位置扫描），
只在交付的两个 Flash 面上生效（`COLIBRI_V4_FLASH_BASE` / `COLIBRI_V4_FLASH_SERVE`）。

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

## 合并进交付分支（2026-09-17）

合并提交 `993a6a8`（`merge: serve a whole prompt with one layer pass instead of one per position`），
父提交是交付分支的 `b0766ec`。**开关仍然默认关，所以合并本身不改交付行为**；
把它打开是另一个决定（而且打开会让基线数字作废）。

`generate.rs` 三处冲突，都在这儿解掉了。**带载荷的那一处**：交付分支在这期间合进了
`exp/routed-read-threads` 和 `exp/routed-slot-resident`（`5a9e35c`），
`run_b2_layer` 因此多了一个 `routed_slot_scratch` 参数——本分支的 chunk 循环已经改成同样传它。
另两处是纯函数位置重合（隔壁新加的 `load_flash_b1_layer_from_host` 与本分支的
`b2_prompt_chunk_span` 落在同一处；以及隔壁的 `from_prefetch` 遥测与本分支按位置展开的
`metadata_rows`）。**另一个会话未提交的 `AGENTS.md` 刻意没有带进这次合并。**

合并树在 WSL 侧单独建了 worktree（`/srv/localmoe/build/merged-prompt-chunk`，detached `993a6a8`），
构建 `58.16 s`、exit 0、**59 条警告**（与合并前一致）。交付分支那个 WSL 仓库没动，
仍然干净停在 `b0766ec`。

**已在合并树上重验的：**

| 项 | 结果 |
| --- | --- |
| 单步 `argmax` / `hidden_bf16_digest` / 层数 | `55222` / `5ed8bf88beae3cfe` / `43`，全同 |
| 单步层摘要链 `ordered_layer_digest_sha256` | `53de221f…`，与本分支合并前一模一样 |
| 8 步 8 个 `hidden_bf16_digest` | 逐个与钉死列表相同 |
| 单步/8 步固定参数字节 | `6,708,949,192` / `53,671,593,536`，与合并前逐字节相同 |
| 峰值显存 | `1500053504`——**这是隔壁 `routed-slot-resident` 记录过的新值**（`+32 MiB`，一个槽位的设备缓冲，真实峰值没变大、只是采样点挪了），不是这次合并带出来的回归 |

**合并树上的 chunk>1 路径也验完了**（机器空着跑的，四跑交替：关-开-开-关）：

| 跑 | 顺序 | 整请求墙钟 | 固定参数扫描 | `chunk_positions` |
| --- | --- | --- | --- | --- |
| `merged-pc-off-5tok-8step` | 1 | `106.55 s` | 12 | 1 |
| `merged-pc-on-5tok-8step` | 2 | `77.01 s` | 8 | 5 |
| `merged-pc-on2-5tok-8step` | 3 | `77.72 s` | 8 | 5 |
| `merged-pc-off2-5tok-8step` | 4 | `99.75 s` | 12 | 1 |

关 `106.55 / 99.75`（均值 `103.15`）→ 开 `77.01 / 77.72`（均值 `77.37`）：**省 `26.03 s`、降 `25.2%`**。
反向顺序同样复现；顺序漂移 `0.9–6.4%`，效应是漂移的 4–28 倍。

**四跑的 12 个 `hidden_bf16_digest`、12 个层摘要链、`generated_ids`、`routed_source_bytes` 完全相同**，
而且与合并前那一批逐字节相同（`e8a549edb330f197, 9ced1708b6b7f2fd, 588751df624cce01, 47fac5fd66b13ba3,
ffbefa495c76432a, 02edbae43189b90b, 5bcb214ab5759e30, e78f833da8f64512, 7ffd7b02a18b6251,
24719a9390f06216, 9ffb92fb2d97a62b, 298cadce7c1b3f7d`）。账本同样是整数倍：
`80,507,390,304 → 53,671,593,536`，正好 4 趟 × `6,708,949,192`。峰值显存四跑都是 `1500053504`。

**这次的 `25.2%` 和 serve 那次的 `46%` 不是同一个分母**：这里是整请求（12 趟里省 4 趟），
serve 那次是首字（15 趟里省 7 趟）。别混着引用。

## 两个默认值改了（2026-09-17，显式决定）

- `4a72109`：**prompt chunk 默认开**。`COLIBRI_V4_FLASH_PROMPT_CHUNK` 不设或 `=1` 是开，
  `=0` 回到合并前的逐位置扫描（旧基线仍可复现）。
- `598958e`：**固定参数的主机常驻默认开**。`COLIBRI_V4_FIXED_HOST_RESIDENT_GIB` 不设时是
  `8 GiB`（实测过的那档，装得下整个 `5.6 GiB` 固定模型），`=0` 仍可关。

两个都只影响交付的两个 Flash 面：Pro、QD4、DSpark、多请求一律不变。

### 新默认下的实测（serve、真实中文 prompt、四跑交替）

| 跑 | 首字 | 每字间隔 |
| --- | --- | --- |
| 新默认 ① | `80.24 s` | `10.00 s` |
| 旧行为（两个都关）① | `141.29 s` | `16.78 s` |
| 旧行为（两个都关）② | `137.82 s` | `16.65 s` |
| 新默认 ② | `79.47 s` | `10.03 s` |

首字 `139.6 → 79.9 s`（`−42.8%`），每字间隔 `16.7 → 10.0 s`（`−40%`）。四跑 token id 全同。
每字间隔一起降，是因为主机常驻对**每个** token 都省掉一次固定参数的读盘与解码，
而 prompt chunk 只管 prompt 那一段。

钉死值不变：单步 `argmax=55222` / `5ed8bf88beae3cfe` / 43 层 / 峰值 `1500053504`；
8 步 8 个 `hidden_bf16_digest` 逐个相同（`defaults-pc-1step.jsonl`、`defaults-pc-8step.jsonl`，
峰值 RSS 到 `7.24 GB`，与常驻那篇记的 `6.96–7.25 GB` 一致）。

### 一个还没量到的数：常驻开着之后 chunk 还剩多少

两个改动的收益**重叠**。常驻开着时固定参数整个运行只从盘读 **1 趟**
（8 步那跑 `fixed_source_bytes = 6,708,949,192`，正好一趟），
而本文前面那个"12 趟 → 8 趟"的账本判据是**常驻关着**的时候量的。
chunk 原来靠"少读 4 趟"拿到的收益，被常驻吃掉了一大块，剩下的只有 H2D 和每层解码那部分。

试着量过一次增量对照（常驻都开，chunk 开 vs 关），**但那批无效**：标签前缀与脚本分支
对不上，四跑配置其实完全相同。有效的信息只有一条——**同一配置的四跑自己就在
`78.1–82.8 s` 之间漂（约 `6%`）**，比这个漂移小的效应用墙钟看不见。

**所以现在不能说 chunk 在新默认下还有正收益，也不能说它是负的。** 要答这个问题得换更硬的判据
（例如固定参数的 H2D 字节数之差），不是墙钟。如果量出来它确实不赚，关掉是一行：
把 `4a72109` 里那个 `None => Ok(true)` 改回去。

## 服务路径一直白白慢一倍：预取线程的门卫只认命令行（2026-09-17）

`2026-09-16-sm120-multitoken-and-serve.md` 的未决事项 5（"serve 每迭代比 CLI 贵约 1.75 倍，没解释"）
**解释在这里，不是那一刀弄出来的，也不是本分支弄出来的**。

`flash_b1_prefetch_enabled`（`b2_runtime.rs:17`）要求启动时带 `COLIBRI_V4_FLASH_BASE`，
而服务走的是 `COLIBRI_V4_FLASH_SERVE`（`main.rs:245`），**不设 `FLASH_BASE`**。
于是服务上每层固定参数的"读盘 + 解码 + 申请缓冲 + H2D"整套**串在关键路径上**，
命令行那边它是被后台线程和上一层计算重叠掉的。
当初那行注释写明是故意只给命令行用的（预取当时还是实验品），后来没人往回接。

**实测**（同一个二进制、同一份权重，其它环境一模一样，只差这一个变量）：

| 跑 | 首字 | 每个字 |
| --- | --- | --- |
| `base1`：也设上 `COLIBRI_V4_FLASH_BASE=1` | **`27.1 s`** | **`4.52 s`** |
| `nobase1`：只设 `FLASH_SERVE`（现在的默认） | `79.9 s` | `9.87 s` |

token id 两边相同。而且这次顺序对结论**不利**——`base1` 先跑（缓存更冷）却快 3 倍，
所以缓存顺序的偏差反而是站在"没效果"那边的。

顺带把"serve 比 CLI 贵"这件事**翻过来了**：开预取之后服务每个字 `4.52 s`，
比命令行那条 8 步的 `5.3 s/步` 还低。那个 1.75 倍就是这一处。

**为什么服务现在比命令行还快**：两边现在都开着预取和常驻，差别是服务有真实 prompt、
命令行那次 prompt 只有 1 个 token——而 serve 的 ITL 是"第一次出字之后"的量，首字那段不计在里面。

### prompt 长度会不会影响每个字的速度

不会（在能跑到的范围内）。

| prompt 长度 | 首字 | 每个字 |
| --- | --- | --- |
| 8 个 token | `27.1 s` | `4.52 s` |
| `91` 个 token（`serve-longprompt.log`） | `240.9 s` | **`4.44 s`** |

差 `1.8%`，噪声。**首字**才跟着长度涨：每多一个 prompt token 约 `2.6 s`
（`(240.9 − 27.1) / (91 − 8)`），按 121 的上限外推约 `5` 分钟出头。

结构上也说得通：prompt + 生成 ≤ 128，所以 128 行的裸 KV 环**从不绕回**，
每个生成 token 要看的压缩行只从个位数涨到三十几行，而每层注意力的大头是
compressor/indexer 那几处固定开销（`2026-09-16-sm120-*` 那批记录里量过它不随上下文增长）。

注意这次长 prompt 的输出全是 `51616` 重复——base-only 贪心、只比 id，**不评价语义**。

## 未决事项

1. **松开 `KV_WINDOW = 128` 的输入上限。** 交付这条路现在硬拒 `prompt + 生成 − 1 > 128`
   （`b2_generate.rs:91-99`；serve 返回 `CONTEXT_WINDOW_EXCEEDED`，`generate.rs:1055-1057`），
   所以 8 步生成时 prompt 最多 121 个 token。这一步的收益就卡在这个范围里。
   松开会让位置超过 128，走进 128 行 KV 环的绕回 + 两级压缩缓存那个**从没跑过**的区间
   ——独立的下一步，有自己的风险等级。
2. **121 token 的 prompt 实跑一次**，验证收益按公式外推成立，顺便压一下长 T 的边界。
3. **量出常驻开着时 chunk 的增量**（上面那条）。现在两个默认都开着但只量到合并收益，
   分不开是谁的。判据建议用固定参数的 H2D 字节数，不用墙钟。
4. **把预取线程的门卫放宽到服务路径。** 上面那条实测已经指认它是服务慢一倍的原因，
   改法和 chunk 当初做的一样（门卫里 `BASE` 或 `SERVE` 都放行），之后
   **必须按 serve 重新验一遍**：token id 相同、以及那条路上钉死值/8 步 digest 仍成立。
   没验之前不要把 `COLIBRI_V4_FLASH_BASE=1` 当服务的正式跑法写进任何配方。
4. 多行（`REQUESTS > 1`）的 prompt chunk 没做；要做的话 span 对齐和
   `b2_router_metadata_row` 的行号布局都已经就位（`row = offset * rows + row`）。
