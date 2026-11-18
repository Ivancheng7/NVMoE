# routed 六专家并行下发：稳态每 token 省 0.33–0.44 秒（2026-09-21）

## 结论

Flash B1 预取那条路的 routed 段原来是**六个专家排成一队过一条流**：读盘走共享窗口里的六个读线程，
但上传（H2D）+ 三个 matvec 内核是一条流上串行的。这次把下发改成**一个路由槽位一个线程、一条流、一套槽位**：
每个槽位自己读盘、自己 `upload_lane`、自己在自己的流上跑内核，六个专家互相重叠。

数值一字未变：1 步四个钉死值全对、8 步 digest 逐位相同（三个交错轮次都比过）。

**表一：稳态每 token**（第 1–7 个生成 token 的均值，按层墙之和算）

| prompt 形状 | 关（交付默认） | 开 | 差 |
| --- | --- | --- | --- |
| 单 token（`TOKEN_IDS=0`，钉死值那条） | `2.240 s/token`（21 步，stdev 0.178） | `1.805 s/token`（stdev 0.273） | **−0.435（−19%）** |
| 真 32 token prompt | `1.913 s/token`（stdev 0.157） | `1.581 s/token`（stdev 0.168） | **−0.333（−17%）** |

**表二：8 步整条请求**（`request_e2e_ns`，三次交错均值；括号里是层墙之和，即流水线里那 344 条层的和）

| prompt 形状 | 关 | 开 | 差 |
| --- | --- | --- | --- |
| 单 token | **`34.11 s`**（`32.39`）——三次 `35.87 / 32.84 / 33.62` | **`30.76 s`**（`29.05`） | −3.35 s |
| 真 32 token | **`81.87 s`**（`79.78`）——三次 `83.69 / 80.75 / 81.16` | **`67.57 s`**（`65.48`）——`67.56 / 67.09 / 68.06` | −14.30 s |

⚠ **两种 prompt 的"8 步请求"不是一回事，别混**：单 token 那个是常驻/钉死值那批文档里一直引的
`33.9–36.8 s`；真 32 token prompt 因为每层要多跑 32 行预填充，同样 8 步、请求是它的两倍多。
预填充那一段（token 0，43 层 × 32 行）`67.66 → 55.04 s`，摊到 prompt 上是每行 `2.11 → 1.72 s`
（每行要走满 43 层）。这一整段也在同一条路上（每个位置一个单行 pass），所以 fan-out 一样吃到。


代价和边界，都要说清楚：

- **峰值显存 `8175288320 → 8242397184`（+64 MiB）**。一条路由槽位一套设备缓冲（12.75 MiB 权重 + 约 0.08 MiB
  计算缓冲），六条常驻。主机侧同样多锁约 64 MiB 页锁定内存。
- **诊断数字的含义变了，不能再相加**。六条 lane 抢同一条 PCIe 链路，所以每条 lane 自己测到的上传窗口里
  包含了等其它 lane 的时间：`routed.setup_ns` 和 `h2d_ns` 的和从 `6.20 / 5.70`（串行）涨到
  `16.96 / 10.40`（单 token 形态）、`24.53 / 16.90`（真 prompt 形态）。这是重叠的 lane 墙，不是变慢。
  routed 行为此新增 `"fanout": true` 和 `"fanout_lane_max_ns"` 两个字段，嵌套计时的守门条件也改成
  **"最慢那条 lane 必须在层墙之内"**（原来是"三个和值必须落在 `routed.wall` 里"，重叠后不成立）。
- **共享 MLP 的墙被顶起来了**：`shared.main`（就是 MLP）`3.98 → 10.36`（单 token）、`4.18 → 14.47`（真 prompt）。
  六个专家内核和 MLP 抢 GPU/SM，MLP 自己的窗口被拉长——重叠的净收益仍然为正（routed 段掉得更多），
  但 MLP 不再"免费"。这一项还没优化过。
- **A1+A2 组合（两个都开）**：稳态 `2.193 → 1.715 s/token`（单 token 形状，`−0.479 / −21.8%`）、
  `1.943 → 1.549`（真 32 token prompt，`−0.394 / −20.3%`）；e2e `34.39 → 30.28 s` 与 `83.33 → 69.09 s`。
  拆开看：A1 自己贡献 `−0.40 / −0.33 s/token`，A2 在 A1 之上只再给 `0.08 / 0.07`，
  整条请求上被它自己变慢的第一个 token 抵消掉。代价：显存 `+64 MiB`（A1）+ 锁页主机内存 `3.2 GiB`（A2）。
- **没声称**：`COLIBRI_V4_FLASH_ROUTED_FANOUT` **默认关**，交付面行为一个字没动，这是实验分支上的开关；
  serve 面没有重验；`stationary`/QD4/full61 那些路径不走这条路（它们收不到请求作用域的 scratch）。
- **绝对秒数不可跨记录比**：两个批次跑的时候盘速在 `1.7–1.8 GB/s`（慢档），下面的差值是**同一批内交错**得出的。

## 固定输入与模型

- 模型 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，48 个 `*.safetensors` 分片，
  `stat -c %s` 合计 `166886535336` 字节（09-16 的记录写的 `166898773853`，差 12.2 MB；`config.json`
  哈希与 revision 都与那批一致，差异未追）
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- 卡 RTX 5060 Ti 16 GB（`sm_120`），驱动 `591.86`；cuBLAS `12.8.5.5`（`18a5df54…`）
- 常驻显存按默认开（`b620ef7`），两次运行都打印 `fixed residency: 43 layers resident (6708864200 bytes)`
- 同一批里两个臂只差一个环境变量：`COLIBRI_V4_FLASH_ROUTED_FANOUT=0|1`

## 运行证据

构建：交付树 `/srv/localmoe/build/deliver-merge-0918`，`CARGO_TARGET_DIR=/srv/localmoe/targets/deliver-merge-0918-target`，
`./rust/run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`，exit 0，**62 条警告（与改动前相同）**。
二进制 `/srv/localmoe/targets/deliver-merge-0918-target/release/colibri-cuda`。

臂值（每行都相同）：`COLIBRI_V4_FLASH_BASE=1`、`COLIBRI_V4_FLASH_PREFETCH=1`、
`COLIBRI_V4_STREAM_READ_MODE=buffered`、`COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas`、
`COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas`、`COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice`；
每次都 `sync; echo 3 > /proc/sys/vm/drop_caches`。

结果文件（WSL `/srv/localmoe/targets/`，不在仓库里）：

- 主批 `a1-ab/`：`pin-{off,on}-1step.jsonl`、`single-r{1,2,3}-{off,on}.jsonl`、`real-r{1,2,3}-{off,on}.jsonl`
  （14 次运行全部 exit 0；盘速探针 1.7 GB/s）
- 组合批 `a1-ab2/`：`r{1,2,3}-{off,fanout,fanout-cache}.jsonl`（真 32 token prompt，盘速探针 1.8 GB/s）

数值判据：

- `python3 rust/colibri-cuda/tools/verify/verify_baseline.py
  pin-on-1step.jsonl` → **baseline reproduces**
  （argmax `55222`、`hidden_bf16_digest 5ed8bf88beae3cfe`、43 层、`block32`、`canonical_cpu_streamed_bf16`）。
- 三个交错轮次各比一次 off/on：8 个 `hidden_bf16_digest`、8 个 `ordered_layer_digest_sha256`、
  8 个 argmax、层数 **全部 IDENTICAL**，只有峰值显存差 `+67108864` 字节

每层相位（稳态 token 的 43 层均值，毫秒；单 token prompt）：

| 项 | 关 | 开 | 差 |
| --- | --- | --- | --- |
| `outer` | 52.09 | 41.98 | **−10.11** |
| `attention` | 12.99 | 12.93 | −0.06 |
| `router` | 2.57 | 2.60 | +0.03 |
| `shared.wall` | 17.22 | 10.36 | −6.86 |
| ├ `shared.main`（MLP） | 3.98 | 10.36 | **+6.38** |
| ├ `shared.join_tail` | 13.24 | 0.00 | −13.24 |
| └ `shared.read_worker_max` | 16.81 | 16.48 | −0.32 |
| `routed.wall` | 18.23 | 14.98 | −3.25 |
| ├ `routed.read`（和） | 64.17 | 62.83 | −1.33 |
| ├ `routed.setup`（和） | 6.20 | 16.96 | +10.76 |
| ├ `routed.h2d`（和） | 5.70 | 10.40 | +4.70 |
| ├ `routed.kernel`（和） | 10.76 | 12.47 | +1.70 |
| ├ `routed.lane_max` | — | 24.47 | — |
| └ `routed.source` GiB/token | 3.21 | 3.21 | 0 |
| 未归属 gap | 0.60 | 0.63 | +0.03 |

真 32 token prompt 同一张表（毫秒）：

| 项 | 关 | 开 | 差 |
| --- | --- | --- | --- |
| `outer` | 44.49 | 36.77 | **−7.72** |
| `shared.wall`（=`shared.main`） | 9.39 | 14.47 | +5.08 |
| `shared.read_worker_max` | 8.85 | 9.12 | +0.27 |
| `routed.wall` | 18.16 | 5.32 | −12.84 |
| ├ `routed.read`（和） | 35.18 | 35.59 | +0.41 |
| ├ `routed.setup`（和） | 6.18 | 24.53 | +18.35 |
| ├ `routed.h2d`（和） | 5.70 | 16.90 | +11.20 |
| ├ `routed.kernel`（和） | 10.71 | 14.11 | +3.40 |
| ├ `routed.lane_max` | — | 18.84 | — |
| └ `routed.source` GiB/token | 3.21 | 3.21 | 0 |

读盘量两边完全一样（每 token 3.21 GiB、258 次专家载入），所以差值不是少读了什么，
是**读、上传、内核重叠**吃出来的：关的那边共享窗口 = max(MLP, 最慢读) 之后才开始串行下发，
开的那边这一整段压成"最慢那条 lane 的时间"（`lane_max`）。

### 组合批：A1 + A2（真 32 token prompt，三臂同批交错）

A2（按专家身份的缓存，`10f0958`）单独量的时候墙几乎没动，那份记录留的话是
**"收益要与 A1 一起判"**。这批就是把两个叠起来量：

| 臂 | 稳态 s/token | 层墙之和 | **e2e** | 其中 token 0（32 行预填充） | 对 off（e2e） |
| --- | --- | --- | --- | --- | --- |
| `off` | 1.943（`a1-ab2/r{1,2,3}-off.jsonl`） | 81.3 s | **83.33 s** | 67.66 s | — |
| `fanout` | 1.619 | 66.4 s | **68.64 s** | 55.04 s | −14.7 s |
| `fanout` + cache 258 | 1.549 | 67.1 s | **69.09 s** | 56.22 s | −14.2 s |

**整条请求上 `fanout` 和 `fanout+cache` 基本打平（`68.64` 对 `69.09 s`）**：缓存让稳态每 token 快
`0.07 s`，但 token 0 慢 `1.2 s`（要多锁页那些 lane、第一遍的位置全 miss），8 步之内抵消掉。
要看缓存的收益就得看更长的生成，这一批（8 步）看不出来。

**同一个三臂在单 token prompt 形状（`TOKEN_IDS=0`，三次交错）上也量了一遍**（`a1-ab3/`）：

| 臂 | 稳态 s/token | 层墙之和 | e2e | token 0 |
| --- | --- | --- | --- | --- |
| `off` | 2.193 | 32.71 s | 34.39 s | 17.36 s |
| `fanout` | 1.791 | 28.78 s | 30.45 s | 16.24 s |
| `fanout` + cache 258 | **1.715** | 28.51 s | 30.28 s | 16.50 s |

这个形状的命中率低一些（每层命中 `3.4/6`、载入 `6.0 → 2.6`，真 prompt 是 `5.0/6`），
所以缓存多给的还是那 `0.08 s/token`，e2e 上与 `fanout` 只差 `0.17 s`。**两个形状的结论一致：
合并收益基本等于 A1 自己的收益，A2 在它之上只再占 `0.07–0.08 s/token`（约 3–4%）。**

每层相位（稳态均值，毫秒）：

| 臂 | `shared`（=MLP） | `routed` | `lane_max` | `h2d` 和 | `kernel` 和 | 载入/层 | 命中/层 | `source` GiB/层 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `off` | 9.58 | 18.51 | — | 5.70 | 11.00 | 6.0 | 0 | 0.07 |
| `fanout` | 14.41 | 5.98 | 19.45 | 16.88 | 14.38 | 6.0 | 0 | 0.07 |
| `fanout`+cache | 13.18 | 5.76 | 16.83 | 14.44 | 14.79 | **1.0** | **5.0** | **0.02** |

数值同样逐位相同（三臂逐轮比，digest/层摘要链/argmax/层数全 `IDENTICAL`，峰值只差那 `+64 MiB`）。

**A2 在 A1 之上只再省 `0.07 s/token`，而且机制和上一份记录的预期不一样，这里要写清楚**：
- 上一份记录猜的是"读盘本来被 shared MLP 藏住，把读挪出窗口就能全额上墙"。**这只对了一半**：
  读盘确实不再藏在 MLP 后面了，但 fan-out 之后一层的临界路径是**最慢那条 lane 的
  读→上传→内核**，缓存能砍掉其它五条（`lane_max 19.45 → 16.83`），砍不掉那条 miss 的 lane，
  它照旧要读 12.75 MiB。所以省下的是"六条并发读互相抢"那点竞争（`−2.6 ms/层`），不是读盘本身。
- 磁盘流量降得很干脆：真 prompt 形状每层 `6.0 → 1.0` 次载入、`3.21 → 0.86 GiB/token`
  （命中率 83%）。**在这个盘速档位上这没有变成墙上的秒数**——它省的是"盘不被打爆"，
  以及冷跑时那一段。要不要为它付 `3.2 GiB` 锁页主机内存，是可以单独拍板的事。

## 代码变更

提交 `0d12b0f`（交付分支 `codex/business-restart`）。`COLIBRI_V4_FLASH_ROUTED_FANOUT=1` 打开，默认关。
只碰 routed 预取这条路，其它调用点行为不变。

- `expert_slot.rs`：`RoutedSlotScratch` 多一个 `fanout_lanes: Vec<ExpertSlot>` 加一对
  `take_fanout_lanes`/`put_fanout_lanes`。槽位（和它的流，`context.new_stream()`）在请求的第一次
  fan-out 层按需建好，之后每层复用，锁页只付一次。这里照抄的是 QD4 已有的形态
  （`generate.rs`/`full61_b1.rs` 里 `ExpertSlot::new_empty(context, module, &lane_stream)`）。
- `layer_rows.rs`：
  - `routed_fanout_enabled()`（读 `COLIBRI_V4_FLASH_ROUTED_FANOUT`，只认 `"1"`）；
  - `RoutedLaneOutcome`：一条 lane 的产物（输出、read/h2d/setup/kernel、自己的墙、lane、槽位）；
  - 预取块里新增 fan-out 分支：每个路由槽位一个 `scope.spawn`，闭包里
    `bind_to_thread` → `read_into`（缓存命中就跳过）→ `upload_lane` → `execute`，
    主线程同时跑共享 MLP；
  - **窗口口径改了**：`stage.shared_ns` 在 MLP 返回那一刻取（不再等 join），`routed_started`
    也在那一刻取。两个窗口仍然首尾相接、不重叠，`top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns`
    这条闭合不变（未归属 gap 仍是 0.6 ms/层）；读盘一直记在 `routed.read` 里，这次没有改记法。
  - drain 里新增一支：fan-out 的结果已经在别的流上算完了，这一支只放输出、折计数、把 lane 和槽位还回去。
- `flash_arms.rs`：`LayerStageTelemetry` 加 `routed_fanout: bool`（折的时候取与）和
  `routed_fanout_lane_max_ns: u64`（折的时候取最大）。
- `b2_generate.rs`：routed 行加 `"fanout"`、`"fanout_lane_max_ns"`；嵌套计时守门条件里，
  fan-out 这一支改成"最慢 lane ≤ 层墙"，不再拿三个和值去比 `routed.wall`
  （和值超墙不是记账错，是六条 lane 在上传窗口里互相等）。
- `phase_profiles.rs`：只改注释（`shared.join_tail` 在 fan-out 下恒为 0）。

为什么这样做：QD4 那条路早就是"多条 lane + 每条自己的流"的形态（`generate.rs`/`full61_b1.rs` 里
`ExpertSlot::new_empty(context, module, &lane_stream)`，一次四到六条），这次只是把预取这条路对齐到
同一个做法。**六条到底有没有真并行，不靠推断**：同一段读、上传、内核叠起来之后，读盘量一个字没变
（每 token `3.21 GiB`、258 次载入两边相同），而层墙从 `52.09` 掉到 `41.98 ms`（单 token 形态）。

## 未决事项

1. ~~**要不要默认开**。数值逐位不变、每 token 省 17–19%，但峰值显存 +64 MiB、诊断含义变了，
   而且量化口径要在交付文档里改一遍。这是需要显式拍板的合并，不是顺手提交。~~
   → **2026-09-30 已翻成默认开**（`9799a0c`，`=0` 回退）。翻之前先补了这份记录欠下的那件事：
   **服务面从没验过**。用新增的 `tools/serve_arm.sh` 补上——同一个二进制、两臂交错各三轮，
   两臂产出同样的 8 个 id 串，稳态每字 `2.231 → 1.767 s`（`−0.464`）。翻之后的默认面闸门
   （1 步钉死值 + 8 步 8 digest，两个顺序各一趟）全部复现，峰值 `8175288320 → 8242397184`，
   默认面稳态 `2.388/2.403 → 1.986/1.977 s/token`（单 token 形状）。诊断口径已按上面那条
   "和值不能再相加"改过交付文档。记录 `docs/experiments/2026-09-30-sm120-serve-fanout-arm.md`。
2. ~~**MLP 被顶起来那 6–10 ms**：`shared.main` 从 3.98 涨到 10.36（真 prompt 4.18 → 14.47）。
   六条 lane 的内核和 MLP 抢 SM。想拿回来可以试：先跑 MLP 再放开 lane（会损失重叠）、
   给 MLP 的流更高的优先级、或者把 lane 的启动错开。现在没试过任何一个。~~
   → **2026-09-23 量清了，而且"先修 MLP"这个排序是错的。** 用"故意不跑 lane 内核"的消融臂
   （`/srv/localmoe/targets/a1-mlpabl/`，同一个二进制、三臂同批交错、真 32 token prompt，
   消融臂输出无效只当探针）把它拆开：

   | 臂 | `shared.main`（MLP） | `lane_max` | 稳态 s/token |
   | --- | --- | --- | --- |
   | `off` | 4.46 | — | 1.958 |
   | `fanout` | 14.38 | 19.59 | 1.618 |
   | `fanout` + 不跑 lane 内核（消融） | 8.99 | 13.35 | 1.352 |

   `+9.9 ms` 里**约 4.5 ms 来自六条 lane 的上传**（和 MLP 自己那点小拷贝抢同一条 PCIe 链路），
   **约 5.4 ms 来自六个内核抢 GPU**；lane 的**读**不背这个账（关的那一臂在同一个窗口里也在读，
   MLP 仍是 4.46）。
   **但它现在不卡人**：fanout 那一臂 MLP `14.38` **小于** `lane_max 19.59`，层的墙由最慢那条 lane 决定。
   要等 lane 那条链（读→上传→内核）先降到 14 ms 以下，MLP 才会变成绑定项——所以下一步该动的是
   **把读从 lane 的关键路径上挪走**（路由前瞻预取那一路），而不是先给 MLP 排优先级。
   （消融那批的 `fanout` 臂 token 0 有一次 `76.5 s` 的离群值，其余两次 `54.5/54.7`；表里用的是稳态均值。）
3. **上传窗口含等待**：六条 lane 抢一条 PCIe 链路，`setup`/`h2d` 的和值不再等于链路时间
   （串行 5.70 → 并行和值 10.40/16.90）。真实链路占用率没有被单独量过；如果要，得在
   lane 上传外面再加一层链路上的计时，或者收敛成一条拷贝队列。
4. **新的瓶颈是最慢那条 lane 的读→上传→内核**（单 token 形态 `lane_max 24.47`，其中读 `16.48`；
   真 prompt 形态 `19.45`）。**这一项是唯一卡住层的**（见未决事项 2 的消融：MLP `14.38` < `lane_max 19.59`）。
   这正是 A2（按专家身份的缓存）能砍的东西，组合量下来只再省
   `0.07 s/token`，机制见上文——缓存砍不掉那条 miss 的 lane。想让这一项再降，得从
   "一条 lane 串行做完读+上传+内核"这个形状下手：**最值钱的是把读挪出这条链**（路由前瞻预取，
   象 colibri 的 `PILOT=1`；我们只有固定权重的提前一层），其次才是 lane 内 H2D 与内核的双缓冲
   （`−2.4 ms/层`量级）。
5. **六套槽位里有一套是浪费的**：fan-out 只需要每 lane 12.75 MiB 的权重设备缓冲是独立的，
   gate/up/routed/output 这些计算缓冲完全可以共用（它们只在同一个 lane 的内核里活着）。
   现在按 6 套分配，多花约 60 MiB 设备 + 60 MiB 锁页内存。值不值得省，取决于谁在什么显存档位下跑。
6. **serve 面没重验**：`run_real_v4_flash_serve` 走的是同一个 b2 入口，理论上同一个开关就生效，
   但这一批没跑服务。
7. **长 prompt 的预填充**：真 prompt 的 token 0 从 68.4 s 降到 54.4 s，方向是对的，但那一段是
   32 个单行 pass，`chunk_positions` 那一套折叠口径没在 fan-out 下单独验过（数值是一样的）。
8. `tests.rs:1076,1108` 调用 `flash_b1_layer_phase_profile_json` 的还是 6 个参数（现在 7 个），
   `cargo test` 编不过——这是改动前就有的旧账，不是这次引入的。
