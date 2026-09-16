# fixed 相位的窗口拆开了：17.4 ms 是释放那 137 MB 主机缓冲（2026-09-16 晚）

## 结论

把 `fixed` 那一项的墙拆成四段量了一遍（8 步、336 个走预取的层，插桩构建、真跑）：

| 每层 | 均值 | 最小 | 最大 | 是什么 |
| --- | --- | --- | --- | --- |
| `fixed.wall` | `51.41 ms` | 32.64 | 139.93 | 主线程从层循环开头到 H2D 结束 |
| ├ `fixed.h2d` | `23.26 ms` | 18.97 | 27.85 | 真正搬上显存，一天没变过 |
| ├ `window.join` | `10.68 ms` | 0.00 | 97.22 | 等每层提前一个的预取线程 |
| ├ `window.spawn` | `0.11 ms` | 0.00 | 0.35 | 起下一个预取线程，可以忽略 |
| └ `window.release` | `17.36 ms` | 13.59 | 22.04 | **释放那 137 MB 主机缓冲** |

四段之和与墙逐层严格闭合（代码里那条等式照旧成立）。**`window.release` 和"减去前三段剩下的残差"
逐层相等**，也就是说窗口里唯一没被计时的事就是它。

结论三条：

1. **上一份记录问的那个问题，答案是"是"**：那 137 MB 固定参数的主机缓冲**每层都重新申请、用完就还**。
   `v4_state.rs` 里 `read_bf16`、`read_f32`、`read_u8` 全部是 `vec![0; n]`，逐张量、逐层新建。
   一个 token 43 层 × 137.39 MB = 5.9 GB 的申请/释放churn，**释放那一半落在主线程的关键路径上**。
2. **`fixed` 从 43.2 涨到 65.2 ms/层，不是这条路变慢了，是留给预取线程的藏身之处变小了。**
   实测规律（336 层）：`join = max(0, 预取线程的墙 − 上一层剩下给它的空闲)`，误差 sd `1.25 ms`、
   最大 `4.29 ms`。基线每层 285 ms、空闲 244 ms，预取只要 151 ms，所以从不等待，那 17.26 ms
   "缺口"就是 release；槽位复用后每层 188 ms、空闲 128 ms，预取仍要 147 ms，于是开始等。
   这次测量里 `worker_ns` 均值 `162.04 ms`、`slack` 均值 `162.4 ms`——两个数几乎相等，
   所以正好一半的层（168/336）在等。
3. **可捡的钱是明码的**：release `17.36 ms/层` 在关键路径上，纯粹是还内存，复用两个主机槽位就能去掉，
   合 `0.75 s/token`；如果复用同时把预取线程里"申请新页"那部分也去掉、让它重新装得进空闲，
   `join` 也会回到 0，fixed 的墙就只剩 H2D 的 `23.3 ms`（这次数字下合计约 `1.2 s/token`）。
   仓库里已经有这套"两个槽位轮转、提前一层"的实现（`PinnedFixedSlot` + `prepare_stationary_raw_pinned`），
   flash B1 解码这条路没用它。

**没声称的部分**：拆出来的这四段来自 `exp/fixed-window` 上的插桩构建，**不是交付分支的二进制**，
交付分支的二进制测完已经重建回干净状态（59 条警告、单步四个钉死值复现）；
`release` 是"在这个函数末尾释放这块内存的耗时"，机制（内核回收页）是推断；
`join` 那条规律是在**这一次**运行上拟合的；上面第 3 条的省时是算出来的，**没有跑过**。

**同日深夜补测："预取线程那 162 ms 里读盘占多少"也有答案了——读盘不是主角，见下面那一节。**

## 预取线程那 162 ms 的构成：读盘不是主角（同日深夜）

探针加在 `read_u8` / `read_bf16` / `read_f32` 三个分配函数里（预取这条固定参数的路只走这三个），
把三件事分开计时：`vec![0; n]` 本身、往新页里写一遍逼出缺页、然后才是 `read_at`（这时只拷贝）。
累加器用线程本地（loader 整个跑在一个线程上：预取层在预取线程，layer 0 在主线程）。

结果（`fixed-worker-split-b-8step.jsonl`，336 个预取层的均值）：

| 每层 | 均值 | 最小 | 最大 | 是什么 |
| --- | --- | --- | --- | --- |
| `alloc` | `25.68 ms` | 0.50 | 50.82 | `vec![0; n]` 本身 |
| `fault` | `58.45 ms` | 43.06 | 117.54 | 预写逼出来的缺页（真实运行时这批缺页发生在 `read_at` 里面） |
| `read` | `40.56 ms` | 10.24 | 164.32 | 页已就位之后的纯拷贝 |
| `other` | `57.73 ms` | 49.77 | 82.04 | NaN 扫描（`read_matrix` 里对全量权重做 `any()`）、每层重新解析 `model.safetensors.index.json`、开分片文件 |
| 合计 | `182.43 ms` | | | 等于 `read_alloc`，逐层闭合 |

**探针自己多花了一次写**：预写把 137 MB 整块写一遍，所以预取线程从 `161.18` 涨到 `182.59 ms`、
8 步总时长 `79.44 → 85.48 s`。把这一次额外写（约 16–20 ms）扣掉正好回到没插桩时的 `161–166 ms`，
所以这个构成是可信的，**但这几个数字不能当性能数字用**。

**结论：那 162 ms 里读盘不是大头。** 申请 + 缺页 ≈ `84 ms`（52%），纯拷贝 ≈ `41 ms`，
其余 ≈ `58 ms` 是扫描与索引解析。申请和缺页这两样**只有"每层重新申请一块缓冲"才会发生**，
复用缓冲能一次消掉它们。

数值不变：8 个 token 的 `hidden_bf16_digest` 与 `slotreuse-8step.jsonl` 逐个相同，
`argmax=55222`、峰值 `1466499072`。（预写只写 0，随后 `read_at` 把每个字节都覆盖掉，
所以数值不可能变——这也是为什么要核一遍。）

因此上面第 3 条的省时现在有依据了：复用两个主机槽位能同时消掉预取线程里的
**申请+缺页 84 ms** 和主线程上的 **`release` 17.4 ms**，预取线程掉到 `78–98 ms`，
今天的空闲（`162 ms`）和 19:21 那次的（`128 ms`）都装得下，于是 `join` 归零、`release` 归零，
`fixed` 的墙只剩 H2D 的 `23.4 ms`。按今天的数字每层省 `≈38 ms`（`≈1.6 s/token`），
按 19:21 的数字 `≈42 ms`（`≈1.8 s/token`）——和交接文档原来的估算一致，但这次是量出来的构成。

**没声称的部分**：复用本身**没有实现、没有跑**，上面的省时是从构成算出来的；
`other` 那 58 ms 里三个来源各占多少没再分；这个探针的绝对时间偏慢，不能当性能数字。

## 固定输入与模型

与 `2026-09-16-sm120-nonresident-baseline.md`、`2026-09-16-sm120-slot-reuse.md` 同一份权重、同一张卡、同一套臂：

- 模型 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 48 个 safetensors 分片、总计 `166898773853` 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- RTX 5060 Ti 16 GB（`sm_120`）、cuBLAS `12.8.5.5`

## 运行证据

源码：WSL `/srv/localmoe/build/localmoe-clean-git`，测的时候在 `exp/fixed-window` 上，
提交 `b8cd920`（拆 join/spawn/残差）、`190f1cf`（加 release 计时）、`5f2b72e`（拆预取线程）、
`27133ce`（把新字段搬过 `load_b2_raw_runtime_from_host_for` 的重建），父提交 `c3c1be8`（槽位复用）。
测完已切回 `codex/business-restart`（`72e2611`）、工作区干净、二进制重建、单步四个钉死值复现。

> 一个坑，写下来省下一次：`git reset --hard FETCH_HEAD` 挪的是**当前分支**。
> 最后一次同步探针时它正好在 `codex/business-restart` 上，于是克隆里的交付分支被挪到了探针提交
> （三次测量的构建源码都是对的，坏的只是这个标签）。修法：
> `git checkout -f -B codex/business-restart origin/codex/business-restart`。

构建 `18.90 s` / `18.18 s`、exit 0、**59 条警告**（插桩没有多出警告）。环境变量与槽位复用那次完全一致
（`BASE=1`、`TOKEN_IDS=0`、`GREEDY_STEPS=8`、`STREAM_READ_MODE=buffered`、`PREFETCH=1`、
`MHC_LINEAR_ARM=cublas`、`ROUTER_LINEAR_ARM=cublas`、`ROUTER_SCORE_ARM=libdevice`；
`FP4_ACCUM_ARM` / `FINAL_HEAD_ARM` 不设）。

结果文件（都在 `/srv/localmoe/targets/`，都是新名字）：

- `fixed-window-8step.jsonl`（第一次插桩，`79.55 s`）
- `fixed-window2-8step.jsonl`（加 release 计时，**`79.44 s`**，上面窗口那张表出自它）
- `fixed-worker-split-8step.jsonl`（第一次拆预取线程，新字段没被搬过去、三项全是 0，**别引用**）
- `fixed-worker-split-b-8step.jsonl`（**上面构成那张表出自它**，`85.48 s`——探针自己多一次写，见那节）
- `post-fixed-probe-1step.jsonl`、`post-fixed-probe3-1step.jsonl`（切回交付分支后重建的二进制，四个钉死值复现）
- 对照：`slotreuse-8step.jsonl`（`70.06 s`）、`baseline-nonresident-8step.jsonl`（`103.38 s`）

数值不变：8 个 token 的 `hidden_bf16_digest` 与 `slotreuse-8step.jsonl` **逐个相同**，
`argmax=55222`、峰值显存 `1466499072` 字节。

**这次的绝对总时长不能跟 `70.06 s` 比**：同样的 `137391448` 字节源数据，预取线程这次要
`161.87 ms`，而 19:21 那次只要 `146.99 ms`、基线那次 `151.28 ms`。盘/page cache 状态不同，
这次整体慢了一档（每 token outer：`12.71 9.56 8.68 9.06 8.48 8.63 8.44 8.96` 秒，
槽位复用那次是 `8.61 8.70 7.92 8.06 8.07 8.07 7.91 8.28`）。**这条记录立的是拆法，不是新基线。**

进盘/过 PCIe 的字节与旧记录一致：`fixed_source_bytes` `53671593536`、`fixed_h2d_bytes` `53670913600`。

读取脚本在 Windows 侧 `.zcode/`（`fixed_split.py`、`fixed_window_read.py`、`fixed_window_summary.py`），
**未入 git**，是本地工具。

### `join` 那条规律怎么验的

对 336 个走预取的层，比较实测 `join` 与 `max(0, worker_ns − 空闲)`，其中"空闲"取**上一层**的
`outer.wall − fixed.wall`（预取线程是在上一层里被起的，它真正拥有的窗口就是上一层的剩余时间）：

| 用哪一层的空闲 | 平均偏差 | 偏差 sd | 最大偏差 |
| --- | --- | --- | --- |
| 同一层 | `+1.60 ms` | `15.97 ms` | `67.18 ms` |
| **上一层** | `−1.27 ms` | **`1.25 ms`** | `4.29 ms` |

## 代码变更

只在 `exp/fixed-window`（探针分支，未合并），五个文件：

- `generate.rs`：层循环里给 join 前后打点，记下预取线程自己报的 `worker_start_ns` / `worker_end_ns`，
  并给 spawn 打点；把窗口减去 join+h2d+spawn 记成残差。
- `b2_runtime.rs`：在 `load_b2_raw_runtime_from_host_for` 末尾显式计时释放 `host` 和 `raw_wo_a_host`
  （原来是隐式析构，位置相同，只是量出来）。
- `v4_state.rs`：`read_u8` / `read_bf16` / `read_f32` 里把 `vec![0; n]`、预写、`read_at` 分开计时，
  累加到三个线程本地计数器。
- `layer_loading.rs`：在三个 loader 前后重置/取出那三个计数。
- `fixed_prefetch.rs` / `b2_generate.rs`：`FixedLoadTelemetry` 加 10 个探针字段，
  `top_level.fixed.window` / `.worker_split` 里输出。

**为什么这样改**：解码路径传的是 `transition: None`，预取线程的 `worker_start_ns/worker_end_ns`
（`fixed_prefetch_transition_json` 里现成的 `consume_wait_ns`）只走预填充那条路，解码这条路什么都没记。
在没有测量之前，只能靠"字节没变、墙变了"猜，而上一份记录对同一批字节已经猜错过两次。

## 未决事项

1. **复用两个主机槽位（这就是下一步）。** 把预取这条路改成"两个槽位轮转、提前一层、复用主机缓冲"，
   对齐仓库里已有的 `PinnedFixedSlot` / `prepare_stationary_raw_pinned`。按上面的构成，
   预期 worker 掉到 `78–98 ms`、`join` 与 `release` 归零、`fixed` 的墙只剩 `23.4 ms`
   （`≈1.6–1.8 s/token`）。改完用四条复核：`release` 是否归零、`join` 是否变小、
   8 个 digest 是否逐个不变、峰值显存。
   > **这条已经做完了**，但用的是另一种形态：不是两个槽位轮转，而是**每层解出的负载跨 token 常驻**
   > （`exp/fixed-host-reuse`，`COLIBRI_V4_FIXED_HOST_RESIDENT_GIB`）。原因见
   > `docs/experiments/2026-09-16-sm120-fixed-host-residency.md`：填进已有缓冲要改整套 loader
   > （`DecodedLayerHost` 30 个字段），而保留整份负载不用改 loader，还顺带省掉每 token 重读的 5.9 GB。
   > 实测 `release` 归零、预取线程 `162 → 25–34 ms`、8 步 `−7.6% / −11.3%`，代价是 RSS +6.5 GB。
   > 上面那句"`≈1.6–1.8 s/token`"是按**被探针加了一次写的**那轮数字估的，实测是 `0.7–1.2 s/token`。
2. **`other` 那 58 ms 值得单独看一眼，而且和复用正交。** 里面已知有三样：
   `read_matrix` 对全量 137 MB 做 `any()` 找 NaN（这只是读一遍内存，量级约十几毫秒）、
   每层重新解析 `model.safetensors.index.json`（`base_indexed_model` 在三个 loader 里各调一次）、
   每层重新 `ShardFile::open`。后两样缓存一下就能省，不需要动缓冲所有权。各占多少没分过。
3. **主线程花 17.4 ms 还内存这件事本身值得留意**：申请在预取线程上（离关键路径远），
   释放却在主线程上。就算复用做到一半，也应该让释放落在不被等的线程上。
4. 旧的、还挂着的：`shared` `45.4 ms/层` 仍然没有内部计时（现在是第二大项）；
   attention `31.2 ms/层` 一层十来个同步点没动；预填充没接（真实 prompt 每 token 约 17–19 s）；
   `exp/slot-reuse` 要不要合并进交付分支仍待决定（细节在 `docs/experiments/2026-09-16-sm120-slot-reuse.md`）。
