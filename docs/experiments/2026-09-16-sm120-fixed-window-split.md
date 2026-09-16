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
提交 `b8cd920`（拆 join/spawn/残差）、`190f1cf`（加 release 计时），父提交 `c3c1be8`（槽位复用）。
测完已切回 `codex/business-restart`（`c8f206d`）、工作区干净、二进制重建。

构建 `18.90 s` / `18.18 s`、exit 0、**59 条警告**（插桩没有多出警告）。环境变量与槽位复用那次完全一致
（`BASE=1`、`TOKEN_IDS=0`、`GREEDY_STEPS=8`、`STREAM_READ_MODE=buffered`、`PREFETCH=1`、
`MHC_LINEAR_ARM=cublas`、`ROUTER_LINEAR_ARM=cublas`、`ROUTER_SCORE_ARM=libdevice`；
`FP4_ACCUM_ARM` / `FINAL_HEAD_ARM` 不设）。

结果文件（都在 `/srv/localmoe/targets/`，都是新名字）：

- `fixed-window-8step.jsonl`（第一次插桩，`79.55 s`）
- `fixed-window2-8step.jsonl`（加 release 计时，**`79.44 s`**，下面所有数字出自它）
- `post-fixed-probe-1step.jsonl`（切回交付分支后重建的二进制，单步四个钉死值复现）
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

只在 `exp/fixed-window`（探针分支，未合并），三个文件：

- `generate.rs`：层循环里给 join 前后打点，记下预取线程自己报的 `worker_start_ns` / `worker_end_ns`，
  并给 spawn 打点；把窗口减去 join+h2d+spawn 记成残差。
- `b2_runtime.rs`：在 `load_b2_raw_runtime_from_host_for` 末尾显式计时释放 `host` 和 `raw_wo_a_host`
  （原来是隐式析构，位置相同，只是量出来）。
- `fixed_prefetch.rs` / `b2_generate.rs`：`FixedLoadTelemetry` 加 7 个探针字段，
  `top_level.fixed.window` 里输出。

**为什么这样改**：解码路径传的是 `transition: None`，预取线程的 `worker_start_ns/worker_end_ns`
（`fixed_prefetch_transition_json` 里现成的 `consume_wait_ns`）只走预填充那条路，解码这条路什么都没记。
在没有测量之前，只能靠"字节没变、墙变了"猜，而上一份记录对同一批字节已经猜错过两次。

## 未决事项

1. **预取线程那 162 ms 还没拆。** 它 = 从盘读 137 MB + 往新申请的页里写的缺页开销，两者各占多少
   没测。这个比例决定复用主机缓冲能捡多少：如果缺页占大头，worker 会掉到空闲线以下，`join` 归零；
   如果读占大头，复用只能捡回 release 那 `0.75 s/token`。做法：在 `read_bf16` / `read_f32` / `read_u8`
   里把 `vec![0; n]` 与 `file.read_at` 分开计时（往新页里先写一遍逼出缺页，再读）。
2. **把两个主机槽位轮转接进 flash B1 解码这条路**，对齐仓库里已有的 `PinnedFixedSlot` /
   `prepare_stationary_raw_pinned`（它们已经是"两个槽位、提前一层、复用 pinned 缓冲"）。
   改完用同样的四条测量复核：release 是否归零、join 是否变小、8 个 digest 是否逐个不变、峰值显存。
3. **主线程花 17.4 ms 还内存这件事本身值得留意**：申请在预取线程上（离关键路径远），
   释放却在主线程上。就算复用做到一半，也应该让释放落在不被等的线程上。
4. 旧的、还挂着的：`shared` `45.4 ms/层` 仍然没有内部计时（现在是第二大项）；
   attention `31.2 ms/层` 一层十来个同步点没动；预填充没接（真实 prompt 每 token 约 17–19 s）；
   `exp/slot-reuse` 要不要合并进交付分支仍待决定（细节在 `docs/experiments/2026-09-16-sm120-slot-reuse.md`）。
