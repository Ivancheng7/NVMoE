# 读盘直接进锁页：省掉 restage，routed 窗口每层省 4.3 ms（2026-09-18）

## 结论

把 B1/serve 预取路径上那笔主机侧双搬运去掉了：routed 专家的读**直接落进
请求级常驻的锁页读槽**（六个路由槽各一条，正好对上六条读线程），不再"先读进
普通 `Vec`、再由 `restage_weights` 六块 `copy_from_slice` 进槽位"。H2D 的源
从槽位自己的暂存改成读槽，设备侧仍然只有一个执行槽位。

同批交错 A/B（每次清缓存、两个二进制轮着跑，共 6 轮，8 步 base 路径）：

| 指标（每层平均 ms） | 基线（改动前） | 读进锁页 | 差 |
| --- | --- | --- | --- |
| `routed.wall` | 22.86 | **18.45** | **−4.32**（六轮 −4.19…−4.55，很稳） |
| 其中 `setup` | 10.66 | 6.24 | −4.39 |
| 其中 `alloc` | 4.94 | **0.00** | −4.94 |
| `shared` | 17.26 | 16.41 | −0.88 |
| `outer`/层 | 110.80 | 103.78 | −7.01 |
| 八步墙 | 38.95 s | **36.54 s** | −2.42 s |

**八个 `hidden_bf16_digest` 逐位不变**（`5ed8bf88beae3cfe` … `69ed859929ee9042`），
六次 A/B 十二跑全部相同；单步 `hidden/sha256/L0 route_ids/43 层` 也与交付参考全同。

`alloc` 恰好归零是这条路径真的走通了的开关：旧路径每个张量 `vec![0; N]`（每个
专家 6 次），新路径一次都不申请。`setup` 从 10.7 掉到 6.2 ms/层，剩下的是 H2D
本身和事件记账，`restage` 的六块拷贝没了。

没声称的部分：

- **收益按每 token 算只有约 0.30 s**（八步 −2.42 s）。routed 窗口省的 4.3 ms/层
  里，有相当一部分本来就被 `fixed` 抢盘吃掉（`fixed` 六轮 −2.64 但方差大、
  有正有负），最后落到墙上的是这 0.30 s。这是本轮实际到手的数，不是预期。
- **只覆盖预取这条路**：`COLIBRI_V4_FLASH_BASE=1 + PREFETCH=1`（base/serve 会
  走它）。QD4、stationary、prefill 这些调用点没有 `routed_slot_scratch`，读槽
  池为 `None`，仍走旧的 `Vec` 路径，行为一字未变。
- **改动第一版我把 读∥共享MLP 的重叠弄丢了**（见下"自己引入又修掉的问题"），
  修掉后的数字才是上表；没修那版的 `shared` 会比基线还高 2.6 ms/层。
- 本轮 6 轮里 `fixed` 的方差压不住（−13.8…+8.9 ms/层），所以八步墙的绝对值只在
  **同一轮交错对**里比；跨轮不比的规矩还被复现了一次。
- 峰值显存基线本分支就是 `3964207104`（不是记录里常见的 `1466499072`，那是
  2026-09-16 的旧状态），两个二进制**完全相同**；本改动不新增任何设备缓冲
  （只多 6×12.75 MiB 锁页主机内存）。核数、正确性检查没有别的变化。

## 固定输入与模型

- 模型：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，48 分片、`166898773853` 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- 机器 Ryzen 5 7500F（6 核 12 线程）、15 GiB 内存、`/dev/sdd` ext4（WSL2 虚拟盘）、RTX 5060 Ti 16 GB
- 两个二进制都是 `./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`：
  - 基线：改动前提交 `653dccf`，`CARGO_TARGET_DIR=/srv/localmoe/targets/localmoe-baseline-routed-target`
  - 改动：`95157fc`（本分支尖），`CARGO_TARGET_DIR=/srv/localmoe/targets/localmoe-pinned-target`
- 实验条件（两个二进制完全相同）：`COLIBRI_V4_FLASH_BASE=1`、`GREEDY_STEPS=8`、
  `TOKEN_IDS=0`、`STREAM_READ_MODE=buffered`、`FLASH_PREFETCH=1`、
  `MHC/ROUTER_LINEAR_ARM=cublas`、`ROUTER_SCORE_ARM=libdevice`、
  `FP4_ACCUM_ARM=block32`、`FINAL_HEAD_ARM=gpu_exact`

## 运行证据

- A/B 两批各 3 轮（交错、每次 `drop_caches`）：
  `/srv/localmoe/targets/io-ceiling/pinned-fix-{baseline,pinned}-r{1,2,3}.jsonl`、
  `pinned-fix2-{baseline,pinned}-r{1,2,3}.jsonl`；每跑的 stdout 在对应 `.stdout`
- 单步对照：`/srv/localmoe/targets/pinned-lanes-1step.jsonl`
  （`hidden_bf16_digest 5ed8bf88beae3cfe`、43 层、`L0 route_ids [254,222,245,200,53,35]`，
  与 `postmerge-routed-threads-1step.jsonl` 相同）
- 八步：`/srv/localmoe/targets/pinned-lanes-8step.jsonl`，八 token digest
  `5ed8bf88beae3cfe 316e5495ca424b39 a521547b2586e458 1963dfeaaf30264c
   b970da7913a57a1c 12c6b45cecf9d53b e0f441542fa53991 69ed859929ee9042`
- 判据脚本（都在 `exp/routed-pinned-replay` 分支 `.zcode/io-ceiling/`）：
  - `run_pinned_ab.sh`：交错 A/B，跑不进就报 FAIL（第一版脚本用 `>/dev/null` 吞错，
    导致一次 A/B 其实没跑成却打了"done"，已修）
  - `check_pinned_lanes.py`：单步 digest/层数/route 对比
  - `sum_pinned_ab.py`、`compare_pinned_phases.py`：逐层相位汇总
- 共享窗探针（`COLIBRI_V4_FLASH_SHARED_PROBE=1`）确认读与 MLP 重叠：
  基线 `window 19.9 = max(mlp 4.1, worker 19.4)`，改动后 `window 20.7 = max(mlp 4.0, worker 20.3)`

## 代码变更

`rust/colibri-cuda/src/`：

- `expert_slot.rs`：
  - 新增 `RoutedReadLane`（一条路由槽的六个锁页权重新存，`read_into` 用现成的
    `RoutedExpertRangePlan`，构造时校验形状、读取时校验 E8M0）、`RoutedReadPool`
    （六条读槽）。`RoutedSlotScratch` 多持一个 `Option<RoutedReadPool>`，加
    `take_pool`/`put_pool`。
  - 新增 `ExpertSlot::upload_lane(&RoutedReadLane)`：只做六个 H2D + 记时，
    不碰槽位自己的暂存，是 `restage_weights` 在锁页路上的对位实现。
- `layer_rows.rs`：预取分支在 `routed_slot_scratch` 在时建/取读槽池，worker 按
  chunk 领取各自的 `&mut` 读槽直接 `read_into`；消费端从池子逐条 `upload_lane`
  再 `execute`。没有 scratch 的调用点读槽池为 `None`，整段回落到旧的 `Vec` 路径
  （`new_or_restage`/`restage_weights` 保留给那条路）。层末把池还回 scratch。
- `generate.rs` 不动：两条生产路径本就建了跨层 scratch，池子挂在它上面。

### 自己引入又修掉的问题（值得记一笔）

第一版把 `run_layer_shared_expert` 挪到了 spawn 读线程**之前**，于是 `shared`
窗口从 `max(mlp, worker)` 变成 `mlp + worker`——正是"合成一块缓冲最容易顺手
把重叠也合掉"那句警告命中的情形。共享探针一眼看出：基线 `19.9≈max(4.1,19.4)`，
改动后 `19.1≈3.4+15.3` 的和。把 MLP 放回 spawn 之后、join 之前就恢复了
（提交 `d04fed5`）。要不是交错 A/B 里 `shared` 不降反升，这个回归会混在
"routed 省了 4 ms"里看不出来。

## 未决事项

1. **收益小（0.30 s/token）**：routed 窗口实际省 4.3 ms/层，但落到墙上的只有
   0.3 s/八步。要拿到更多，得让省下的时间不被 `fixed` 抢盘吃掉——那是另一个
   题（`fixed` 那条流的盘竞争），不在本次范围。
2. **只在 base/serve 验过**：QD4/stationary/prefill 的调用点仍是旧路径。它们的
   占比决定要不要一起改。
3. 池子现在是"每层 take、层末 put"，错误早退（`?` 返回）时池子不还给 scratch，
   下一层会重建——正确但会重付一次锁页。目前只在真的出错时发生，可接受；要
   收紧就得把池子做成 `&mut` 借出去而不是 move。
4. `COLIBRI_V4_ROUTED_READ_THREADS < 6` 时，每个 worker 领多条读槽（`chunks` 按
   线程数切），仍是每条读槽一个专家，行为正确；`> 6`（比如 12）时 chunk 大小
   为 1，多出的线程空转——`routed_read_threads` 默认 6，没测 12。
5. 与 2026-09-17 的锁页回放量测（`2026-09-17-sm120-pinned-read-replay.md`）互为
   闭环：那份证明"读进锁页不慢"，这份是引擎里兑现的结果。
