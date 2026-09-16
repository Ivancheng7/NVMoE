# routed 专家槽位跨层常驻：每 token 省 1.1 秒（2026-09-16 深夜）

## 结论

routed 预取那条路**每层重建一个专家槽位**，而建槽位最贵的一步是给 12.75 MiB 暂存做页锁定
（驱动 API 直接量：`cuMemHostAlloc`+`free` 一次 17.4 ms）。把它改成**整段请求只留一个槽位**：

| 指标 | 之前 | 之后 |
| --- | --- | --- |
| `routed.wall` | 43.7 / 50.0 ms/层 | **23.3 / 23.6 / 24.6 ms/层** |
| 其中 `routed.setup` | 24.7 / 28.7 ms/层 | **11.4 / 11.7 / 11.9** |
| 八步墙（清缓存、机器空闲、各两遍） | 54.82 / 52.51 s | **43.83 / 43.50 s** |

也就是 **每 token 省 1.1 s（17%）**，八个 token 省 9 s；**八个 `hidden_bf16_digest` 逐个不变**
（`5ed8bf88beae3cfe` … `69ed859929ee9042`），单步 `argmax=55222`、43 层都不变。
注意 `routed.wall` 掉 20 ms 比 `setup` 掉的 13.5 ms 还多：多出来的约 6 ms 是**层末销毁槽位**
（解页锁 + 释放 20 个设备缓冲），以前也算在 routed 窗口里。

**一个钉死值变了：采样峰值显存 `1466499072` → `1500053504`（+32 MiB，正是一个槽位的那堆设备缓冲）。**
原因查清了：全流程只有 `generate.rs:582` 一处采样，位置在**每层 wo_a 解码之前、routed 段之前**；
以前那里没有活着的槽位（上一层的槽位在层末就销毁），现在被 scratch 留着，于是被采到。
**真实峰值没有变大**——每层执行 routed 段时槽位本来就活着，只是那段没有采样点、一直没被记下来。
这次改动让采样值向真实值靠了一步，不是新增了 32 MiB 压力。

没声称的部分：

- **真实峰值仍然没有被采样**。想要它得在 routed 段里加采样点，那会让这个数再变一次，没做。
- 收益只在 `COLIBRI_V4_FLASH_BASE=1 + PREFETCH=1` 这条路上验过；`stationary`、serve、QD4
  这些路径的调用点这次统一传 `None`（保持它们原来的行为），**它们有没有收益没测**。
- `923 ms` 的 H2D 异常仍未解释（`2026-09-16-sm120-fixed-host-residency.md`）。
- 两次对照的机器状态不同：带干扰那批（预填充会话在跑）的墙不可比，只用它看 `routed.wall`
  这个 CPU 侧指标；干净那批是预填充会话结束之后跑的。

## 固定输入与模型

与今天前几份同一份权重、同一张卡、同一套臂：

- 模型 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，48 分片、`166898773853` 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- 机器 Ryzen 5 7500F（6 核 12 线程）、15 GiB 内存、`/dev/sdd` ext4
- 分支：`exp/routed-slot-resident`（`5272628`），已合进交付分支 `5a9e35c`
- cuBLAS `12.8.5.5`（SHA-256 `18a5df54…`），构建 `59` 条警告（与基线相同）

## 运行证据

结果文件（都在 `/srv/localmoe/targets/`）：

- 带干扰批（两个二进制交替，三次）：`slot-ab-before-{1,2,3}.jsonl`、`slot-ab-after-{1,2,3}.jsonl`
- 干净批（清缓存、机器空闲、两次）：`slot-clean-before-{1,2}.jsonl`、`slot-clean-after-{1,2}.jsonl`
- 单步与八步：`slot-resident-1step.jsonl`、`slot-resident-8step-default.jsonl`、
  `slot-resident-8step-resident-on.jsonl`
- 墙的汇总：`slot-ab-summary.txt`、`slot-resident-summary.txt`

关键对照（每层平均毫秒）：

| 运行 | `outer` | `routed.wall` | `setup` | `kernel` | `h2d` | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- |
| `slot-ab-before-1` | 156.9 | 50.0 | 28.7 | 11.0 | 5.7 | `1466499072` |
| `slot-ab-before-3` | 134.8 | 43.7 | 24.7 | 10.6 | 5.7 | `1466499072` |
| `slot-ab-after-1` | 129.8 | 24.6 | 11.9 | 11.0 | 5.7 | `1500053504` |
| `slot-ab-after-3` | 109.2 | 23.3 | 11.4 | 10.7 | 5.7 | `1500053504` |

### `setup_ns` 会重复计数 H2D（顺手查清的一件事）

`restage_weights` 的结尾是 `weight_end.record(); weight_end.synchronize();` 然后才返回，
所以 H2D 上传整段落在 `setup_ns` 的计时窗口里，而调用方又把 `weight_h2d_ms` 加进 `h2d_ns`——
同一段时间被记了两次。判据是改造后的和：`setup 12.4 + kernel 11.2 + h2d 5.8 = 29.4 ms/层`，
**大于** `routed.wall 25.5 ms/层`，不重复计数不可能出现这个和。
去重后这次改动省的是 `18.5 → 6.6 ms/层`（都减掉 5.8 的 H2D）；剩下的 6.6 里约 2.9 是 6 次重填拷贝、
约 3.7 是事件与记账，**没有查不清的残留了**（上一轮"还有 9.5 ms 不明"就是这个重复计数）。

### 复现

```bash
# 两个二进制交替（带干扰时也能比 CPU 侧指标）
bash .zcode/io-ceiling/run_slot_ab.sh
# 干净对照：每次 drop_caches，before/after 各两遍
# after 的二进制是带改动的构建，before 是交付树的构建
env ... COLIBRI_V4_FIXED_HOST_RESIDENT_GIB=7 <binary>
# 判据
python3 .zcode/io-ceiling/check_digests.py "/srv/localmoe/targets/slot-clean-*.jsonl"
```

## 代码变更

`rust/colibri-cuda/src/`：

- `expert_slot.rs`：新增 `RoutedSlotScratch<'a>`（持一个 `Option<ExpertSlot<'a>>`，`take_slot`/`put_slot`）。
- `layer_rows.rs`：`run_real_layer_rows` 多一个 `mut routed_slot_scratch: Option<&mut RoutedSlotScratch<'module>>`；
  预取分支从 scratch 取槽位、层末放回（放进不去的路径行为不变）。
- `generate.rs`：两处调用者在**跨层作用域**建 scratch —— `run_v4_b2_generate_arm` 建在 token 循环外面
  （和 `host_cache` 并排）、`run_flash_base_input` 建在层循环前面；这两处传 `Some`。
- 其余 13 个调用点（`b2_runtime.rs` ×3、`layer_stationary_prefill.rs` ×6、`full61_b1.rs`、
  `qd4.rs`、`qd4_gates.rs`、`generate.rs` 另 2 处）传 `None`。

没有改读取方式、没有改数值路径、没有插桩（59 条警告与基线相同）。

## 未决事项

1. **真实峰值要不要量**：现在的采样点看不见 routed 段内的槽位。想在记录里有一个真实峰值，
   得在 routed 段里加采样点——那会把 `1500053504` 再改一次，做之前先确认值得。
2. 别的执行路径（`stationary`、serve、QD4、`prepare_b2_prefix`）这次都传 `None`，
   它们的槽位仍是每层/每专家重建。要不要一起受益，取决于那些路径的实测占比。
3. `routed.kernel` `0.46 s/token`（`10.6 ms/层`，6 个专家）还没查过是否贴近算术下限。
4. `fixed.h2d` `1.16 s/token` 只能拿显存换（`exp/fixed-residency`）。
5. `923 ms` 的 H2D 异常仍未解释。
