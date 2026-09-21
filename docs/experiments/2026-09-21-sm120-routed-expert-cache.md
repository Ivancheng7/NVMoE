# sm_120 routed 专家身份缓存：读盘降 57%，墙几乎没动（2026-09-21）

## 结论

给 routed 专家读盘加了一个**按专家身份 `(layer, expert)` 键的请求级缓存**，默认关
（`COLIBRI_V4_ROUTED_EXPERT_CACHE=<槽数>`），改动**逐位不变**：打开了之后 1 步的四个钉死值
全部复现，8 步的 8 个 `hidden_bf16_digest`、8 条层链、argmax 与关着时、以及与常驻默认开的
参照 `defaulton` 都**逐位相同**。

它确实命中：引擎账里的 `routed.diagnostics.cache_hits` 在一次 8 步单 token 请求里记到 **1028**，
与我在跑之前离线数 route 流得到的 1028 完全一致；每 token 的**磁盘读从 258 次降到 111 次**，
`source_bytes` 从 `3.21 GiB` 降到 `1.38 GiB`。

**但墙几乎没动。** 稳态 `shared.wall` 只降 `16.26 → 14.29 ms/层`（约 `0.08 s/token`），而且那
`routed.wall` 本身还在噪声里晃（`0.802 → 0.804`）。原因这就是 A3 刚补上的账：**读盘那笔本来就被
shared MLP 藏住了一部分**——A3 之前大家以为 shared 行四分之三是读盘、专家 kernel 干等数据，现在
量清楚了，`shared` = MLP `4.1 ms` + **等 worker 的 join 尾巴**；缓存把 worker 的读墙砍短了
（六条 lane 求和 `61.3 → 40.4 ms/层`，最慢那条 `15.8 → 13.4`），**可墙的收尾仍由 join 决定，
而 join 里还有 MLP 和排空的开销垫着**，所以省下的读时间只露出一小截。

**没声称的部分**：

- **没有自称是加速**。它在单 token 那组省约 `0.08 s/token`（`outer 2.228 → 2.148`），在做 prompt
  变体的那组 `outer` 反而 `+0.06`（噪声）。两次批的盘都落在**慢档**（`1.7–1.8 GB/s`，见下），
  绝对时间不可跨时段比。
- 只在 `FLASH_BASE + PREFETCH=1` 这条路上验过；serve、QD4、stationary 等路径没碰。
- 缓存尺寸（槽数）与命中率的完整曲线**没在引擎里测**，只有离线按 route 流算的 LRU 曲线（见下）。
- 缓存用**锁页主机内存**（每槽 `12.75 MiB`），默认关所以不占；开 `258` 槽约 `3.2 GiB`。

## 固定输入与模型

- 模型 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，48 分片、`166898773853` 字节
- `config.json` sha256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` sha256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- RTX 5060 Ti 16 GB（`sm_120`）、cuBLAS `12.8.5.5`

## 运行证据

构建（WSL 交付树 `/srv/localmoe/build/deliver-merge-0918`）：

```bash
CARGO_TARGET_DIR=/srv/localmoe/targets/deliver-merge-0918-target \
  ./rust/run-identified.sh build --arch sm_120 -- --release -p colibri-cuda
# 62 条警告（与改动前基线相同），Finished release ~20 s
```

盘速门：两次 A/B 批跑前各量一次 `dd iflag=direct bs=1M count=2048`，分别 `1.8` 和 `1.7 GB/s`
——**都落在慢档**（历史快 `3.66`/慢 `2.7`）。因此**本批的绝对时间不作收益判据**，只看字段关系和
逐位一致性。

结果文件（`/srv/localmoe/targets/`）：

- `a2-ab/`：交错 A/B，单 token 形状 `single-r{1,2,3}-{off,on}.jsonl`、真 prompt 形状
  `real-r{1,2,3}-{off,on}.jsonl`（off = `COLIBRI_V4_ROUTED_EXPERT_CACHE=0`，
  on = `258`；其余臂同常驻默认）
- `a2-final/{off,on}-{1step,8step}.jsonl`：收口验收（单 token 形状，8 步用 `TOKEN_IDS=0`）
- 离线 route 流分析脚本与真 prompt：`.zcode-a2-*.py`（本次会话临时，未入库）

关键数字（稳态 token 1–7，均值）：

| 指标 | off | on | 说明 |
| --- | --- | --- | --- |
| 磁盘读/ token | 258 | **111**（单 token）/ **54**（真 prompt） | `expert_loads` |
| `source_bytes`/token | 3.21 GiB | 1.38 / 0.68 GiB | 命中槽不读盘 |
| `cache_hits`（全程） | 0 | **1028** | 与离线数路由流一致 |
| `shared.wall`/层 | 16.26 ms | 14.29 ms | 会话内省窗口 |
| ├ MLP `main` | 4.12 | 4.06 | 不变 |
| ├ `join_tail` | 12.15 | 10.23 | 读墙被砍后仍收尾 |
| └ `read_worker_sum` | 61.34 | 40.41 | 六条 lane 读墙求和 |
| `routed.wall`/token | 0.802 s | 0.804 s | 噪声内 |
| `outer`/token | 2.228 s | 2.148 s | 约 `−0.08`，慢档批 |

### 离线 route 流分析（跑之前的判据）

用引擎自己记的 `route_ids`（8 步单 token × 43 层），做按位置的 LRU：

- 全局 LRU 需要**整 token 工作集（258 槽 / 3.2 GiB）才开始命中**：`256` 槽以下命中率是 `0`，
  `258` 起跳到 `43.4%`、`344` 到 `52%`。**复用距离是一整趟 43 层**（中间隔 257 次别的读），
  所以小缓存必抖动。
- 静态"钉最热专家"上界在 `258` 槽时 `54%`——比 LRU 的 `43%` 好，但差距不大。
- 真 prompt（32 个不同 id、生成 8 步）上复用更强：连续 token 同层重叠 `63.2%`，8 步内每层只碰
  约 24/256 个专家。**注意单 token 形状（`TOKEN_IDS=0`）的 prompt 只有一个 id，会自回归成重复模式
  （第 5/6/7 个 token 的 route 行逐字节相同），复用率偏高**；所以真 prompt 那组是更可信的样本。

### 锁页可行性（改之前验的）

`cudaMallocHost` 344 × 12.75 MiB = `4.28 GiB` 连续锁页成功（`ulimit -l` 显示 64 MiB，但实际
分配不受它限制），所以"整 token 工作集"这个量级的锁页缓存是放得下的。

## 代码变更

`rust/colibri-cuda/src/`，提交 `10f0958`（交付分支 `codex/business-restart`）：

- `expert_slot.rs`：新增 `RoutedExpertCache`——按 `(layer, expert)` 存**整条锁页 lane**
  （命中即零拷贝交给 H2D），带 clock-LRU 淘汰；`RoutedReadPool` 改成按值 `take_lanes`/`give_lanes`
  借还，让池子和缓存共用同一条 worker/排空路径；`RoutedSlotScratch` 多一个 `Option<RoutedExpertCache>`
  和 `ensure_cache`/`take_cache_lanes`/`give_cache_lanes`。
- `layer_rows.rs`：`routed_expert_cache_capacity()`（读 `COLIBRI_V4_ROUTED_EXPERT_CACHE`，默认 0）；
  预取块里 lane 从缓存或池子取，miss 才 `read_into`；排空按 route 槽 `upload_lane`，命中槽不记
  `source_bytes`/`expert_loads` 但照常 H2D；层末把 lane 还给缓存或池子。
- `flash_arms.rs`：`LayerStageTelemetry` 加 `routed_cache_hits`（并入 `fold!`）。
- `b2_generate.rs`/`phase_profiles.rs`/`full61_b1.rs`：`routed.diagnostics` 增发 `cache_hits`。

**为什么不改数值**：命中槽的 lane 里就是 `read_into` 会读进来的同一批字节，H2D 照做，
`hidden/route/KV` 摘要不变。相位 JSON 不进 `ordered_layer_digest_sha256`。

**踩过的坑**：第一次批的 `cache_hits` 全是 `0`，因为构建脚本只同步了 3 个文件、漏了发字段的
`b2_generate.rs`/`phase_profiles.rs`；补上重建后才是 1028。**构建脚本要按 `git status` 同步全部改动文件**。

## 未决事项

1. **要不要默认开。** 现在默认关。它逐位安全、命中率真实，但在现有形状下**收益很小**（读盘大半被
   MLP 藏住，省下的只露出约 `0.08 s/token`，慢档批内还在噪声边缘）；而代价是 `3.2 GiB` 锁页主机内存
   （机器共 15 GiB）。**这是 A1 的前提**：只有先把读盘从 `shared` 窗口里挪到别处、让它不再被藏住，
   缓存的收益才会全额上墙——**A2 和 A1 应当一起判**。
   > **2026-09-21 已判，上面这个预期只对了一半，作废一半。** A1 落地（fan-out）后同批三臂交错：
   > 真 32 token prompt `off 1.943` / `fanout 1.619` / `fanout+cache258 1.549 s/token`，
   > 单 token 形状 `2.193` / `1.791` / `1.715`——**A2 在 A1 之上只再省 `0.07–0.08 s/token`**，
   > 而整条请求上（8 步）它基本没用：`fanout` 对 `fanout+cache` 的 e2e 是 `68.6` 对 `69.1 s`、
   > `30.45` 对 `30.28 s`，稳态省下的刚好被它自己变慢的第一个 token 吃掉。
   > 磁盘流量确实降了（真 prompt 每层 `6.0 → 1.0` 次载入、`3.21 → 0.86 GiB/token`），
   > 但"读盘不再被藏住"不等于"读盘变成墙"：fan-out 之后一层是**最慢那条 lane 的读→上传→内核**在收尾，
   > 缓存砍得掉其它五条（`lane_max 19.45 → 16.83`），砍不掉那条 miss 的 lane。省下的是六条并发读
   > 之间的竞争，不是读盘本身。数字以 `docs/experiments/2026-09-21-sm120-routed-fanout.md` 为准。
2. **缓存尺寸曲线没在引擎里测。** 只有离线 LRU/钉热曲线。真机上 `258` 槽 vs `344` vs 更小，命中率与
   墙的对应关系要一次扫出来（清缓存的交错批）。
3. **只覆盖 `FLASH_BASE + PREFETCH=1`**。serve、QD4、stationary、多行未接。
4. 锁页主机内存的**释放与上限**：请求结束后缓存随 scratch 析构，但没测过多次请求累积/碎片。
5. 命中率对 prompt 高度敏感（自回归 prompt 会自造重复）。要在别的真 prompt 上重复本测量。
