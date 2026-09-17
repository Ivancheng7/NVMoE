# 固定权重的 H2D 带宽：锁页值多少、锁页本身多贵（2026-09-17）

## 结论

新默认帧里 `fixed` 是每 token `1.01 s`（占一层循环 `3.78 s` 的 27%），而它的内容现在只剩一件事：
**每层把 `137.4 MB` 固定权重从主机推上显存**（相位账 layer0：`cache_status=host_resident_hit`、
`read_alloc_ns=0`、`source_bytes=0`、`h2d_bytes=137.4MB`、`decode_ns=0`、`wall=20.7ms`）。
所以"这段还能不能更快"等价于"这 137.4 MB 能不能搬得更快"。**用 ctypes 直接调 `libcudart` 量了四种搬法**：

| 搬法 | 每层耗时 | 带宽 |
| --- | --- | --- |
| 普通内存，一次 137.4 MB | `11.07 ms` | `12.97 GB/s` |
| 锁页内存，一次 137.4 MB | `10.07 ms` | `14.26 GB/s` |
| 普通内存，20 次 6.8 MB（引擎现在的形状） | `11.60 ms` | `12.39 GB/s` |
| 锁页内存，20 次 6.8 MB | `10.06 ms` | `14.28 GB/s` |

三条结论：

1. **锁页只值约 10%**（`13.0 → 14.3 GB/s`）。"把常驻负载锁页就能快 2–3 倍"这个假设**不成立**，
   不要按它派活。
2. **碎片化（20 小块 vs 1 大块）只值 5%**，也不是问题。
3. **锁页本身很贵**：`137.4 MB = 146.8 ms`，`5.9 GB = 6389 ms`（`1.083 ms/MB`）。
   和 routed 槽位那次量到的 `1.36 ms/MB` 是同一量级，互相印证。整模型锁页要 6.4 秒一次，
   短请求根本收不回。

**因此真正的缺口是另一个数**：引擎实测每层 `20.6 ms`（`6.7 GB/s`），而同一台机器、同样的字节、
同样的"20 小块"形状只要 `11.6 ms`（`12.4 GB/s`）。**差的 `9–10 ms/层`（≈ `0.4 s/token`）没有解释。**
`DeviceBuffer::from_host`（cuda-oxide `crates/cuda-core/src/device_buffer.rs`）的实现是
`cuMemAlloc` + 从普通主机内存 `memcpy_htod_async`，而 `LayerDeviceWeights::new_layer` 每层要
建约 20 个张量 → **每层、每 token 约 20 次设备内存分配/释放**，挪出去的正是这一点。
本记录**没有**插桩验证这个归因，它只是当前最像的解释。

**没声称的部分**：这次是独立微基准（ctypes 调 `libcudart`），**没有跑引擎**，也没有改任何推理代码；
引擎那 `20.6 ms/层` 是相位账读数（`attn-split-pc-8step.jsonl`，token 7 / layer 0），不是同一次测量出来的。
微基准里设备缓冲是复用的，引擎里不是——这正是两条曲线之间的差。

## 固定输入与模型

与 `2026-09-17-sm120-attention-split-new-default.md` 同一份：模型目录
`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，源码树
`/srv/localmoe/build/merged-prompt-chunk`（HEAD `598958e`）。引擎那侧的相位读数取自
`/srv/localmoe/targets/attn-split-pc-8step.jsonl`。

## 运行证据

- 微基准源码与运行脚本：`.zcode/h2d_probe.py`（未跟踪，不进库；从 WSL 侧跑）
- 运行环境：WSL2、RTX 5060 Ti 16 GB、CUDA 12.8（`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12`）
- 原始输出（本次）：

```text
pageable, one big copy             1x137.4MB     11.07 ms/copy   12.97 GB/s
pinned,   one big copy             1x137.4MB     10.07 ms/copy   14.26 GB/s
pageable, 20 copies/layer          20x6.8MB    11.60 ms/layer  12.39 GB/s
pinned,   20 copies/layer          20x6.8MB    10.06 ms/layer  14.28 GB/s

pin cost: 137.4 MB = 146.77 ms  5.9 GB = 6388.96 ms (1.083 ms/MB)
```

- 一个弯路记下来：这台机器的 `gcc` 是 15，`nvcc`（CUDA 12.8）解析不了它的头文件
  （`__is_pointer` 未定义、`cospi/rsqrt` 异常规格冲突），`-allow-unsupported-compiler` 和
  `-ccbin gcc-14` 都没救。**结论：这类小基准直接用 ctypes 调 `libcudart`，别开 nvcc。**
  `.zcode/h2d_probe.cu`（没编过）和 `.zcode/h2d_probe.c`（同样编不过）留在库里当反面记录。

## 代码变更

没有改推理代码。

## 追加（同日）：引擎内的分配账，以及一条被旁路的老开关

### 老开关不能用：`COLIBRI_V4_DEVICE_FIXED_CACHE_GIB` 在 Flash 路径上是死的

先试了最便宜的路——`FixedLayerCache` 里本来就有 `device_hit`（整层权重留在显存、后续 token 完全不上传），
`COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=8` 跑 8 步（`/srv/localmoe/targets/devcache-8step.jsonl`）**结果一行没变**：
`fixed` 每 token 还是 `0.98–1.03 s`、`h2d=6708.9 MB/token`、峰值显存 `1500053504`（与默认完全一致），
逐层的 `cache_status` 只有 `host_resident_hit 301 / prefetched_host 42 / miss 1`——**没有一次 `device_fill` 或 `device_hit`**。

原因：默认翻成"主机常驻"之后，Flash 解码路径（`generate.rs:448`）走的是新的
`fixed_host_cache::FixedHostCache` + `resolve_host_layer`，**老 `FixedLayerCache::access` 根本没被调用**。
所以那组 `COLIBRI_V4_DEVICE_FIXED_CACHE_GIB` 之类的老开关在现在的默认配置下是摆设，
"换个环境变量就能拿 0.85 s"这条捷径不存在——要它得改代码。

### 引擎内插桩：分配/释放占 `fixed` 的 49%

探针打在 `layer_loading.rs::upload_layer_runtime_for`：在真实的 `LayerDeviceWeights::new_layer` **之后**
（有意放在后面，免得探针的分配去污染紧接着的真实上传），按同一层的字节总量做 20 次
`cuMemAlloc_v2`/`cuMemFree_v2`，分段计时。源码 `.zcode/fixed_alloc_probe_patch.py`，
结果 `/srv/localmoe/targets/fixedalloc-probe-1step.stderr`（43 层，1 步运行，`exit 0`）。

| 层型（设备字节） | 层数 | `build` 中位 | alloc 中位 | free 中位 | 拷贝（build−alloc−free） |
| --- | --- | --- | --- | --- | --- |
| `99 MB`（window-only） | 2 | 16.16 ms | 5.37 ms | 3.15 ms | 7.63 ms |
| `107 MB`（Hca128） | 20 | 17.12 ms | 6.21 ms | 3.69 ms | 7.22 ms |
| `127 MB`（Csa4） | 21 | 20.93 ms | 6.24 ms | 3.53 ms | 11.16 ms |

整 token（43 层）：`build = 847.8 ms`，其中 **分配 `259.5` + 释放 `154.9` = `414.4 ms`（49%）**，
拷贝 `433.3 ms`（51%）。和相位账的 `fixed = 1.01 s/token` 对得上。

**两条修法的价码因此是量出来的，不是估的**：

| 改法 | 每 token 拿回 | 代价 | 要做什么 |
| --- | --- | --- | --- |
| 一层大小的可复用设备槽位（跨层 restage） | **`414 ms`（约 11%）** | 显存 `+137 MB` | 写一个 restage（往已有设备缓冲里拷），槽位归请求所有 |
| 整模型常驻显存（43 层都不动） | **`848 ms`（约 22%）** | 显存 `+5.9 GB`（峰值 `1.5 → 约 7.4 GB`） | 把设备缓存接进新的主机常驻流程（老开关已废） |

**没声称的部分**：探针只量了"同尺寸 20 次 malloc/free 在这台机器、这个显存状态下要多久"，
**没有真的实现任何一种修法**，所以那 `414 / 848 ms` 是"这两段开销的当前值"，不是"改完必然拿到的收益"；
restage 自己也会有成本（`copy_from_host` 会 `stream.synchronize()`，是 per-tensor 同步，得挑异步接口）。
探针的 20 次分配会在真实上传之后释放，给驱动的分配器留下可复用的块——**这一条很可能让下一层的真实分配变便宜**，
也就是说探针本身对 `build` 有轻微影响，两边都要带着这个偏差读。

## 追加（同日）：可复用设备槽位的实测——`-0.31 s/token`，数值逐位不变

照 `ExpertSlot::new_or_restage` 的范式，在探针树里给 `LayerDeviceWeights` 补了
`restage_from_host`（按 `new_layer_with_wo_a` 的字段表把宿主负载拷进已有设备缓冲，
`copy_from_host_async_unchecked` + 长度校验），并加一个由 Flash B1 请求驱动持有的槽位
（`.zcode/fixed_slot_patch.py`，新文件 `fixed_layer_slot.rs` + `generate.rs`/`b2_runtime.rs` 各两处）。

**一个设计坑先记下**：Csa4 与 Hca128 是**逐层交替**的，一个槽位每 token 只能复用一次
（第一版就是这么写的，`restages` 每 token 只有 1）。必须按层型分三个槽位（window-only / Hca128 / Csa4），
或者做成按几何索引的槽位池。

结果（8 步、遥测开、`COLIBRI_V4_FIXED_SLOT_PROBE=1`，`/srv/localmoe/targets/slotkind-8step.jsonl`）：

| token | baseline `fixed`（`attn-split-pc-8step`） | 槽位版 `fixed` |
| --- | --- | --- |
| 0 | 9.456 s | 9.187 s |
| 1 | 1.071 s | 1.097 s |
| 2–6（稳态） | 0.998–1.032 s | **0.686–0.741 s** |
| 7 | 1.006 s | 1.236 s（噪声） |

- **稳态 `fixed` 从 `~1.02 s` 降到 `~0.71 s/token`：`-0.31 s`，约一层循环的 `-8%`。**
- **显存代价：峰值 `1500053504 → 1716060160`（`+216 MB`）**，就是三个槽位本身。
- **数值逐位不变**：`.zcode/outputs_match.py` 对交付参照的 8 个 hidden digest、8 条
  `ordered_layer_digest_sha256` 链、8 个 argmax、43 层全部 `IDENTICAL`；
  **唯一差异就是显存峰值**（那是预期的代价，不是数值）。单步四个钉死值也复现。
- 剩下的 `~0.585 s/token`（43 层 × 13.6 ms）**是 restage 自己的拷贝**，按 137 MB/13.6 ms ≈ `10 GB/s`，
  和这台机器量到的可达带宽（12.4 GB/s）只差两成——**非常驻路线上已经接近地板**，
  要再往下只能整模型常驻显存。

**账本会说谎，交付版要修**：`fixed_h2d_bytes` 是按缓冲尺寸算出来的
（`b2_raw_runtime_ledger(..., runtime.weights.num_bytes(), ...)`），槽位版实际只搬了 restage 那些字节，
但账本仍会写 `6708.9 MB/token`。判定这一改动的证据是 **`fixed` 的墙**和 digest，不是字节账。

**探针与交付的差距**（照这个改要补的）：探针用 `COLIBRI_V4_FIXED_SLOT_PROBE` 开关、
槽位由请求驱动持有（无跨请求复用）；交付版要想清楚"槽位活多久、显存预算怎么算"，
并把账本改成记真实搬运字节。**这次没有改交付分支上任何一行代码。**

**还没拆的**：`mhc`（`1.64 ms/层`、`70 ms/token`，是 16 个小步里最大的一步）走的是设备侧 cuBLAS 臂、
没有主机往返，却要 1.64 ms，里面是什么没看过；下一个探针应该打它。

## 未决事项

1. **引擎那 `9–10 ms/层` 的缺口要插桩定位**：把 `LayerDeviceWeights::new_layer` 按张量拆成
   `cuMemAlloc` / H2D 两段计时（照 `.zcode/compressor_probe_patch.py` 那套办法），
   一次 1 步运行即可判定是不是设备内存分配。
2. 若是分配：**一个可复用的设备槽位**（一层大小 `137 MB`，跨层反复写）能省掉分配/释放，
   保住那 `11.6 ms` 的搬运——上限 `0.4 s/token`，显存只多 `137 MB`。
3. 若想要那完整的 `0.89 s/token`，只有**整模型常驻显存**（`5.9 GB`，峰值 `1.5 → 约 7.4 GB`）。
   `exp/fixed-residency` 当年是因为"15% 换 6.56 GB"被否的；现在同样的绝对收益是 `3.78 s` 里的
   **23%**，值得用当前数字重新决定一次，而不是沿用旧结论。
