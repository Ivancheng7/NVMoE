# 固定层设备槽位进交付分支：验证记录（2026-09-17）

## 结论

**探针树里量好的"固定层设备槽位"已经搬进交付分支并真机验证**（`7cdd0b1`，默认开，
`COLIBRI_V4_FLASH_FIXED_SLOT=0` 关）。原状是每层重建一遍 `LayerDeviceWeights`（约 20 次
`cuMemAlloc` + 20 次 `cuMemFree`），现在按层型分三个设备槽位、每层把宿主负载拷进去（restage）。

8 步、真实命令行、两个顺序各跑一对，结论三条：

1. **数值逐位不变。** 开着那臂的 8 个 `hidden_bf16_digest`、8 条 43 层摘要链、8 个 argmax 与
   `defaults-pc-8step` 逐项一致，**唯一差异是显存峰值** `1500053504 → 1716060160`（`+216 MB`，
   就是三个槽位本身，和探针那次逐字节相同）；关掉那臂与参照 **IDENTICAL**（连峰值都一样）。
   1 步跑的四个钉死值也逐项复现。
2. **稳态 `fixed` 从 `1.02` 掉到 `0.73–0.75 s/token`（`-0.28`）**，两个顺序都是这个数——
   而同一个 token 的 `outer` 差值（`-0.20 ~ -0.27`）**正好等于 `fixed` 的差值**，说明省下的时间
   没有被别的相位吃掉。
3. **整请求墙：两个顺序里"开着"那臂都更快**（`-0.55 s` 与 `-3.23 s`，8 个 token 的请求）。
   这个跨度不是改动的收益，是**跑的顺序**：先跑的那一臂要付冷页缓存里那 6.7 GB 固定权重的读。
   第一对（开先跑）里 token 0/1 看起来慢 `0.42 / 0.68 s`，换序之后**反过来**（快 `0.09 / 0.42`），
   所以那个差是顺序差，不是槽位的代价。

**没声称的部分（别照着脑补）**：

- **服务路径一次没跑。** 槽位在 serve 面上同样默认开，但这次只用命令行量过。
- **"省下的是分配/释放"是引用来的**：那 `414 ms/token` 出自 9-17 的引擎内插桩
  （`2026-09-17-sm120-fixed-h2d-bandwidth.md`），不是这次测的。这次只测到墙。
- **`attention` 在这四次跑里都异常高**：`1.29–1.41 s`（占 `33–37%`），而 11:00 那帧是
  `0.89–1.01 s`。**两个臂一样高**，所以它不影响这次的 A/B，但它本身**没查**（不是本次改动引起的，
  因为关掉那臂同样高）。
- **早先那个 `42.41 s` 的整请求读数今天在四个臂里都没复现**（都在 `46.4–49.6 s`）。整请求墙
  今天不适合当判据，相位才是。
- 没走 `rust/run-identified.sh`，结果文件里**没有源码指纹**；下面的源码树 + HEAD + 二进制 SHA-256
  是这次能对上的身份。
- 两份批日志当时写在 `/tmp`（tmpfs），**已经丢了**；结果 jsonl 都在 `/srv` 上。脚本已改成把日志
  写进结果目录（`$OUT/batch.log`），这次之后不会再丢。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- `48` 个 safetensors，合计 `166886535336` 字节（整个目录 `166898773853`）
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- 与 `2026-09-17-sm120-attention-split-new-default.md`、`...-fixed-h2d-bandwidth.md` 同一份权重、同一台卡

## 运行证据

源码树 `/srv/localmoe/build/merged-prompt-chunk`（`CARGO_TARGET_DIR=/srv/localmoe/targets/merged-prompt-chunk-target`）：

| 批 | 签出的提交 | 说明 |
| --- | --- | --- |
| `run1` | `e156c4e` | 先跑**开着**那臂 |
| `swap` | `9e0c6ae` | `ORDER=off-first`：先跑**关掉**那臂 |

两个提交之间只差 `tools/` 里的脚本，所以两次跑的是**同一个二进制**：

```text
release/colibri-cuda  SHA-256 9373dda9c00262dfbe9704fa23e716c8ba313b59ce6b308166559c4d50366b2f
cuBLAS   /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5
```

一条命令跑完一批（同步 + 构建 + 1 步 + 两臂 8 步 + 逐位比对 + 相位账）：

```bash
bash rust/colibri-cuda/tools/verify_fixed_slot.sh <commit> [tag]      # 先跑开
ORDER=off-first bash rust/colibri-cuda/tools/verify_fixed_slot.sh <commit> [tag]   # 先跑关
```

结果文件（WSL 路径，不在仓库里）：

```text
/srv/localmoe/targets/fixed-slot-run1/{1step-slot-on,8step-slot-on,8step-slot-off}.jsonl
/srv/localmoe/targets/fixed-slot-swap/{1step-slot-on,8step-slot-on,8step-slot-off}.jsonl
```

两臂只差一个变量：`8step-slot-off` 带 `COLIBRI_V4_FLASH_FIXED_SLOT=0`；`1step-slot-on` 是
`GREEDY_STEPS=1`、不带遥测，另外两跑是 `GREEDY_STEPS=8` + `COLIBRI_V4_BENCH_TELEMETRY=1`。
其余环境两臂逐字相同（`FLASH_BASE=1`、`TOKEN_IDS=0`、`STREAM_READ_MODE=buffered`、
`MHC/ROUTER_LINEAR_ARM=cublas`、`ROUTER_SCORE_ARM=libdevice`；`FP4_ACCUM_ARM` 与
`FINAL_HEAD_ARM` 故意不设，用来确认默认臂仍是 `block32` / `canonical_cpu_streamed_bf16`）。

### 数值

```text
1 步（开着）     四个钉死值 + 43 层 + block32 + canonical 全 OK，峰值 1716060160，
                 request_e2e 15.33 s（swap）/ 17.32 s（run1）
8 步 开着        DIFFERS: peak device bytes      ← 只差峰值（1716060160）
8 步 关掉        IDENTICAL                        ← 连峰值都回到 1500053504
```

### 墙（单位 s，token 2–7 均值）

| 批 | 臂 | `fixed` | `outer` |
| --- | --- | --- | --- |
| run1 | 开 | **0.75** | 3.80 |
| run1 | 关 | 1.03 | 4.07 |
| swap | 开 | **0.73** | 3.84 |
| swap | 关 | 1.02 | 4.04 |

冷帧（token 0 的 `fixed`）：run1 开先跑 `8.30` 对关后跑 `7.88`；swap 关先跑 `9.13` 对开后跑 `8.22`
——**先跑的总是更慢**，与臂无关。整请求墙：run1 `47.56`（开）/ `48.11`（关）；swap `46.39`（开）/
`49.62`（关）。

## 代码变更

- **新增 `rust/colibri-cuda/src/fixed_layer_slot.rs`**：三个槽位（window-only / Hca128 / Csa4）
  加一个 spare indexer 缓冲。Csa4 与 Hca128 逐层交替，混在一个槽位里永远匹配不上
  （探针第一版就是这么写错的）。槽位由 Flash B1 请求驱动持有，与 `host_cache` 同寿命。
- **`device_weights.rs`**：加 `restage_copy`（长度不一致直接报错，不静默读旧数据）、
  `DeviceFp8Matrix::restage_from`、`DeviceLayerIndexer::restage_from`、`LayerDeviceWeights::restage_from_host`。
  字段表逐项对齐 `new_layer_with_wo_a`（含 `hash_router` 的两种分支），`wo_a` 保持"由调用方
  解码后换进来"的例外。
- **`b2_runtime.rs`**：加 `flash_b1_fixed_slot_enabled`（两个交付面默认开，和 chunk / 预取同一个写法）
  与 `load_b2_raw_runtime_from_host_ref_with_weights`（`_for` 变成它的 `weights=None` 包装）。
- **`generate.rs`**：层循环持有槽位，加载时先 `take_restaged`、一层跑完 `release`。
- **`tools/verify_fixed_slot.sh`**：新增。一条命令跑完一批（含构建与逐位比对），跑前自己查机器空不空，
  支持换序（`ORDER=off-first`），日志写进结果目录。
- **为什么要改**：这是"量完就搬"这条新规矩的第一次执行——两件已量未搬的改动里较大的那件
  （探针 A/B：`fixed 1.02 → 0.71 s/token`、逐位不变、`+216 MB`）。
- **故意没改的**：`fixed_h2d_bytes` 那个字节账照旧。restage 拷的是同一批张量，`num_bytes()`
  量的就是它，所以两臂字节账相同；`fixed_h2d_ns` 在 restage 分支只覆盖 shard 打开这一步，
  判据是 `fixed` 的墙 + digest（代码注释里写明了）。

## 未决事项

1. **压缩器主机缓冲复用（第二件已量未搬的）——它不是两行的活，形状先说清楚**：探针 A/B 量的是
   `layer_ops.rs::run_compressor_official_cublas_host` **同一个函数内**复用两块主机展开缓冲
   （`420.5 → 369.4 ms/token`，热帧）。仓库里现成的 `StrictCompressorScratch`（`run_from_device`
   那条路）**已经穿到 `run_real_layer_attention_rows` 的签名上**，但交付解码路径传的是 `None`；
   给它传 `Some` 会**换成另一条实现**（不是被 A/B 过的那条），数值和墙都要重新量。
   要么按被量过的形状做（把主机缓冲复用做进 `_host`），要么先量 `run_from_device` 再决定。
2. **服务面没量**：`COLIBRI_V4_FLASH_FIXED_SLOT` 在 serve 上也默认开，但一次 serve 没跑过。
   按服务重验一次（token id + 那条路上的钉死值）。
3. **`attention` 今天为什么全体高 30%**（四次跑 1.29–1.41 s 对 11:00 的 0.89–1.01 s）：两臂同样高，
   不影响这次的 A/B，但它会污染任何跨时段的比较。要查就先插桩。
4. **`shared` 的 0.65 s 仍然没拆**（老问题，且它可能推翻整张相位加法表）。
5. **整请求墙跨时段不可比**：`42.41 / 47.0 / 47.6 / 48.1 / 46.4 / 49.6 s` 都是 8 代 token 的请求。
   判改动用相位和 digest，整请求只在同一批里比。
