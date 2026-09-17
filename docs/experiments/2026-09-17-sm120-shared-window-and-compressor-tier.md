# shared 那 0.72 s 里没有 shared，以及 attention 两档的根因是一个写错的常量（2026-09-17 晚）

## 结论

三件事：一条**读账的更正**、一条**根因**、一条**修好之后的数字**。

### 一、`shared` 那一行不是 shared 的账

`shared` 相位的墙（`16.64 ms/层`，`0.72 s/token`）是一个**套住两件事的墙**：这一层的
shared MLP 在主线程跑，同时 6 个 worker 在并排读这一层的 routed 专家
（`layer_rows.rs` 的 `thread::scope`，读完 join 才结束这段计时）。探针把两边分开量：

| | prefetch 开（交付默认） | prefetch 关 |
| --- | --- | --- |
| shared 窗口 | `16.64 ms/层`（`0.72 s/token`） | `3.31 ms/层`（`0.14 s/token`） |
| 其中 shared MLP 自己 | `4.34 ms/层` | `3.31 ms/层` |
| 其中 routed 读盘（最慢那个 worker） | `16.20 ms/层`（`0.70 s/token`） | — |
| 6 个 worker 读盘时间之和 | `69.04 ms/层` | — |
| shared MLP 返回后还要等的 | `12.30 ms/层`（`0.53 s/token`） | `0.1 us` |
| **这一层由谁收尾** | **97% 的层是"等读盘"** | 100% 是 MLP 自己 |

所以那张加法表里的 `shared` 需要拆开读：**`0.14–0.19 s` 是 shared MLP，其余 `0.53–0.58 s`
是 routed 读盘**（读盘属于 routed 那一摊，只是墙记在 shared 名下）。shared MLP 本身
只占一个 token 的 `3.7%`，**它不是一个值得动的地方**。

顺带量到一个大数：同一套代码、同一份权重，**关掉预取 worker 之后 8 步从 `44.73 s`
变成 `96.45 s`**（2.2 倍）。这一条只是顺带看到的，没拆，别当独立结论用。

### 二、attention 两档的根因：压缩器权重的"展开 + 上传"，而且专门治它的开关从来没生效

先按测得的子步把档位钉住。同一天同一份代码，5 连跑（`compressorprobe-tier-a`）：
attention `0.914 / 1.337 / 1.387 / 0.927 / 0.912 s`——**三快两慢，同一台机器同一份代码**。

用现成的 bench 块把大档位定到"哪一类层"，再用新探针把一次调用切成九段（`bd313bc`/`b7da660`）：

| 段（`outputs=1024, ratio=4`，21 层/ token） | 快档 `r1` | 慢档 `r2` | 差 |
| --- | --- | --- | --- |
| `weights_f32_h2d`（wkv/wgate 主机展开 + 上传） | `6422 us` | `11394 us` | **`+4972`** |
| `row_loop` | `993` | `975` | `-18` |
| `row_upload` / `gemm1` / `gemm2` / `row_d2h` | `311 / 55 / 12 / 243` | `306 / 52 / 11 / 226` | 噪声 |
| 合计 | `7419` | `12373` | `+4954` |

三种 layout 的增量都只落在这一段，而且**与权重元素数成正比**：`+1201 / +2409 / +4972 us`
对应 `1.05M / 2.10M / 4.19M` 个元素，即约 `+1.15 ns/元素`。其余八段在两档之间平到 2% 以内。

**第二遍独立跑把档位定到了两个循环上。** 第二批（`arena-fix-b`）的关池子臂三跑里，
前两跑是慢档、第三跑是快档，而三跑之间只有这两个展开循环变：

| `outputs=1024, ratio=4` 的段（`us`） | off-r1（慢） | off-r2（慢） | **off-r3（快）** | on-r1（池子开） |
| --- | --- | --- | --- | --- |
| `expand_wkv` | `5221.9` | `5252.3` | **`1189.0`** | `716.3` |
| `expand_wgate` | `8062.8` | `7948.4` | **`1702.3`** | `799.4` |
| `h2d_wkv` / `h2d_wgate` | `2246.9 / 2040.7` | `2272.1 / 2055.8` | `2306.5 / 2062.5` | `2918.9 / 2074` |
| `row_loop` / `gemm1` / `gemm2` / `row_d2h` | `611 / 52 / 13 / 217` | `624 / 52 / 12 / 224` | `622 / 57 / 13 / 233` | `-` |
| 同一跑的 attention 墙 | `1.341 s` | `1.344 s` | **`0.950 s`** | `0.784 s` |

**H2D 三次跑完全一样（`2247/2272/2307`），变的是展开。** 所以档位不是 `DeviceBuffer::from_host`
那一步（申请 + 上传），就是主机侧那两个 `u16 -> f32` 循环：同一份输入、同一段代码，
一次 `2.9 ms`（`1189+1702`）另一次 `13.3 ms`（`5222+8063`），差 `10.4 ms` 正好是这一趟 attention 的差。

**然后是这个改动真正的发现**：`7e07056` 加到交付分支的"压缩器复用池"（`COLIBRI_V4_FLASH_COMPRESSOR_ARENA`，
默认开）**在交付帧里一次都没跑过**。分支探针（`b7da660`）打出来：

```text
COMPPROBE_BRANCH outputs=1024 ratio=4 strict=true scratch=true input_len=4096 arena=7168 pool_reuse=false
COMPPROBE_BRANCH outputs=512 ratio=128 strict=true scratch=true input_len=4096 arena=7168 pool_reuse=false
```

池子用 `v4_state::HIDDEN`（`7168`，Pro 的几何）建，复用判据也拿这个常量比；Flash 喂进来的是
`4096` 宽的一行，**判据永远不成立**，于是永远走那条"每次调用重新申请 2×16.8 MB 主机缓冲 +
设备缓冲"的逐行 cuBLAS 路径。`7e07056` 那次的 A/B（两臂差 `15.34` 对 `15.42 ms`）其实是
**开 vs 开**——开关两边跑的是同一份代码，所以"池子拿不掉慢档"这个结论不成立。

### 三、把池子接上（`9efd662`）：attention `1.34 → 0.79 s/token`，两档没了

交错跑 3 轮（`on r1, off r1, on r2, off r2, on r3, off r3`），两臂都开探针。**跑了两批**
（`arena-fix`、`arena-fix-b`），下面两列都是各自批内的数：

| `attention`（后 4 token 均值） | 第一批 | 第二批 |
| --- | --- | --- |
| 池子**开** | `0.789 / 0.792 / 0.810 s` | `0.784 / 0.785 / 0.785 s` |
| 池子**关**（= 修之前的行为） | `1.339 / 1.324 / 1.336 s` | `1.341 / 1.344 / 0.950 s`（第三跑撞上快档） |

| 其余 | 第一批 | 第二批 |
| --- | --- | --- |
| 8 步整跑（开 / 关） | `43.86/43.52/43.66` 对 `47.39/45.61/45.90 s` | `43.54/43.30/45.32` 对 `44.72/45.34/44.46 s` |
| `(1024,4)` 总的每调用（开 / 关） | `6407/6524/6644` 对 `19754/19075/18534 us` | `6391/6425/6539` 对 `18549/18531/8249 us` |
| `(512,128)`（开 / 关） | `3484/3446/3645` 对 `6868/6683/7715` | `3431/3381/3585` 对 `6781/6749/4696` |
| `(256,4)`（开 / 关） | `1967/2012/2037` 对 `2720/2782/2782` | `1975/1965/1964` 对 `2824/2849/2789` |

- **`attention` 少 `0.55 s/token`（−41%）**，稳态 token 从 `3.76` 到 `3.21 s`（−15%）。
- **两档消失**：开着那六跑离散 ≤`2.7%`（第二批 `0.13%`），而且开着的值比"快档"还低
  （`0.79` 对 `0.95`）——池子连"快档"剩下的那点每次申请/首触也一起去掉了。
- 关着那一臂的整跑墙只在撞上慢档时高（第一批 `+2.62 s`，第二批第三跑回到快档所以只有 `-0.86 s`），
  说明整跑墙的差就是档位本身。
- 其余相位（`fixed`/`routed`/`shared`/`router`）两臂逐 token 一样。

**数值**：两批各 6 个跑，8 个 `hidden_bf16_digest`、8 条 43 层摘要链、8 个 argmax 全部一致，
比对只报 `peak device bytes`（代价不是数值）；1 步的四个钉死值两批都复现。
显存峰值 `1869152256 → 1823014912`（**`-46 MB`**）——修之前是既付了池子的内存
（`+153 MB`，`7e07056` 记录里那笔）又照旧每次申请。

**没声称的部分（别照着脑补）**：

- **同一段展开代码为什么会是两档，只定位到段、没定位到原因。** 测到的是：只有那两个循环变，
  `H2D`、`gemm`、`row_d2h`、`row_loop` 三跑全平；池子（复用那两块宿主缓冲）把它压到 `716/799 us`，
  **比快档还低一半**。**"每次申请到的页是新的、首触要缺页"是假设**——它和这三条观测都吻合，
  但没直接量过缺页计数。
- **服务路径一次没跑。** 服务面同样吃这个修复（它走同一条交付路径），但这次只用命令行量过。
- **没跑 `verify_delivery_arm.sh` 的换序那一遍。** 这次用的是交错 3 轮、两批，`attention` 是稳态
  后 4 token 的均值，冷帧不进比较；整跑墙只有同批内可比。
- **整请求墙不跨时段比。** `43.68` 对 `46.30 s` 只在第一批里成立；两批之间不可比。
- **routed 读盘那 `0.53 s` 没动**，它仍然是"每 token 必须从盘上搬 3.45 GB"的那部分。
- 没走 `rust/run-identified.sh`，结果文件里**没有源码指纹**。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- `48` 个 safetensors，合计 `166886535336` 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- 与 `2026-09-17-sm120-fixed-layer-slot-delivery.md` 同一份权重、同一台卡

## 运行证据

源码树 `/srv/localmoe/build/merged-prompt-chunk`（`CARGO_TARGET_DIR=/srv/localmoe/targets/merged-prompt-chunk-target`），
构建 `cargo oxide build --arch sm_120 -- --release`，`RUSTC_WRAPPER=rust/rustc-human-wrapper.sh`。

| 批 | 交付提交 | 二进制 SHA-256 | 结果目录（WSL 路径） |
| --- | --- | --- | --- |
| shared 探针 | `41da6e1` | `74dfc3d64f8ef2fe838fe8400661665726d505053461b17d309f92db91f67989` | `/srv/localmoe/targets/sharedprobe-shared-probe-a` |
| 分段探针（5 跑） | `bd313bc` | `3ab0a6fc1766806082d7e0ba6a3a247732117a31cf9e7e69d72a079a886e7f7d` | `/srv/localmoe/targets/compressorprobe-tier-a` |
| 分支判定 + 逐行分段（2 跑） | `b7da660` | 同上一行（同一棵树增量构建） | `/srv/localmoe/targets/compressorprobe-tier-b` |
| 池子 A/B 第一批 | `9efd662` | `af28cee73ac798d3452fd35e37bdc7aa45cf8743cf3dd81f6a95eaae1f42fa30` | `/srv/localmoe/targets/arenaab-arena-fix` |
| 池子 A/B 第二批 | `93d3154` | `af28cee73ac798d3452fd35e37bdc7aa45cf8743cf3dd81f6a95eaae1f42fa30`（**与第一批同一个二进制**） | `/srv/localmoe/targets/arenaab-arena-fix-b` |

两批之间只差 `docs/` 与 `tools/` 的提交，所以第二批是**同一个二进制重新量一遍**。

cuBLAS：`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`。

驱动脚本都在仓库里（`rust/colibri-cuda/tools/`），每条都是"同步 + 构建 + 跑 + 事后统计"一次跑完：

```bash
bash rust/colibri-cuda/tools/run_shared_probe.sh 41da6e1 shared-probe-a
bash rust/colibri-cuda/tools/run_compressor_probe.sh bd313bc tier-a 5
bash rust/colibri-cuda/tools/run_compressor_probe.sh b7da660 tier-b 2
bash rust/colibri-cuda/tools/run_arena_ab.sh 9efd662 arena-fix 3
bash rust/colibri-cuda/tools/run_arena_ab.sh 93d3154 arena-fix-b 3
```

（前三次跑的时候这三个脚本还在仓库外的 `.zcode/`，内容相同；第二批是从 `tools/` 里跑的，
路径已经过。`verify_baseline.py` / `outputs_match.py` / `attention_share.py` 仍在 `.zcode/`，
沿用 `verify_delivery_arm.sh` 的老约定。）

判据是 `.zcode/verify_baseline.py`（钉死值）与 `.zcode/outputs_match.py`（8 digest + 8 摘要链 +
8 argmax），两条都过了；没改它们。命令行配方用状态页第二节那套，**没有额外开关**，只多了两个探针：

```text
COLIBRI_V4_FLASH_BASE=1  COLIBRI_V4_FLASH_TOKEN_IDS=0  COLIBRI_V4_GREEDY_STEPS=8
COLIBRI_V4_STREAM_READ_MODE=buffered  COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas
COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas  COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice
COLIBRI_V4_BENCH_TELEMETRY=1  # 探针那次加 COLIBRI_V4_FLASH_SHARED_PROBE=1 / COMPRESSOR_PROBE=1
```

## 代码变更

- `b4aefcc` + `41da6e1`：`COLIBRI_V4_FLASH_SHARED_PROBE=1`。`layer_rows.rs` 里给 shared 窗口
  加内部计时：主线程自己返回的时刻、每个读盘 worker 自己的窗口、剩下的 join 等待。
  这就是把 `shared` 拆开的那一刀；默认关，只多 stderr。
- `bd313bc`：`COLIBRI_V4_FLASH_COMPRESSOR_PROBE=1`。把复用池的一次调用切成九段。
  九对 `Instant` 是无条件打点的（`~0.6 us` 对 `~10 ms` 的调用），只有打印受门控——
  这样开探针那一跑执行的代码和不打开时**完全一样**。
- `b7da660`：把复用池的分支输入打出来（严格模式、池子在不在、输入宽度、池子宽度、复用判据结果），
  并把逐行 cuBLAS 路径也切开；`weights_f32_h2d` 拆成 `expand_wkv/expand_wgate/h2d_wkv/h2d_wgate/workspace_alloc`。
  没有这一刀就不知道 `weights_f32_h2d` 里是哪一半。
- `9efd662`：**修交付代码**。`StrictCompressorArena` 的宽度从 `v4_state::HIDDEN` 改成由调用方给
  （Flash 面给 `spec.hidden`，full61 静止面给 `v4_state::HIDDEN`），复用判据改成问池子自己
  （`scratch.hidden()`）——宽度对不上就退回逐行路径，而不是报错。

改动只碰这三处所有权：`layer_ops.rs`（arena 结构与两个调用点的判据）、`generate.rs`（建池子时传宽度）、
`layer_stationary_prefill.rs`（同）。

## 未决事项

1. **为什么同一段展开循环会两档**（`2.9` 对 `13.3 ms`）。现在已经定位到段（那两个 `u16 -> f32` 循环）、
   定位到"和分配有关"（复用缓冲的池子把它压到 `716/799 us`），但没量到机制。
   **下一步**：把池子那两块宿主缓冲遮起来再量——在 `expand_wkv` 前后读 `/proc/self/stat` 的
   `majflt`/`minflt`，把"首触缺页"这条假设变成事实或否掉。
2. **同样形状的每次申请还在别处**：`run_compressor_official_cublas_host` 的 `input`/`kv`/`score`、
   `run_layer_shared_expert` 的三个 `DeviceBuffer::from_host`、`row_upload`（现在是 `0.3 ms`，看着不像）。
   1 做完之后再决定要不要一起处理。
3. **服务路径的相位账**。服务现在是主要交付面，`4.81 s/字` 内部还没拆；这个修复在服务面上
   应该同样生效（attention 少 `0.55 s/token`），但没量。
4. **`shared` 那一行的读法要改**：相位表把 `0.53 s/token` 的 routed 读盘记在 `shared` 名下。
   要不要在 `phase_profiles.rs` 里把 shared 窗口的"读盘部分"标出来（新增一个诊断字段），
   让加法表自己不再误导。
5. **routed 读盘 `0.53 s/token` 的余量**：worker 之间不平衡（快档均值 `11.4 ms`、最慢 `16.2 ms`），
   理想均分大约是 `0.49 s/token`。要不要动，取决于 4 做完之后那 `0.53` 还算不算数。
6. **`attention` 剩下 `0.79 s/token` 里还有什么**：`compressor` 已经掉到 `6.5 / 3.4 / 2.0 ms` 一次，
   剩下的是 `q_*_projection`、`kv_*`、`mhc`、`wo_*` 那些步。
