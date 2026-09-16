# 固定参数的主机负载跨 token 复用：8 步 76.09 → 70.34 s（2026-09-16 深夜）

## 结论

每层解出来的固定参数主机负载现在**留在内存里跨 token 复用**，不再每个 token 从分片重建一遍。
开关是 `COLIBRI_V4_FIXED_HOST_RESIDENT_GIB=<GiB>`（**默认 0 = 关闭**，跟旧的常驻显存实验一个风格）。

同一个二进制、背靠背对跑两组（每组先关后开）：

| 组 | 关闭 | 开启 | 差 |
| --- | --- | --- | --- |
| 第一组 | `76.09 s` | **`70.34 s`** | `−5.75 s`（−7.6%） |
| 第二组 | `81.34 s` | **`72.11 s`** | `−9.23 s`（−11.3%） |

数值一字未变：8 个 token 的 `hidden_bf16_digest` 与 `slotreuse-8step.jsonl` 逐个相同，
`argmax=55222`、峰值显存 `1466499072` 字节。

**这条改动干了什么，有四个不受盘状态漂移影响的硬证据**（对照 → 开启）：

| 每层 | 对照 | 开启 | 说明 |
| --- | --- | --- | --- |
| `window.release` | `17.16 / 17.76 ms` | **`0.00 ms`** | 那块内存不再每层还回去 |
| 预取线程的墙 | `162.70 / 169.38 ms` | **`34.26 / 24.67 ms`** | 命中时它只是取个引用（均值被 token 0 的 43 次真加载拉着） |
| 进 SSD 的固定参数字节 | `53.67 GB` | **`6.71 GB`** | 只有 token 0 读了一遍，等于 8 个 token 省下 47 GB |
| 过 PCIe 的 H2D 字节 | `53.67 GB` | `53.67 GB` | **没省**，每层照样重传（这是它该干的） |

代价是常驻内存：峰值 RSS 从 `0.73 GB` 涨到 `6.96–7.25 GB`（43 层 × 131 MiB ≈ 5.6 GiB）。

**没声称的部分**：
- 效应大小只在**这两组**背靠背对照里成立；这台机器同一二进制的两次对照组本身就差 `5.25 s`
  （`76.09` vs `81.34`），所以别把 −7.6% / −11.3% 当成精确值，方向可信、幅度在同一量级。
- `fixed` 的墙降了 `22–35 ms/层`，但第一组里 `shared` 涨了 `19.8 ms/层`（第二组反而降 `1.7`），
  两边不一致，**我没有把它当因果**，也没法声称 fixed 省下的时间 1:1 变成 token 时间。
- **预算只测了 0 和 8 GiB**。8 GiB 装得下整个模型（5.6 GiB），所以是"全常驻"。
  预算小于模型时的前缀策略（代码写了、没跑）以及换层策略都没测。
- 预填充那条路**没接**，它仍然每个 prompt token 重建 43 层。
- 开启那轮里 `fix.h2d` 出现过 `923 ms` 的单层异常（均值被抬到 `32.62`，干净时是 `23.5`），没解释。
- **没有合并到交付分支**，改动在 `exp/fixed-host-reuse`。

## 固定输入与模型

与 `2026-09-16-sm120-nonresident-baseline.md`、`2026-09-16-sm120-slot-reuse.md` 同一份权重、同一张卡、同一套臂：

- 模型 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 48 个 safetensors 分片、总计 `166898773853` 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- RTX 5060 Ti 16 GB（`sm_120`）、cuBLAS `12.8.5.5`

## 运行证据

源码 WSL `/srv/localmoe/build/localmoe-clean-git`，分支 `exp/fixed-host-reuse`，提交 `29009b9`；
它的三个父提交是 `c3c1be8`（槽位复用）+ `b8cd920` / `190f1cf`（那两个不改行为的相位探针，
用来量 `fixed.wall` / `join` / `release`）。构建 `18.84 s`、exit 0、**59 条警告**（与基线相同）。

环境变量与槽位复用那次完全一致，只多一个开关；`GREEDY_STEPS=8`、`PREFETCH=1`、
`MHC_LINEAR_ARM=cublas`、`ROUTER_LINEAR_ARM=cublas`、`ROUTER_SCORE_ARM=libdevice`。

结果文件（都在 `/srv/localmoe/targets/`，都是新名字）：

- `fixed-host-reuse-off2-8step.jsonl`（`76.09 s`，关）/ `fixed-host-reuse-on3-8step.jsonl`（**`70.34 s`**，开）
- `fixed-host-reuse-off3-8step.jsonl`（`81.34 s`，关）/ `fixed-host-reuse-on4-8step.jsonl`（**`72.11 s`**，开）
- `fixed-host-reuse-on2-8step.jsonl`（`89.33 s`）——**别引用**：那是把缓存建在 token 循环里的那版，
  每个 token 都重建一遍、一次都命不中，是踩坑记录
- 峰值 RSS 用 `/usr/bin/time -v` 量，读 `Maximum resident set size`

每层相位（均值，毫秒）：

| 运行 | `fixed.wall` | ├ `h2d` | ├ `join` | ├ `release` | 预取线程墙 |
| --- | --- | --- | --- | --- | --- |
| off2 | 68.03 | 23.67 | 27.12 | 17.16 | 162.70 |
| on3 | **32.67** | 23.69 | 8.89 | **0.00** | **34.26** |
| off3 | 58.12 | 23.51 | 16.76 | 17.76 | 169.38 |
| on4 | **36.02** | 32.62 | 3.32 | **0.00** | **24.67** |

`release` 精确读到 `0.00`、`worker` 掉到命中价——这两项是这次改动是否生效的判据。
`join` 也跟着小了（`27.12 → 8.89`、`16.76 → 3.32`），因为预取线程不再需要那 137 MB 的时间。

## 代码变更

`exp/fixed-host-reuse`（`29009b9`），五个文件：

- 新增 `fixed_host_cache.rs`：`FixedHostCache`（按层存 `Arc<ResidentLayerHost>`，按预算决定留不留）
  + `resolve_host_layer`（命中就返回引用，未命中才从分片加载并按预算留下）
  + 这个解码循环自己的 `ResidentPrefetch` / `ResidentPrefetched`（带 `Arc`，不用共享给预填充路径的那两个结构）
- `b2_runtime.rs`：`load_b2_raw_runtime_from_host_for`（吃所有权）改成
  `load_b2_raw_runtime_from_host_ref_for`（借），调用方自己决定什么时候释放。
  raw `wo_a` 单独传，不再从 host 里 `take`。
- `generate.rs`：缓存建在 token 循环**外面**（一次运行一个）；两条加载路径都走缓存；
  释放的计时留在这个函数里（原来在 `b2_runtime.rs`，语义一样）。
- `b2_generate.rs`：`top_level.fixed` 多输出 `cache_status` 和 `host_resident`（命中与否的逐层证据）。
- `main.rs`：登记新模块。

**为什么把 raw `wo_a` 从 host 里拿出来单独放**：`LayerDeviceWeights::new_layer` 有一条守卫
（`device_weights.rs:121`）——host 里还带着 raw `wo_a` 就直接报错"raw wo_a requires the production
GPU decode upload path"。原来是在调用前 `take` 走的；改成借用之后必须保持同样的形状，
于是按仓库里已有的 `RawDecodedLayerHost { host, wo_a_raw }` 那样把两者并排放。

**踩过的第二个坑**：缓存一开始建在 token 循环里面，每个 token 重建一次，**一次都命不中**，
还白白留下又扔掉 5.6 GiB。是探针里的 `worker wall` 一眼看出来的（命中应该是微秒级，
实测 `195 ms`，比对照的 `162 ms` 还慢）。写在这里是因为"每 token 重建一次缓存"这种错，
只在有逐层计时的时候才看得见。

## 未决事项

1. **要不要合并进交付分支。** 现在 `exp/` 上有两条待决：`exp/slot-reuse`（数值零变化、8 个 digest 做证）
   和这条（数值零变化、8 个 digest 做证，但有 6.5 GB 内存代价和"效应幅度受漂移影响"的边界）。
   合并会让基线数字第三次作废，所以仍然是显式决定。
2. **RSS 代价值不值**：+6.5 GB 在这台 15 GB 的机器上还有余量，但再往上就没有了。
   仓库里 `fixed_layer_planning.rs` 已有"按预算选层常驻"的思路（`select_fixed_cache_layers`），
   可以拿来做"只驻留一部分层"，而且**预算小于模型的前缀策略还没跑过**。
3. **和常驻显存那条实验的关系**：`exp/fixed-residency`（把固定参数钉在显存里，峰值显存 8.03 GB）
   当时测到 12.51 → 10.58 s/token。这次是省内存侧的同一件事，两者能不能叠加、要不要二选一，
   没测过。
4. **预填充那条路没接**：它每个 prompt token 也重建 43 层，而且没有这条缓存。
5. 没解释的：开启那轮的 `h2d 923 ms` 单层异常；第一组里 `shared` 涨 `19.8 ms/层` 而第二组降 `1.7`。
6. 旧的还挂着：`shared` 每层 45–74 ms 没有内部计时；attention 31.2 ms 一层十来个同步点；
   `other` 那 58 ms 里的每层 index 解析（缓存一下就能省，与这条正交）。
