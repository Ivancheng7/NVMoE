> **这份是 2026-09-21 新写的（2026-09-21 晚按交付线 `codex/business-restart` 的 09-20 现状校正过一遍），
> 不在 2026-09-17 那批归档里。** 它记的不是一次实验，是"我们和上游 colibri 差在哪"的对比账；
> 状态仍然只看 `现状.md`，这份只在排下一步时读。它接在 `交接说明-固定层槽位与注意力拆分.md`（2026-09-17）之后。

# 交接说明：和 colibri 的差距（硬件之外）（2026-09-21）

## 一句话现状

我们的交付路径（`rust/colibri-cuda/`，跑 Flash-0731）和上游 `JustVugg/colibri`
同一个模型、同一条"显存+内存+盘当三层仓库"的路子。把硬件速度剥掉之后，差距主要是
三件事：**没有按专家身份跨 token 复用的缓存、没有专家前瞻预取、读盘只跟很小一块计算重叠**。
第一件我们根本没有；后两件是量过金额的既有开销。（初版并列的第四件"同一批字节搬了两遍 /
restage `0.44 s`"**已经在交付树上消掉了**，见 §三.4 的更正。）

**这份没有跑任何新东西**，引用的数字全部来自仓库里既有的记录和两侧的文档，不是本次量测。

## 一、colibri 和我们是什么关系

**仓库根的 `c/` 就是上游 colibri 引擎的一份快照**，不是我们自研的 C 参考实现：

- `c/colibri.c` 是 **GLM-5.2** 引擎（开头注释"Motore GLM-5.2 … in C puro"，意大利语），
  另有 `c/olmoe.c` 和共享头 `st.h`（safetensors）/`quant.h`（int4、int8 解码）/`tier.h`（三层布局）。
- git 历史是上游的：`8bd2fea Dual-SSD streaming: COLI_MODEL_MIRROR…`、
  `b5380f8 Merge pull request #506 from JustVugg/fix/401-context-truncation`。
  最后一次改动是 2026-07-22 `engine: keep the 107 KB GrDraft struct out of .data`。
- **`c/deepseek_v4.c` 从来没进过这棵树**（`git log --all -- c/deepseek_v4.c` 为空）。
  colibri 后来为 DeepSeek-V4 写的那个引擎，我们是**自己在 Rust 里另写的一套**。

所以：机制能在 `c/colibri.c` 里读它怎么写（GLM 版），但**没有现成的 V4 版可以搬**，
要动就得在 Rust 里重做。`AGENTS.md` 那句"C/CUDA 可作为迁移期间的参考实现"对得上，
但别把 `c/` 当成我们自己的代码去改。

## 二、它为什么快（它自己的文档写的，非我们的量测）

它 DeepSeek-V4 文档里给的账很直白：**decode 约 75% 的时间就是 routed 专家读盘**。
整套设计都围着"少读、早读、边读边算"：

| 机制 | 它报的数 | 来源 |
| --- | --- | --- |
| `PIPE=1` 异步读池（默认 8 worker） | 盘服务时间 −18% | colibri `docs/tuning.md` |
| `PILOT=1` 路由前瞻预取 | 一层前能猜中真实 top-8 的 **71.6%** | 同上 |
| `O_DIRECT` | Strix Halo 上 +65%（它注明跟盘有关，有的盘无效） | 同上 |
| 双 SSD 镜像（`COLI_MODEL_MIRROR`） | 带宽 ×2 | colibri `docs/deepseek-v4.md` |
| 热专家缓存 + pin（`.coli_usage` 学历史） | 命中就不读盘；"越用越快" | colibri `docs/tuning.md` |
| dense 常驻 RAM（int4） | ~6.3 GB（V4） | colibri `docs/deepseek-v4.md` |
| MLA 压缩 KV + 落盘 | 576 float/token（小 57×），重启热加载 | colibri `docs/tuning.md` |
| 一个专家三个矩阵盘上相邻、**一次 `pread`** | — | colibri `docs/deepseek-v4.md` |

它的速度上限就是**专家缓存命中率**：16GB 卡上 VRAM 命中率只有约 6%，decode 一半靠盘；
到 6×RTX 5090 把 19456 个专家全塞进显存+内存，才是 5.8–6.8 tok/s。

**别把它的头条当同机对比**：它 V4 的 `1.5–1.6 tok/s`（≈`0.64 s/token`）跑在
**RTX 5080 + 两块真 NVMe（聚合 ~6 GB/s）+ 32 GB RAM**；我们 `2.305 s/token`（README）
跑在 **16GB 单卡 + 一块 WSL 虚拟盘（实测 `3.66 GB/s`，见下）+ 15 GiB RAM**。

## 三、硬件之外的差距

按"值多少钱"排。前三条与盘快慢无关，是结构问题。

### 1. 没有按专家身份跨 token 复用的缓存（最大一项）

我们**每个 token 把 routed 专家 `3.45 GB` 从头读一遍**。证据是引擎自己的账：
`docs/experiments/2026-09-16-sm120-disk-ceiling.md` 里 `slotreuse-8step.jsonl`
344 条层记录（43 层 × 8 token）**每层恒定 `expert_loads = 6`、`source_bytes = 80216064`**，
没有任何一次因为"上一步刚读过同一个专家"而省掉。

- 全树搜 Rust 路径的 `cache_hit`/`expert_hit`/`reuse_expert`/`hit_rate`：**没有**。
  有两个名字像、但不是它的东西：一个是 `FixedLayerCache`（`device_hits`/`host_hits`），
  管的是**固定权重**，不是 routed 专家；另一个是 `COLIBRI_V4_QD4_IMMUTABLE_EXPERT_CACHE`，
  `env_flags.rs` 把它硬绑在 `layer_stationary_prefill` + `COLIBRI_V4_ROUTED_QD4` 上，
  是实验开关，不是 decode 路径的常驻缓存。
- 交付树上唯一的"槽位复用"是 `expert_slot.rs` 的 `RoutedSlotScratch`：它只在**单请求内跨层**
  复用同一块锁页暂存（省的是每层重新 `cuMemHostAlloc` 的约 21 ms），**不按专家身份跨 token**——
  命中与否跟"上一步读过谁"无关。
- colibri 的整个"越用越快"就建立在这上面（LRU 专家缓存 + 热专家 pin + `.coli_usage` 学历史）。
  > **2026-09-21 已补上一半**：`COLIBRI_V4_ROUTED_EXPERT_CACHE=<槽数>`（`10f0958`，默认关）按
  > `(layer, expert)` 存整条锁页 lane，命中零拷贝交给 H2D，逐位不变。真 prompt 每层载入
  > `6.0 → 1.0` 次、`3.21 → 0.86 GiB/token`。**但它在 fan-out 之上只再省 `0.07–0.08 s/token`**
  > （收尾是最慢那条 lane，砍不掉那条必须读盘的那条）。**还差两半**：(a) 缓存只在**主机锁页内存**里，
  > 命中照样要 H2D（`12.75 MiB × 6 × 43 = 3.2 GB/token` 的上传）；(b) 只有纯 LRU，没有
  > `.coli_usage` 那样的历史学习。显存现在空着约 8 GB ≈ 600 个专家，够放一个多 token 的工作集。

### 2. 对 routed 专家没有路由前瞻预取

我们有"提前一层"，但对象是**固定权重**：`fixed_prefetch.rs`、
`b2_runtime.rs::flash_b1_prefetch_next_layer`（明确 `current_layer + 1`）。
对真正的大头——每 token `3.45 GB` 的 routed 专家——**没有**对应的前瞻。
colibri 的 `PILOT=1` 用路由可预测性提前把下一层专家读进来（它量到 71.6%）。

### 3. 读盘只跟很小一块计算重叠

`现状.md` 自己写着：`shared` 那个窗口套住了"shared MLP 在主线程跑"和"6 个 worker 并排读
routed 专家"两件事，**98–99% 的层是"等最慢那个读盘 worker"收尾**。
也就是说读盘只和 `0.16 s` 的 shared MLP 重叠，**专家 kernel 仍要等数据**。
colibri 的 `PIPE=1` 是让读盘和整块 matmul 重叠。

（读盘本身已经并行到 6 条了：`现状.md` 记"6 条并发，最慢那条 16.2 ms/层"；
`2026-09-20-sm120-read-concurrency-ceiling.md` 再证并发到 6 条到顶、再加没用。
所以差距不是并发数，是**重叠**和**缓存**。）

> **2026-09-21 已落地**（`COLIBRI_V4_FLASH_ROUTED_FANOUT=1`，`0d12b0f`，默认关）：六个路由槽位各一个
> 线程/流/槽位，读→上传→内核在 lane 内串、lane 之间并行。稳态单 token `2.240 → 1.805`、
> 真 prompt `1.913 → 1.581 s/token`；记录 `docs/experiments/2026-09-21-sm120-routed-fanout.md`。
> **重叠做完之后，收尾换成了"最慢那条 lane 的读→上传→内核"**（`lane_max 19.45~24.47 ms/层`），
> 而那一整段里**读仍占大头**（`16.48/9.12 ms`）——所以这条差距只是从"没重叠"变成了
> "读还在关键路径上"，**真正的下一步是 §三.2 的路由前瞻预取**（把读挪到上一层里去）。

### 4. 同一批字节搬了两遍（restage）——已在交付树上消掉（2026-09-21 更正）

初版把这条列为"能直接做"，**现在它已经落地了**：routed 读盘不再先读进临时宿主缓冲、再拷一次
进锁页暂存区，而是直接读进锁页 read pool（`exp/routed-pinned-replay`：`575d36a` 读锁页、
`dc22c8e` 合入；交付 `现状.md` 第一节的已完成条目里已把这条划掉）。当时那 `0.44 s/token`
的第二次拷贝没有了。

所以专家路径现在剩的是三块（09-20 稳态实测）：
- **读盘 `≈0.48 s`**（记在 `shared` 名下那一段）——**已贴冷盘上限**：`3449 MB / 0.791 s = 4.36 GB/s`
  对冷盘 `3.66 GB/s`，没有余量，别再排（09-20 查清，`现状.md` §四.3）；
- **专家 kernel `0.47 s`**——算力，258 次小启动（43 层 × 6 专家）；
- **H2D `0.25 s`**——贴 PCIe 链路（`14.0 GB/s` 对链路 `14.27`）。

colibri 那边是直接读进专家槽位（`expert_load_impl` 里 mmap 或对齐 `pread` 进 `s->slab`，跨层复用）。
我们现在也读进锁页槽位了，差的是上面 §三.1/§三.3 的"缓存"和"重叠"。

### 5. 每个专家 6 次 `pread`，不是 1 次

我们的布局是权重区和 scale 区分开：一个专家要做 3 次权重读 + 3 次 scale 读，
**跨两个区 = 每专家两次寻道**（`2026-09-16-sm120-disk-ceiling.md` 第 4 节已列）。
colibri 把专家的三个矩阵盘上相邻放，**一次 `pread` 读完一个专家**。

### 6. attention 里最大的两块还在主机侧

现在 attention `0.57 s`，其中 compressor `0.20 s` + indexer `0.08 s` 是**主机实现**
（`现状.md` §四.2 明说；那 16 个小步已随设备链搬走）。colibri 的 V4 把 compressor/indexer
投影和稀疏 attention 都放在 GPU（`COLI_CUDA_ATTN_BATCH=1`）。

### 7. 没有前缀复用 / KV 持久化

colibri 把压缩后的 MLA KV 落盘、重启热加载，还有 prefix checkpoint；
它文档里量到"同一 system prompt 的后续会话 `6–9 s`（首字）"。
我们在 Rust 路径里搜不到 prefix checkpoint / KV 落盘，**每个请求都要重新 prefill**。

### 8. 能力覆盖（不是速度，但决定"能不能用"）

- **输入上限 `128`**（`prompt + 生成 − 1 ≤ 128`），prompt 超 121 直接拒
  （`generate.rs` 报 `CONTEXT_WINDOW_EXCEEDED`，`v4_state.rs:29 KV_WINDOW = 128`）。
  colibri V4 默认 `4096`、示例到 `20000`。
- **单模型、greedy、单 KV 槽**。colibri 已跑通 9 个模型家族、多槽、
  OpenAI/Anthropic 双协议（工具调用我这边有 `v4_chat.rs` 的 DSML，但覆盖面窄）。

## 四、哪些能直接做、哪些是我们的护城河

**能用"不碰数值"的方式补的（两条）：**

1. ~~让读盘直接落进锁页缓冲，删掉那 `0.44 s` restage~~ —— **已落地**（见 §三.4）。
2. ~~让读盘和专家 kernel 重叠~~ —— **已落地**（见 §三.3 的 2026-09-21 注：fan-out，默认关）。
3. ~~加按专家身份的缓存~~ —— **已落地一半**（见 §三.1 的注：主机锁页 LRU，默认关）。
   **还剩的两半**：显存常驻那份、以及 `.coli_usage` 那样的历史学习。

**这三条做完了，排在后面的（按价值）：**

4. **路由前瞻预取**（§三.2，colibri `PILOT=1`，它量到猜中 71.6%）：现在最值钱，因为 fan-out 之后
   一层的墙就是"最慢那条 lane 的读→上传→内核"，把读挪到上一层就能把它从关键路径上摘掉。
   **前提是先量我们自己的路由可预测性**（同一批里比"猜下一层"和"实际下一层"的重合率）。
5. **每专家 6 次 `pread` 收敛成 1 次**（§三.5）：要动盘上布局或读计划；它也在 lane 的关键路径里。
6. **compressor/indexer 上设备**（§三.6）：`0.20 + 0.08 s/token`，是 attention 的一半，纯主机实现。
7. **前缀复用 / KV 落盘**（§三.7）：服务面"能不能用"的那一类；我们现在每个请求都重新 prefill
   （真 prompt 32 行首字 `55–68 s`）。
8. **`O_DIRECT`/`fadvise` 两个读模式端到端没量过**（§二 里 colibri 报 Strix Halo +65%）：
   引擎里有 `COLIBRI_V4_STREAM_READ_MODE={buffered,fadvise,direct}`，但我们所有验收批都用 `buffered`。

**要靠移植或架构调整的：** 每专家单次 `pread`（要动盘上布局或读计划）、
compressor/indexer 上设备、前缀复用/KV 落盘、输入上限与并发。

**做不了的（是护城河）：** colibri 有一部分速度来自改路由或改精度——
`--topp` 自适应专家 top-p（它自己标 lossy）、`CACHE_ROUTE` 缓存感知路由、int4 dense 常驻。
我们的"**逐位不变**"纪律 + 原生 fp4/fp8 精度**做不了这些**。
所以上面两条能补，更远的才是取舍。

## 五、容易踩的地方

- **别把 colibri 的数字当我们同机的对手数**：它那台是 5080 + 双真 NVMe + 32 GB；
  我们是 16GB 单卡 + WSL 虚拟盘 + 15 GiB。差距里约一半是环境
  （盘 `3.66` 对我们这边快窗口，它聚合 `~6`）。
- **盘速本身在两档之间晃 `25%`**（`disk-speed-two-tier`）：任何 A/B 之前先量一次盘，
  落在 `2.7` 档就作废本批。
- **`c/` 是上游快照，别当我们的代码改**；它里面**没有 V4 引擎**，V4 的机制没有现成实现可抄。
- **这两条"能直接做"的都不许动数值**：验收照旧是四个钉死值 + 8 步 8 个 digest 逐个相同
  （`现状.md` 第一节）。缓存只改驻留、重叠只改时机，都必须过这一关。

## 六、引用到的东西在哪

- 我们侧：`现状.md`（唯一状态页，09-20 版）、`docs/experiments/2026-09-16-sm120-disk-ceiling.md`
  （每 token 读量）、`docs/experiments/2026-09-20-sm120-read-concurrency-ceiling.md`（并发到顶）、
  `rust/colibri-cuda/src/{expert_slot,fixed_prefetch,b2_runtime,env_flags}.rs`。
  ~~`现状.md` 里的 `restage 0.44 s`~~ 已划掉（`dc22c8e`）。
- 上游侧：`c/colibri.c`（本仓库快照）、`colibri` 的 `docs/deepseek-v4.md` 与 `docs/tuning.md`
  （数值出自这两份，非本仓库）。
