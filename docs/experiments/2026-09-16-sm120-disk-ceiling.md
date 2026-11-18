# sm_120 裸盘天花板与 routed 专家读盘拆解（2026-09-16 晚）

## 结论

这次没有改引擎，也没有跑引擎，只量盘。结论：**盘不是 1 GB/s 的天花板。**

routed 专家每 token 要读的 `3.45 GB`，在同样的字节量、同样的块大小下：

- 单线程、冷缓存、每次都读进新申请的缓冲（引擎现在的读法）：`0.78–0.84 GB/s`
- 单线程、冷缓存、缓冲复用：`1.0–1.4 GB/s`
- **六个专家各一条并发流、缓冲复用：`3.79 GB/s`（`0.91 s/token`）**

三条互不依赖的量测都指向同一个数：自定义回放 `3.8 GB/s`、六个并发 `dd` 独立复核 `3.31 GB/s`、
单流顺序 `dd` 走 `O_DIRECT` 也有 `2.7 GB/s`。所以交接说明里"如果裸盘能跑 3 GB/s 以上，
这里有约 2 s/token（24%）"的判断成立——瓶颈是**并发深度**（现在是单线程一条流），
外加一项可单独量出来的开销：**每次读都写进新申请的零初始化缓冲**。

顺带更正交接说明里的一个数字：`3.1 s/token`（`70.9 ms/层`）是**冷态悲观端**。
交付跑 `slotreuse-8step.jsonl` 八步的实际平均是 `35.9 ms/层` 读 + `8.5 ms/层` 申请
= `1.9 s/token`，token 之间从 `21` 到 `48 ms/层` 都有。按这个平均，多线程能省的是
`0.9 s/token` 量级；按冷态算才是 `2 s/token`。两个数都对，场景不同。

没声称的部分：

- **没有在引擎里验证过多线程读**。下面所有 `3.79 GB/s` 都是独立回放工具的数字，
  引擎那边一行没改。这是下一步要做的事（未决事项 1）。
- 回放用的专家 id 是固定种子生成的伪随机六个/层，**不是真实路由结果**。形状与真实一致
  （每层六个随机 12 MiB 区），但具体数字不代表某一次真实运行。
  **2026-09-16 晚更正**：这份记录原先在这里写"结果文件里不记专家 id"，**是错的**——
  每层路由到的专家 id 就在结果文件的 `requests[].tokens[].layers[].route_ids` 里
  （例如 `postmerge-routed-threads-1step.jsonl` 第 0 层 `[254, 222, 245, 200, 53, 35]`），
  43 层 × 每 token 都有。所以回放可以换成真实轨迹，比这份的合成 id 更硬；
  本次没重跑是因为结论（并发 vs 单流、新缓冲 vs 复用）与取哪六个专家无关。
- `digest` 只用来证明数据真的读到了（所有运行 `errors: 0`；同一线程数下，
  换缓冲策略、换 buffered/`O_DIRECT`，digest 完全相同；多线程时因为累加顺序不同值不同）。
  不涉及任何数值正确性——这条路上没有 FP4，纯字节搬运。
- 这次没有产生新的引擎结果文件，所以**没有新的四个钉死值或 8 步 digest**。
  引用的引擎数字全部是从已有结果文件里读出来的，不是这次跑的。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- `48` 个 safetensors 分片，`166898773853` 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- 盘：`/dev/sdd`，ext4，挂载点 `/`（WSL2 里的虚拟盘，`lsblk` 里是 `Virtual Disk`，
  `rotational` 标志为 `1` 但实测速度是 SSD 量级，见下表）
- 机器：AMD Ryzen 5 7500F（6 核 12 线程）、15 GiB 内存、4 GiB swap
- 工具链：gcc 15.2.0、Python 3.14.4、coreutils `dd`

被引用、但**不是这次跑**的引擎结果文件：

- `/srv/localmoe/targets/slotreuse-8step.jsonl`（交付跑，43 层 × 8 token，`70.06 s`）
- `/srv/localmoe/targets/baseline-nonresident-8step.jsonl`（不常驻基线，`103.38 s`）

## 运行证据

全部在 `/srv/localmoe/targets/io-ceiling/`（WSL 路径，不在仓库里）。

### 1. 单流顺序读天花板（`dd`，1 GiB）

命令与原始输出在 `dd-ceiling.txt`：

```bash
sync; echo 3 > /proc/sys/vm/drop_caches; sleep 1
dd if=$M/model-00002-of-00048.safetensors of=/dev/null bs=4M count=256            # buffered 冷
dd if=$M/model-00002-of-00048.safetensors of=/dev/null bs=4M count=256            # buffered 热（紧接着）
sync; echo 3 > /proc/sys/vm/drop_caches; sleep 1
dd if=$M/model-00002-of-00048.safetensors of=/dev/null bs=4M count=256 iflag=direct
```

| 读法 | 轮 1 | 轮 2 |
| --- | --- | --- |
| buffered，冷缓存 | 1.6 GB/s | 1.5 GB/s |
| buffered，页缓存热 | 5.7 GB/s | 6.1 GB/s |
| `O_DIRECT`，冷缓存 | 2.7 GB/s | 2.7 GB/s |

### 2. 六路并发的独立复核（不经过自定义工具）

六个单线程 `dd` 同时读六个不同分片，各自 `bs=4M count=171 iflag=direct skip=8`：

| 并发数 | 字节 | 墙 | 吞吐 |
| --- | --- | --- | --- |
| 1 | 0.72 GB | 0.281 s | 2.56 GB/s |
| 6 | 4.30 GB | 1.301 s | 3.31 GB/s |

### 3. 按引擎真实 I/O 形状回放（`matrix.jsonl` / `matrix2.jsonl` / `matrix3.jsonl`）

每条运行前 `drop_caches`，干净冷态。计划为 43 层 × 6 专家 × 6 张量 = `1548` 次 `pread`，
`80216064` 字节/层、`3.449 GB/token`——与引擎的 `routed.source_bytes = 80216064` 完全一致。
每次读的耗时与"申请缓冲"的耗时分开记账（`read_s` / `alloc_s`）。

| 读法（并发） | 墙（s/token） | 吞吐 | 每层读 |
| --- | --- | --- | --- |
| 1 线程，新申请缓冲，buffered | 4.09 / 4.31 / 4.42 | 0.78–0.84 GB/s | 87–95 ms |
| 1 线程，缓冲复用，buffered | 3.43 / 2.49 / 2.51 | 1.01–1.39 GB/s | 58–80 ms |
| 1 线程，新申请缓冲，`O_DIRECT` | 2.86 / 2.76 / 2.81 | 1.21–1.25 GB/s | 64–67 ms |
| 1 线程，缓冲复用，`O_DIRECT` | 2.94 / 2.95 / 2.68 | 1.17–1.29 GB/s | 62–69 ms |
| 2 线程，新申请缓冲，buffered | 2.64 / 2.41 | 1.31–1.43 GB/s | |
| 3 线程，新申请缓冲，buffered | 1.68 / 1.63 | 2.06–2.11 GB/s | |
| 6 线程，新申请缓冲，buffered | 1.10 / 1.06 / 1.06 | 3.12–3.26 GB/s | |
| **6 线程，缓冲复用，buffered** | **0.96 / 0.91 / 0.91** | **3.59–3.79 GB/s** | |
| 6 线程，缓冲复用，`O_DIRECT` | 1.19 / 1.15 / 1.15 | 2.91–3.00 GB/s | |
| 12 线程，新申请缓冲，buffered | 1.07 / 1.08 | 3.19–3.22 GB/s | |
| 1 线程，完全顺序的同样字节量（对照） | 2.53 / 1.97 / 2.27 | 1.43–1.83 GB/s | 45–58 ms |

三件事可以直接读出来：

1. **并发是主要项**：单线程 `1.0–1.4 GB/s`，六线程 `3.6–3.8 GB/s`。
   六个专家本来就是六段互不相邻、各自连续 12 MiB 的区（下面第 4 小节），天然可以并行。
   到 6 线程就吃满了（12 线程没有更多收益），正好对应"一个专家一条流"。
2. **缓冲复用是第二项**：buffered 下单线程从 `87–95 ms/层` 降到 `58–80 ms/层`，
   差 `30 ms/层 ≈ 1.3 s/token`。钱不花在申请调用本身（`alloc_s` 全量只有 `0.009–0.012 s`，
   约 `6 µs`/次），而是花在 `pread` 往全新页里拷贝时的缺页/清零。
   `fresh_mmap`（每读一次 `mmap`+`munmap`）与 `fresh` 同速，证明这一点。
   有趣的是 `O_DIRECT` 下这项基本消失（`64–67` vs `62–69 ms/层`）。
3. **`O_DIRECT` 更好，但赢不了"buffered+复用+六线程"**：`2.9–3.0` 对 `3.59–3.79 GB/s`。
   而且 `O_DIRECT` 每次还要多申请一块对齐暂存、再 `memcpy` 一次
   （`physical_bytes` 比 `logical_bytes` 多 `0.2%`，是对齐扩展）。

### 4. 张量在分片里的布局（回放计划就是按这个生成的）

一层一个分片（`model-{layer+2:05d}-of-00048.safetensors`，43 层对 43 个分片一一对应，已逐个核对）。
片内：`256` 个专家的 `scale` 先排成一整块，然后是权重块；权重块里**每个专家 12 MiB 连续**
（w1、w2、w3 各 4 MiB 相邻），`w_stride = 12582912`。

所以引擎每层对每个专家做 6 次 `pread`：3 × 4 MiB（权重区，专家内连续）+ 3 × 256 KiB
（scale 区，专家内连续）。跨两个区 = 每专家两次寻道；六个专家 = 每层六段 12 MiB 随机区。

这与 `v4_expert.rs::load_with_prefix` 的读取顺序一致：
`w1.weight, w1.scale, w2.weight, w2.scale, w3.weight, w3.scale`。

### 5. 引擎侧的实际数字（从已有结果文件读出，非本次运行）

`slotreuse-8step.jsonl`，344 条层记录（43 层 × 8 token），每层 `expert_loads = 6`、
`source_bytes = 80216064`、`host_prefetched = true`（全部 344 条）：

| 项 | 每层平均 | 每层中位 | 8 步合计 |
| --- | --- | --- | --- |
| `routed.read` | 35.9 ms | 34.7 ms | 12.36 s |
| `routed.alloc` | 8.5 ms | 4.2 ms | 2.93 s |
| `shared`（= read + alloc + 1.7 尾巴） | 46.1 ms | 45.9 ms | 15.86 s |
| `routed.wall`（setup 24.8 + kernel 10.8 + h2d 5.7） | 47.9 ms | 47.2 ms | 16.49 s |
| `fixed` | 62.6 ms | 57.6 ms | 21.55 s |
| `attention` | 30.6 ms | 23.0 ms | 10.52 s |
| `outer` | 190.8 ms | 187.1 ms | 65.62 s |

读盘按 token 拆：`0.90 / 2.07 / 1.84 / 2.04 / 1.33 / 1.38 / 1.13 / 1.68 s`
（对应 `3.84 / 1.67 / 1.87 / 1.69 / 2.60 / 2.49 / 3.06 / 2.06 GB/s`），
`alloc` 只有 token 0 是 `1.47 s`，后七步每步 `0.20 s`。

两点确认：

- `shared` 减去 `read + alloc` 的残差是 `1.7 ms`，交接说明第 4 条更正
  （"`shared` 是盘的状态表，不是 shared expert"）**成立**，这次是独立复算。
- 读盘这段墙是**暴露在关键路径上**的：`shared` 窗口 = 预取线程的 `read + alloc` + join 尾巴，
  主线程那点 shared expert 计算藏在里面。所以读盘省下来的时间基本等于 token 时间。

### 复现方式

```bash
gcc -O2 -pthread disk_ceiling.c -o disk_ceiling
python3 gen_plan.py expert 1234 plan-expert.tsv          # 43 层 x 6 专家 x 6 张量
python3 gen_plan.py sequential 1234 plan-sequential.tsv  # 同字节量的顺序对照
bash run_matrix.sh ; bash run_matrix2.sh                 # 每次冷跑前 drop_caches
DIGEST_STRIDE=0 disk_ceiling plan-expert.tsv reuse 0 6   # 最好的一档：buffered+复用+6 线程
python3 layer_averages.py /srv/localmoe/targets/slotreuse-8step.jsonl
python3 token_breakdown.py /srv/localmoe/targets/slotreuse-8step.jsonl
```

工具源码：`docs/assets/io-ceiling/`。

## 代码变更

**引擎一行没改**，也没有重建二进制，所以交付分支的二进制仍然是干净基线。

新增的都是量测工具，按项目惯例没有入库，只在 `.zcode/io-ceiling/`（Windows）与
`/srv/localmoe/targets/io-ceiling/src/`（证据目录）各留一份：

- `disk_ceiling.c` —— 按计划文件回放 `pread`，把"读"和"申请缓冲"分开计时；
  支持 `reuse` / `fresh`（`posix_memalign`+`free`，等价于 Rust 的 `vec![0; n]`）/
  `fresh_mmap`（`mmap`+`munmap`），`buffered` 或"引擎式 `O_DIRECT`"
  （对齐扩读 + 拷出中间窗口，抄 `safetensors.rs::read_direct_aligned`）；线程按"专家"切分。
- `gen_plan.py` —— 从分片头部 JSON 取真实偏移，生成与 `load_with_prefix` 同顺序的计划。
- `run_matrix.sh` / `run_matrix2.sh` —— 矩阵与 `drop_caches`。
- `layer_averages.py` / `token_breakdown.py` —— 从结果文件里读分层/分 token 平均。

## 未决事项

1. **在引擎里把六个专家的读并行起来**（每专家一条 12 MiB 流）。这是这次量出来的最大一块：
   冷态 `3.05 s → 0.91 s`、交付平均 `1.9 s → 约 0.9 s`。数值上是无操作（读的是同样的字节，
   顺序不影响结果），验收照旧：四个钉死值 + 8 步 8 个 digest 逐个相同。
   注意这台机器只有 6 个物理核，读线程要和主线程的 shared expert 抢核，
   先量 3 线程和 6 线程两档，别默认 6 最好。
2. **缓冲复用**（去掉每个专家 6 次 `vec![0; N]`）。buffered 下单线程省 `30 ms/层 ≈ 1.3 s/token`，
   与第 1 项正交、可叠加。难点只有一个：复用缓冲要保证上一位使用者（`ExpertSlot::new`
   的解码与 H2D）已经读走，别覆盖还没消费的字节。
3. **把页缓存命中率量出来**。八步平均只有 `35.9 ms/层`（`2.23 GB/s`），
   而干净冷态回放是 `58–95 ms/层`——中间差的这一块应该是"同一个专家被再次路由"的命中，
   现在是推断，没有直接证据。要一个字段能一眼看出命中（交接说明最后一条就是这条规矩）。
   顺带：`final_heads.cpu_streamed_final_head` 每 token 读 `1059061760` 字节、
   `streamed_weight_read_ns` 只有 `41–149 ms`（合 `7–25 GB/s`），那个区域显然整块被缓存住了，
   可以当作"页缓存能到什么程度"的现成参照。
4. **更正交接说明里的 `3.1 s/token`**：8 步实际平均是 `1.55 s`（读）+ `0.35 s`（申请）
   ≈ `1.9 s/token`；`3.05 s/token` 只在冷态成立。下一步的排序（"先量盘"）不用改，
   但"这里有约 2 s/token"这句话要按冷/热分开说。
5. 与本次无关但同属"分配很贵"的那条：交付跑里 `routed.setup`（槽位重填）平均 `24.8 ms/层`
   = `1.07 s/token`，已超过读盘。`disk_ceiling` 证明这笔钱不在盘上，
   和之前"90% 是锁页不是拷贝"的结论一致，不要混到读盘这条线里解决。
