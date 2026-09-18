# 读进锁页不比读进复用缓冲慢：省掉 restage 的量测关过了（2026-09-17 深夜）

## 结论

想在 B1/serve 预取路径上省掉 `restage_weights` 那次主机侧双搬运，前置问题是：
"盘对读到什么缓冲极其敏感——同样单线程 buffered，读进新申请缓冲比复用缓冲慢
1.3 s/token。锁页是第三种状态，没人量过。" 这一次把它量了：

**用引擎同款 `cuMemHostAlloc` 页锁定缓冲、按大小复用，6 线程 buffered 读真实路由
轨迹，和普通 `posix_memalign` 复用缓冲同轮交错对比：没有系统性差异。**

| 轮 | pinned 墙 | reuse 墙 | 差 |
| --- | --- | --- | --- |
| 1（冷） | 7.111 s | 7.743 s | pinned 快 0.63 s |
| 2 | 4.336 s | 4.655 s | pinned 快 0.32 s |
| 3 | 4.721 s | 4.357 s | pinned 慢 0.36 s |

三轮方向不一致、差值都在轮间噪声带（同一天两轮 reuse 自己差 0.3 s）内，判
**没有惩罚**。两种模式 digest 逐位相同（`00000000000000f7`）、`errors: 0`、
物理字节严丝合缝（27,594,326,016）。

这符合先验：fresh 缓冲慢的 1.3 s/token 是"往全新页里写时的缺页+清零"——锁页
缓冲在 `cuMemHostAlloc` 那一刻就 fault 好且跨层不释放，行为属于"复用"档，而
不是介于两者之间的第三种状态。

`cuMemHostAlloc` 的一次性成本也量到了：每 run 12 次分配合计 0.19–0.29 s
（≈16–24 ms/次，与 2026-09-16 量出的 17.4 ms/12.75 MiB 一致），摊到整段请求
可忽略；但单 token 短请求要一次性多付这个数，是已知代价不是未知风险。

**判据：go。** 把预取 worker 改成"读进锁页池"没有磁盘侧的拦路虎，剩下的是
纯代码成本（worker i 认领第 i 套锁页缓冲，静态分配零锁）。

没声称的部分：

- **引擎一行没改。** 这次只改了回放工具；省 restage 本身还没做，0.2–0.35 s/token
  的预期收益仍是推算不是实测。
- 回放是纯字节搬运，没有 FP4、没有 H2D；它证明的是"读进锁页不慢"，不是整个
  改造的端到端收益。
- 第 1 轮两种模式同时慢 60%（7.4 s vs 4.5 s），是宿主页缓存状态（WSL2 虚拟盘
  背后的 Windows 侧缓存）在轮间漂移，两种模式同受影响；本记录只使用**同轮
  交错对**做对比，绝对值不跨轮比。
- 6 线程这档的绝对带宽（3.9–6.4 GB/s）比 09-16 那批（3.59–3.79 GB/s）整体
  偏快，同样是宿主缓存状态差异；两组各自的内部对比都有效。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，48 分片、`166898773853` 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- 机器：Ryzen 5 7500F（6 核 12 线程）、15 GiB 内存、`/dev/sdd` ext4（WSL2 虚拟盘）、RTX 5060 Ti 16 GB
- 回放计划：**真实路由轨迹**（不是伪随机 id）——`/srv/localmoe/targets/slotreuse-8step.jsonl`
  的 `requests[0].tokens[].layers[].route_ids`，8 token × 43 层 × 每层 6 专家，
  12,384 次 pread、27.594 GB（每 token 3.449 GB），生成脚本一次性内联在运行
  命令里（读 `model.safetensors.index.json` 解析绝对偏移）

## 运行证据

- 工具：`docs/assets/io-ceiling/disk_ceiling.c` 新增 `pinned` 模式（分支
  `exp/routed-pinned-replay`，提交 `07bbbeb` + `bd265ce`），`cuMemHostAlloc`/
  `cuMemFreeHost` 走 dlopen，无 GPU 机器上其他模式不受影响。
  2026-09-18 起这个文件从 `.zcode/io-ceiling/` 移到 `docs/assets/io-ceiling/` 入库
  （`.zcode/` 是客户端工作区，探测脚本按惯例不入库）
- 编译：`gcc -O2 -Wall -Wextra -o disk_ceiling_pinned docs/assets/io-ceiling/disk_ceiling.c -ldl -lpthread`，0 警告
- 计划文件：`/srv/localmoe/targets/io-ceiling/plan-real-routed.tsv`（新建）
- 结果文件：`/srv/localmoe/targets/io-ceiling/pinned-ab-0917.jsonl`（6 行，pinned/reuse 交错 × 3 轮）
- 命令形态：每跑前 `sync; echo 3 > /proc/sys/vm/drop_caches`，然后
  `./disk_ceiling_pinned plan-real-routed.tsv <pinned|reuse> 0 6`
- 完整数字（wall_gbps / alloc_s / read_s / cores_busy）：
  - pinned：`3.88 / 0.1893 / 35.713 / 4.40`、`6.36 / 0.2875 / 22.300 / 3.50`、`5.84 / 0.2234 / 25.238 / 4.29`
  - reuse：`3.56 / 0.0033 / 42.299 / 4.81`、`5.93 / 0.0027 / 23.720 / 3.74`、`6.33 / 0.0165 / 22.703 / 3.59`
- `read_s` 是六线程 read 时间之和（诊断口径），`wall_s` 才是每线程墙最大值；判对比只看 wall
- digest：两种模式全部 `00000000000000f7`，`errors: 0`，`physical_bytes` 两种模式完全一致

## 代码变更

只有回放工具，引擎一行没动：

- `disk_ceiling.c` 加 `pinned` 模式：缓冲纪律与 `reuse` 档完全一致（每线程按
  大小缓存、跨层复用），唯一差别是缓存申请换成 `cuMemHostAlloc` 页锁定内存。
  worker 线程开头 `cuCtxSetCurrent` 认领 context（锁页绑定调用线程的当前
  context），失败也过 barrier 再退出，避免其他线程死等。
- WSL 克隆里只 checkout 过 `exp/routed-pinned-replay` 编译工具，跑完已切回
  `codex/business-restart`。

## 未决事项

1. **省 restage 本身还没做**：预取 worker 改成从 6 套锁页池直接
   `RoutedExpertRangePlan::read_into`、routed 窗口从池子直接 H2D + execute、
   `restage_weights` 在这条路上退役。预期收益 0.2–0.35 s/token（restage 拷贝
   0.12 + fresh Vec 零初始化 0.37 里 shared 窗口能兑现的部分），是推算，做完
   要用同批交错 A/B + 八步 digest 验。
2. 保住两个重叠别顺手弄丢：worker 读 ∥ 共享 MLP（shared 76→17 ms/层 全靠它）、
   fixed worker 的跨层流水。实现按"worker i 认领第 i 套缓冲"静态分配，别让
   下一层的读去排上一层的槽位。
3. 第 1 轮慢 60% 的宿主缓存漂移没有对策字段；对照实验永远用同轮交错对。
4. 锁页内存总量：6 套 × 12.75 MiB = 76.5 MiB（若让槽位自己的 w*_host 退役，
   净增约 64 MiB），对常驻默认开（8 GiB）的部署可忽略。
