# 把每层固定参数常驻显存：A/B 实测（2026-09-16）

## 结论

把 43 层的固定参数（每层 156 MB，合计 **6.71 GB**）钉在显存里，**能省掉一半的 SSD 流量，
但只换来 15% 的稳定态加速**：

| | 不常驻 | 常驻 | 变化 |
| --- | --- | --- | --- |
| 整个 8-token 请求 | `106.71 s` | `102.60 s` | −3.9% |
| 稳定态每个 token（第 2–8 个） | 均值 `12.51 s` | 均值 `10.58 s` | **−15.4%（1.18×）** |
| 第一个 token | `15.18 s` | `24.59 s` | **+62%（填缓存的钱）** |
| 峰值显存 | `1.47 GB` | `8.03 GB` | +6.56 GB |
| SSD 读入合计 | `89.74 GB` | `42.78 GB` | **−52%** |
| H2D 合计 | `81.26 GB` | `34.30 GB` | −58% |
| 生成的 token | `[55222, 33832, 42549, 42549, 42549, 42549, 42549, 127384]` | **完全相同** | — |

**这条结果推翻了我们上一份记录里的推断。** 那份记录说"12 秒花在权重流上、有效速率 0.83 GB/s"，
读起来像是"读盘就是瓶颈"。实测把它否掉了：**0.83 GB/s 是"读盘与计算重叠之后"的整体速率，
不是关键路径**。固定参数的读之所以没占满时间，是因为预取线程（`COLIBRI_V4_FLASH_PREFETCH=1`）
在后台把它和计算重叠掉了。所以砍掉 52% 的字节，只兑现了 15% 的时间。

剩下每个 token 的 10.58 秒，按引擎自己的相位账大致是：

| 项 | 每个 token | 来源 |
| --- | --- | --- |
| routed 专家：槽位准备 | ≈3.65 s | `routed.diagnostics.setup_ns` 84.8 ms × 43 层，**纯软件开销** |
| routed 专家：读 + kernel | ≈2.3 s | 3.45 GB 读（已预取）+ 10.6 ms/层的 kernel |
| shared expert | ≈3.24 s | `shared.wall_ns` 75.4 ms × 43 层 |
| attention | ≈1.33 s | `attention.execution_ns` 30.9 ms × 43 层 |
| final head | ≈0.4 s | 1.06 GB 从盘读 + CPU 上算 |

**没声称的部分**：这次只测了 8 token 的单条请求、`block32` + CPU final head 这一套臂；
没测长 prompt、没测 serve 路径上的效果（按同样的重叠逻辑，serve 那边每个 prompt token 的
固定读也会被省掉，但省下来的时间应该同样有限）；没有把预取留在填缓存那一遍（见未决事项 1）。

## 固定输入与模型

- 模型目录 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 与 `2026-09-16-sm120-multitoken-and-serve.md` 同一份权重、同一张卡（RTX 5060 Ti 16 GB，`sm_120`）

## 运行证据

同一份二进制、同一套环境变量，只差一个 `COLIBRI_V4_FIXED_RESIDENT_GIB`：

- 二进制 `/srv/localmoe/targets/delivery-8step-target/release/colibri-cuda`
  （构建 `18.86 s`，exit 0，**59 条警告**，与记录的基线一致）
- 不常驻 `/srv/localmoe/targets/resident-off.jsonl`
- 常驻 `/srv/localmoe/targets/resident-on2.jsonl`（`COLIBRI_V4_FIXED_RESIDENT_GIB=10`）
- 共同环境：`COLIBRI_V4_FLASH_BASE=1`、`COLIBRI_V4_FLASH_TOKEN_IDS=0`、`COLIBRI_V4_GREEDY_STEPS=8`、
  `COLIBRI_V4_STREAM_READ_MODE=buffered`、`COLIBRI_V4_FLASH_PREFETCH=1`、
  `MHC_LINEAR_ARM=cublas`、`ROUTER_LINEAR_ARM=cublas`、`ROUTER_SCORE_ARM=libdevice`、
  `COLIBRI_CUBLAS_EXACT_LIB=…/libcublas.so.12.8.5.5`
- 常驻那次的 stderr 摘要：`fixed residency: 43 layers resident (6708864200 bytes), 301 layer-visits served from VRAM`
  （344 = 8 迭代 × 43 层，减掉填缓存的 43 次，剩余 301 次全部命中）

## 代码变更

`rust/colibri-cuda/src/generate.rs` 的 b2 贪心循环里加了一个自选的常驻表（提交 `6f578cd`、`ba2b2a8`）：

**代码不在交付分支上。** 这是一次实验，所以它放在 `exp/fixed-residency` 分支；`codex/business-restart`
上用两笔 `git revert`（`5329f4a`、`76ed97e`）把它去掉了，历史是追加的，没有改写。
要重新拿到这套改动：`git cherry-pick 6f578cd ba2b2a8`。

细节：

- `COLIBRI_V4_FIXED_RESIDENT_GIB` 为 0（默认）时行为完全不变；
- 设了预算就按层捕获 `B2RawRuntime`，后续 token 直接取用，`fixed_source_bytes` / `fixed_h2d_bytes`
  在命中时记 0（填缓存那一遍已经记过），所以台账上能直接看出省了多少字节；
- 捕获沿用了 stationary 缓存已经在用的 raw-wo_a 协议：FP8 的 wo_a 保持原样单独放，
  每次用的时候解码进共享 scratch，所以常驻的是源字节（6.71 GB）而不是展开后的（8.55 GB）；
- 预算不够就停止捕获、退回流式，不会失败。

**为什么没有直接接 `FixedLayerCache`**：那套机器（`fixed_layer_cache.rs`，1264 行）要求
`COLIBRI_V4_FIXED_CACHE_BASELINE_JSONL` 提供每层 `bench` 遥测，而且会用清单字节数和基线逐层核对，
是给"预算装不下全部、要挑最有价值的层"设计的。这里 6.71 GB 装得下全部，用不着挑，
所以先写了最小版本测效果；要不要收敛到那套机器（拿到它的租约协议和记账）是后面的事。

第一次实现有个 bug，被这次 A/B 抓出来：命中时把层取出来用，却没有放回槽位，
于是第二轮命中就把缓存清空了——`0 layers resident (8550341608 bytes), 55 layer-visits`，
整条请求反而比不常驻慢（144 s）。修法是让捕获/回填对命中与未命中都执行。

## 未决事项

1. **第一个 token 多花 9.4 秒**，因为填缓存那一遍关掉了预取（两者不能混：预取的工作线程槽位
   必须每轮被消费）。把预取保留到填缓存那一遍就能消掉，长生成里本来会被摊薄，单 token 请求会被它拖慢。
2. **剩下的墙是 routed 专家的槽位准备（每 token ≈3.65 s 纯软件开销）**，比它自己读数据的
   （≈1.87 s）还大。这是下一步该看的，不是磁盘。
3. **`shared.wall_ns` 每层 75.4 ms** 在权重已经常驻之后仍然存在，说明它不是等数据。
   是计算还是同步，没查。
4. **没测 serve 路径**：同一份改动对长 prompt 的首 token 等待应该也有帮助（每个 prompt token 少
   6.71 GB 读），但按这次的结论，兑现的比例大概率同样远小于字节比例。
5. **预算是显式数字，没有按空闲显存预检**。`COLIBRI_V4_FIXED_RESIDENT_GIB=10` 在 16 GB 卡上
   峰值 8.03 GB 安全；在 12 GB 卡上得自己调小，没有 `preflight_cache_budget` 那种保护。
