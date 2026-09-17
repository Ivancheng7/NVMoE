# attention 那 30% 落在哪，以及压缩器复用池进交付（2026-09-17 下午）

## 结论

两件事，第一件是**定位**，第二件是**落地**，但第二件的收益**今天量不出来**——原因和第一件是同一个。

### 一、`attention` 下午高 30%，落在 `compressor` + `indexer` 的主机侧展开上

逐子步对比（token 7，两次都是遥测开、同一份权重）：

| | 上午 11:00 帧 | 下午 15:05 帧 |
| --- | --- | --- |
| `compressor` 合计 / 每次 | `288.2 ms` / `7.03` | `629.1 ms` / **`15.34`** |
| `indexer` 合计 / 每次 | `131.2 ms` / `6.25` | `211.6 ms` / **`10.07`** |
| attention 墙 | `1.008 s` | `1.407 s` |

其余 16 个小步基本不动（`mhc 70.7 → 59.4`、`q_b_projection 59.7 → 55.0` 都在噪声里）。
`compressor` 41 层 + `indexer` 21 层算下来约 `+0.33 s/token`，**正好是 attention 涨的那 `0.38 s`**。
所以"下午的 attention 高 30%"这句话现在有具体地址了：**那两步是整条解码链上唯一"纯主机内存"的活**
（把 bf16 权重展开成 f32），其余步要么是 H2D，要么是设备侧 kernel。

### 二、但"下午"不是原因：同一份代码在这段活上有 `7` 与 `15.5 ms` 两档，跟开关无关

这是这次最要紧的一条。7 跑（都在 14:45 之后、同一份代码、同一份权重）里 `compressor` 每次调用：

| 时刻 | 跑 | 压缩器复用池 | `compressor` 每次 | attention 墙 |
| --- | --- | --- | --- | --- |
| 15:04 | `ab-comparena-probe` 8step-on | **开** | **6.97 ms**（快档） | `0.902 s` |
| 15:05 | 同上 8step-off | 关 | `15.34` | `1.407` |
| 15:08 | `ab-comparena-swap2-off` 8step-on | **开** | **`15.42`**（慢档） | `1.355` |
| ~15:1x | `drift-idle1`（单跑，空机器 10 分钟后） | 开 | `15.61` | `1.380` |

**开着池子照样落在慢档。** 我最初把 15:04 那一跑读成"池子把 attention 拉回 0.90"，
那是**混淆**：快档恰好落在开着那一臂上。两个顺序各跑一次才看出来——这正是"换序"这条规矩的价值。

### 三、压缩器复用池（`7e07056`）：数值判过了，收益今天判不了

- **数值逐位不变**：8 个 `hidden_bf16_digest`、8 条 43 层摘要链、8 个 argmax 与
  `defaults-pc-8step` 逐项一致（两臂都是，比对只报 `peak device bytes` 差异）。
- **代价：峰值显存 `1716060160 → 1869152256`（`+153 MB`）**，就是三个 arena 的宿主+设备缓冲。
- **收益今天没有可信数字**：池子要省的是"每次调用重新申请 2×29 MB 主机缓冲 + 设备缓冲"，
  而机器在这段活上的漂移（`7` ↔ `15.5 ms/次`）**比它本来的收益大一个量级**
  （探针受控 A/B 是热帧 `420.5 → 369.4 ms/token`，即 `-51 ms`）。
  两臂都落在慢档时，`15.34` 对 `15.42`——**池子没能把慢档拿掉**，所以慢档的代价不是"每次调用的分配"。
- 结论：这个改动留在交付分支（它用的是仓库自己的 arena，数值不动，路上少了一条已知的每次分配），
  但**它的收益要等两档的根因清楚之后才能量**。`COLIBRI_V4_FLASH_COMPRESSOR_ARENA=0` 回旧路。

### 已排除的解释（都量过，别再重复）

| 假设 | 量到什么 | 判定 |
| --- | --- | --- |
| 每次调用的分配 | 池子开着照样 15.42 ms/次 | 不是它 |
| 主机内存带宽退化 | 此刻 `9.1–9.4 GB/s`（spread-copy，256 MB 与 1024 MB 两档） | 不是它 |
| CPU 降频 | `/proc/cpuinfo` 全核 `3693 MHz`、load `0.71` | 不是它 |
| 别的会话抢机器 | 跑前 `ps` 查过，引擎 0 个 | 不是它 |
| 宿主内存压力 | `MemFree 15.2 GB`、swap 未用 | 不是它 |
| "机器闲一会儿就恢复" | 10 分钟空跑后再跑，仍是慢档 | 不是它 |
| 其他主机侧小步 | `mhc`/`q_b_projection`/`kv_projection` 两档几乎一样 | 只有 compressor/indexer 翻倍 |

### 没声称的部分

- **两档的根因没找到。** 只知道它只打"纯主机内存"那两步，且不随空闲时间恢复。
  下一个探针是现成的：`.zcode/compressor_probe_patch.py` 的六段分解（权重 D2H / 主机展开 /
  权重 H2D / 工作区 / 逐行 / SGEMM），**在两个状态下各跑一次**就能指出是哪一段翻倍。
- **上面两张表都不是新的绝对基线**：整请求墙同一天在 `3.35–4.4 s/token` 之间漂，别跨时段比。
- 服务面仍然一次没跑。

## 固定输入与模型

- 模型目录 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`（`48` 个分片、`166886535336` 字节）
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- 与 `2026-09-17-sm120-fixed-layer-slot-delivery.md` 同一份权重、同一台卡

## 运行证据

- 源码树 `/srv/localmoe/build/merged-prompt-chunk`，检查过 `e0d0e57`（`7e07056` 池子 + 脚本泛化），
  二进制 SHA-256 未单独记（批日志里有；`tools/` 那一笔不影响编译）
- 结果文件（WSL，不在仓库里）：
  - `/srv/localmoe/targets/ab-comparena-probe/{1step-comparena-on,8step-comparena-on,8step-comparena-off}.jsonl`
  - `/srv/localmoe/targets/ab-comparena-swap2-off/{1step-comparena-on,8step-comparena-off,8step-comparena-on}.jsonl`
  - `/srv/localmoe/targets/drift-idle1.jsonl`（单跑，池子开）
  - 上午那帧：`/srv/localmoe/targets/attn-split-pc-8step.jsonl`
- 一条命令跑完一批：`bash rust/colibri-cuda/tools/verify_delivery_arm.sh <commit> [tag]`
  （`ARM=<开关变量> LABEL=<标签>` 换臂；`ORDER=off-first` 换序；
  `.zcode/run_arm_ab.sh <commit> <arm> <label> <tag>` 跑换序那一半）
- 子步账：`.zcode/attention_split.py <jsonl> --token 7`；主机带宽：`.zcode/bandwidth_probe.py <MB>`

## 代码变更

- **`7e07056`**：解码路径把 `StrictCompressorScratch` 接通——`run_b2_layer` → `run_real_layer_rows`
  → `run_layer_compressor` 本来就有 `strict_scratch` 形参，只有解码侧传 `None`。
  两处压缩器调用点（主 + indexer）加了"单行才用池子"的回落：arena 按 `v4_state::HIDDEN` 定尺，
  分块 prefill 那一趟是多行，回落按行实现。
  门卫 `flash_b1_compressor_arena_enabled`（两个交付面默认开，`=0` 关）。
- **`e0d0e57`**：`tools/verify_fixed_slot.sh` → `tools/verify_delivery_arm.sh`，臂做成参数。

## 未决事项

1. **两档的根因**（最优先）：用 `.zcode/compressor_probe_patch.py` 的六段分解在两个状态下各跑一次。
   在那之前，**任何主机侧改动的 A/B 都要报出 compressor 每次调用的值**（7 ms = 快档、15.5 = 慢档），
   否则把档位差当成改动收益。
2. **`shared` 的 0.65 s** 仍没拆（老问题，会推翻相位加法表）。
3. **服务面相位账**：serve 仍不写 `phase_profile`。
4. 池子的收益要重判：等档位稳定（或能按档位归一）之后再跑一次两臂。
