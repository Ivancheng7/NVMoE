# 整模型固定权重常驻显存：稳态省 0.72 s/token（−23%）（2026-09-20）

## 结论

把 43 层的固定参数（每层约 156 MB，合计 **6.71 GB**）在第一次读到之后**钉在显存里**，后续每个
token 直接用，不再每 token 重新 H2D 那 6.71 GB。开关 `COLIBRI_V4_FIXED_RESIDENT_GIB=<GiB>`，
**默认 0 = 关，行为与现在逐字节相同**。

实测（同一二进制，只切这一个开关，8 步真实 token，每次都先 `drop_caches`）：

| | 不常驻（默认） | 常驻 | 变化 |
| --- | --- | --- | --- |
| 稳态每 token（第 1–7 个，层循环外含 final head） | `3070.6 ms` | `2353.9 ms` | **−716.7 ms / −23.3%** |
| 第一个 token | `15163 ms` | `18129 ms` | **+2966 ms（填缓存的钱）** |
| 整请求 e2e | `37.45 s` | `35.27 s` | −2.18 s / −5.8% |
| 峰值显存 | `1825 MB` | **`8175 MB`** | **+6.35 GB** |
| `fixed` H2D 字节（8 步合计） | `53.67 GB` | `6.71 GB` | **−87%** |
| 生成的 8 个 token | `55222,33832,42549×5,127384` | **完全相同** | — |

8 个 `hidden_bf16_digest`、8 条 `ordered_layer_digest_sha256` 链、8 个 `argmax`、`generated_ids`
**四项逐项相同**。常驻那次 stderr：`fixed residency: 43 layers resident (6708864200 bytes),
301 layer-visits served from VRAM`（344 = 8×43 层趟，减掉填缓存的 43 次，其余 301 次全命中）。

**这一项比 2026-09-16 那次同名实验（15.4%）大得多**，因为层循环本身在这之后快了很多
（attention 设备链、final head 并行、routed 读锁页），固定流那 6.71 GB 从"被重叠掉大半"
变成了稳态里一块明显的东西。现在稳态每 token 里 `fixed` 是 `0.0 ms`（常驻）对约 `0.72 s`（不常驻）。

**这次没有声称的部分**（别照着脑补）：

- **要 +6.35 GB 显存**（峰值 `1.83 → 8.18 GB`，16 GB 卡上安全）。这是所有收益的价格，**没有改
  代码默认**——要不要默认开是待拍板的事，见 `现状.md` 第五节。
- **第一个 token 慢约 3 s**：填缓存那一遍按老办法关掉了预取（预取的工作线程槽位每轮必须被消费，
  两者不能混）。长生成里这 3 s 被摊薄，单 token 请求会被它拖慢——与 2026-09-16 那次未决项一致，
  **仍然没做**"把预取保留到填缓存那一遍"。
- **`fixed` 常驻时占的是 `B2RawRuntime` 的源字节（6.71 GB）**，不是展开后的 8.55 GB：FP8 的 `wo_a`
  保持原样单独放，每次用的时候解码进共享 scratch（沿用 stationary 缓存的 raw-wo_a 协议）。
- **没测服务面**。同一个 b2 循环，逻辑上同样生效，但一次 serve 没跑。
- **跑序偏差**：这次是"不常驻先跑、常驻后跑"，先跑那臂替整批付了 token 0 的冷页缓存。稳态那
  `−716.7 ms` 不受影响——常驻臂稳态里 `fixed` 恰好是 `0.0 ms`，是结构性的，不是噪声；但单看
  token 0 的 `+2966 ms` 会带上一点"后跑那臂缓存更热"的便宜，往保守方向读。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 权重与既有记录同一份（同目录、同 revision），未重算 SHA-256。

## 运行证据

- 源码树：WSL `/srv/localmoe/build/deliver-merge-0918`，`codex/business-restart` @ `e75d804`。
- 构建：`CARGO_TARGET_DIR=/srv/localmoe/targets/deliver-merge-0918-target`、`--release`、
  `62 warnings`、exit 0、约 21 s。
- 结果文件（WSL 绝对路径）：
  - 不常驻 `/srv/localmoe/targets/residentab-stream.jsonl`（e2e `37.45 s`）
  - 常驻 `/srv/localmoe/targets/residentab-resident.jsonl`（`COLIBRI_V4_FIXED_RESIDENT_GIB=10`，e2e `35.27 s`）
- 共同环境：`FLASH_BASE=1`、`FLASH_TOKEN_IDS=0`、`GREEDY_STEPS=8`、`STREAM_READ_MODE=buffered`、
  `FLASH_PREFETCH=1`、`MHC_LINEAR_ARM=cublas`、`ROUTER_LINEAR_ARM=cublas`、`ROUTER_SCORE_ARM=libdevice`；
  `FP4_ACCUM_ARM` / `FINAL_HEAD_ARM` 不设。
- 复现脚本：`.zcode/resident_ab.sh`；判定脚本 `.zcode/resident_ab_sum.py`、`.zcode/resident_steady.py`。

## 代码变更

`rust/colibri-cuda/src/generate.rs` 的 b2 贪心循环：

- `COLIBRI_V4_FIXED_RESIDENT_GIB` 为 0 时不建常驻表、行为完全不变；
- 设了预算就按层捕获 `B2RawRuntime`（首次读之后），后续 token 直接取用，命中时把
  `fixed_source_bytes` / `fixed_h2d_bytes` 记 0（填缓存那遍已经记过），台账上直接看得出省了多少字节；
- **常驻开着时关掉固定层设备槽位与预取**：常驻项自己就是那个"槽位"，两个复用机制同时开会把同一份
  权重交给两个所有者；
- 每次服务完把这一层放回槽位（命中与刚捕获都放），否则第一次命中就会把槽位清空而字节仍被计数
  （这是 2026-09-16 那次抓到并修掉的 bug，`ba2b2a8`）。

这次实现沿用 `exp/fixed-residency` 分支的 `6f578cd` + `ba2b2a8` 两笔的思路，针对现在多了
`FixedLayerSlot` 的循环做了适配。

## 未决事项

1. **要不要默认开**：+6.35 GB 换 −0.72 s/token（−23%）。是否把它设成默认，是 `现状.md`
   第五节要拍的板；这条实现是默认关的。
2. **消掉第一个 token 的 +3 s**：把预取保留到填缓存那一遍（2026-09-16 就列过，没做）。
3. **服务面没测**：同一个循环，应该同样生效，但得跑一次 serve 确认首字与稳态。
4. **预算没按空闲显存预检**：`COLIBRI_V4_FIXED_RESIDENT_GIB=10` 在 16 GB 卡上峰值 8.18 GB 安全；
   小显存卡要自己调小，没有 `preflight_cache_budget` 那种保护。
