# 两个相位改动叠加验证：attention + routed 读锁页（2026-09-18）

## 结论

`exp/attention-device-chain`（attention 18 步搬设备侧）和本分支的
`exp/routed-pinned-replay`（routed 读盘直接进锁页）**改的是同一层里两个互不重叠的相位**，
合并后三条臂（改动前 / 只 routed / 两个都带）交错跑 6 轮：

**相位账干净地叠加**（每 token，43 层）：

| 相位 | 改动前 | 只 routed | 两个都带 | both−base |
| --- | --- | --- | --- | --- |
| `attention` | 16.99 | 16.97 | **13.33 ms/层** | −157 ms/token |
| `routed` | 22.63 | 18.45 | **18.44 ms/层** | −180 ms/token |
| `outer`/层 | 108.84 | 104.93 | **102.71 ms/层** | −264 ms/token |

两个改动的相位数**各自保持自己的值**：routed 从 22.63 降到 18.44（+0.01 相对只 routed），
attention 从 16.99 降到 13.33（−0.01 相对只 routed 那臂的 16.97）。**互相不干扰、不抵消**；
`outer` 省的 264 ms/token ≈ 157 + 180 再减去 shared/fixed 的几十毫秒互抵。

**八个 `hidden_bf16_digest` 在三条臂、六轮共 18 跑里逐位相同**（`5ed8bf88beae3cfe`
… `69ed859929ee9042`），合并后单跑也与交付参考全同（`ordered_layer_digest_sha256 53de221f…`、
`L0 route_ids [254,222,245,200,53,35]`、43 层、峰值显存 `3964207104` 不变）。

没声称的部分：

- **墙钟（e2e）仍被噪声主导，不能当判据。** 6 轮配对：只 routed 相对改动前 均值 −1.35 s
  （中位 −0.51，范围 −7.01…+2.20），两个都带 均值 −2.11 s（中位 −1.18，范围 −7.48…+0.86）；
  `base>routed>both` 严格单调的只有 4/6 轮。**方向对、幅度不可信**——这台机的墙钟
  同一天能飘 ±3 s 以上（t2/r1 的 baseline 一次跑到 43.64 s）。判据用相位账。
- **合并只做到"编译过 + 数值同 + 相位各自保持"**，两个改动的交互（如果存在）只能从
  相位数没互相污染来推断，没有单独构造交互实验。
- 新二进制是 62 条警告（合并前 60，attention 带进来 2 条它自己的历史警告），非新增回归。
- 本记录只回答"叠加成不成立"。两个改动各自的收益见各自记录：
  `2026-09-18-sm120-pinned-lane-read.md`（routed）、`2026-09-18-sm120-attention-device-chain.md`（attention）。

## "8 步总时长"里有一半是第一个 token 的一次性成本

这一条是本轮才看清的，之前所有把"8 步墙"当收益判据的记录都踩了这个坑：

**token 0 要建整个固定参数常驻库**（43 层的 host payload 解码 + 页锁定），
在 `fixed.read_alloc` 里一次性花掉 `327 ms/层 × 43 ≈ 14 s`；token 1–7 那一项是 0。
把 8 步总时长当分子，这 14 s 就占了 42%。

同一个二进制（`both-changes`）按 token 拆开：

| token | e2e | `fixed`/层 |
| --- | --- | --- |
| **0** | **15.09 s** | 278 ms（`read_alloc` 327） |
| 1–7 | 2.75–3.38 s | 16–26 ms（`read_alloc` 0） |

三臂交错 6 轮的 token0 / 稳态：

| 臂 | token0 | 稳态 | 稳态×8 |
| --- | --- | --- | --- |
| 改动前 | 13.87 ± 1.66 s | 3.386 ± 0.149 s/token | 27.09 s |
| 只 routed | 13.58 ± 0.62 s | 3.235 ± 0.143 | 25.88 s |
| **两个都带** | 14.23 ± 0.98 s | **3.033 ± 0.036** | 24.26 s |

**稳态才是干净的判据**：两个改动稳态省 `0.35 s/token`（3.386→3.033），而 8 步总时长
只看到 2.1 s——因为那 14 s 一次性成本把百分比压扁了。按 N 步外推（token0 一次性 + (N−1)×稳态）：

| N | 改动前 | 两个都带 | 省 |
| --- | --- | --- | --- |
| 8 | 37.6 s | 35.5 s | 2.1 s (6%) |
| 43 | 156.1 s | 141.6 s | 14.5 s (9%) |
| 100 | 349.1 s | 314.5 s | 34.7 s (10%) |

所以"8 步省 2 s"不等于"收益只有 2 s"：**步数越多，同一份稳态收益占比越高**。
以后报收益要同时给"稳态 s/token"和"N 步总时长"，只给后者会把一次性成本算进成绩里。

### 稳态相位格局（两个都带，token 1–7 均值 ×43）

| 相位 | ms/token | 占比 |
| --- | --- | --- |
| `routed` | 785 | 27.5% |
| `fixed` | 783 | 27.4% |
| `shared` | 630 | 22.1% |
| `attention` | 550 | 19.3% |
| `router` | 106 | 3.7% |
| 合计 | 2855 | |

**没有一块占多数**，所以不是"改完了"，而是要继续得同时动几块。已知的两条：
`fixed` 稳态每 token 还要 H2D **6.71 GB**（常驻省的是读盘，省不了上传；实测 27.5 GB/s，
已到 H2D 带宽量级，只能靠显存常驻或减字节）；`routed` 稳态读 3.45 GB、窗口 785 ms
（≈4.4 GB/s，已贴近这块盘的并发上限）。`shared`/`attention` 是计算，还没像 attention
那 18 步一样往设备侧搬干净。

## 固定输入与模型

- 模型：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 机器 WSL2、RTX 5060 Ti 16 GB（`sm_120`）、Ryzen 5 7500F、`/dev/sdd` ext4
- 三个二进制都 `./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`：
  - `localmoe-baseline-routed-target`：改动前 `653dccf`
  - `localmoe-pinned-target`：只 routed `95157fc`
  - `localmoe-both-changes-target`：合并后 `52433f4`
- 实验条件（三条臂完全相同）：`FLASH_BASE=1`、`GREEDY_STEPS=8`、`TOKEN_IDS=0`、
  `STREAM_READ_MODE=buffered`、`FLASH_PREFETCH=1`、`MHC/ROUTER_LINEAR_ARM=cublas`、
  `ROUTER_SCORE_ARM=libdevice`、`FP4_ACCUM_ARM=block32`、`FINAL_HEAD_ARM=gpu_exact`

## 运行证据

- 3 臂交错 A/B 共 6 轮（每轮三条臂、每次 `drop_caches`）：
  `/srv/localmoe/targets/io-ceiling/three-t1-{baseline,routed,both}-r{1,2,3}.jsonl`、
  `three-t2-…-r{1,2,3}.jsonl`
- 合并后单跑：`/srv/localmoe/targets/both-changes-8step.jsonl`
- 脚本（都在本分支 `.zcode/io-ceiling/`）：`run_three_arm_ab.sh`（交错、跑不进就 FAIL）、
  `sum_three_arm.py`（逐相位汇总）
- 合并提交：`52433f4`（`layer_rows.rs` 两边改的是不同区域，自动合并无冲突；
  编译通过才算数）

## 代码变更

合并 `exp/attention-device-chain`（`6c66759` + `b694858` + `c9631c1`）到本分支，
无手工解冲突。合并后 `layer_rows.rs` 同时有：预取读锁页（本分支）和
`attention_scratch` 参数 + `run_real_layer_attention_rows_device`（attention 分支）。

## 未决事项

1. **两个改动都还没合进交付分支**（`codex/business-restart`）。要合，按顺序：
   attention（独立、数值已验）→ routed 读锁页。
2. 墙钟的幅度要报的话，得换更稳的环境（独占机器 / 多次长时间跑取中位），
   现在这台机的 ±3 s 漂移盖过了 2 s 量级的收益。
3. 服务路径对两个改动都还没量（serve 不写 `phase_profile`）。
