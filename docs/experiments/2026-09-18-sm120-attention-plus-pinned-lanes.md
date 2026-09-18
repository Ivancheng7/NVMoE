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
