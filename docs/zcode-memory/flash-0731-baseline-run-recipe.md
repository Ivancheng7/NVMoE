---
name: flash-0731-baseline-run-recipe
description: 复现 Flash-0731 sm_120 业务基线的确切环境变量、命令与期望输出（token 55222 / digest 5ed8bf88beae3cfe）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_24caf8ff-f2e0-48f8-bebb-91b720903e59
---

2026-08-30 那次可复现基线运行是 `0731-sm120-clean-cuda128-default-block32-r2`，只有一个 token（`greedy_steps=1`），结果文件是单行 JSON。

期望输出（改动推理路径后必须复现这四个值）：

- `argmax` = `55222`，`generated_ids` 首元素 `55222`
- `hidden_digest` = `5ed8bf88beae3cfe`
- 43 层完成，`fp4_accum` = `"block32"`（不设 `COLIBRI_V4_FLASH_FP4_ACCUM_ARM` 走新默认）
- `final_head` = `"canonical_cpu_streamed_bf16"`（不设 `COLIBRI_V4_FLASH_FINAL_HEAD_ARM`）

运行时使用的臂：`router_linear.arm=cublas`、`router_score.arm=libdevice`、`mhc_linear.arm=cublas`，cuBLAS 解析到 `/usr/local/cuda-12.8/.../libcublas.so.12.8.5.5`（sha256 `18a5df54...`，version 120805）。

**2026-09-17：峰值显存那个数变了，别的三个没变。** `1466499072` → **`1500053504`**（+32 MiB）。原因是 `routed-slot-resident` 那一刀让槽位活到了采样点之前，**真实峰值没有变大，只是采样位置挪了**；`argmax` / `hidden_bf16_digest` / 43 层 / 两个臂名全部照旧。看到 `1500053504` 不要当成回归。同一个改动带来的一次性差异：`4a72109` + `598958e` 之后单步里"固定参数扫描次数"从按 token 数变成整个运行 1 趟（见 [[resident-on-is-the-comparison-baseline]]）。

复现命令（在 WSL 内，`<src>` 是源码目录，`<target>` 是 ext4 上的 CARGO_TARGET_DIR）：

```bash
env \
  COLIBRI_V4_FLASH_BASE=1 \
  COLIBRI_V4_FLASH_TOKEN_IDS=0 \
  COLIBRI_V4_GREEDY_STEPS=1 \
  COLIBRI_V4_STREAM_READ_MODE=buffered \
  COLIBRI_V4_FLASH_PREFETCH=1 \
  COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas \
  COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas \
  COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice \
  COLIBRI_DSV4_MODEL_DIR=/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  COLIBRI_CUBLAS_EXACT_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5 \
  COLIBRI_V4_RESULT_JSONL=<结果路径>.jsonl \
  <target>/release/colibri-cuda
```

注意 `COLIBRI_V4_FLASH_FP4_ACCUM_ARM` 和 `COLIBRI_V4_FLASH_FINAL_HEAD_ARM` 故意不设，用来顺带验证默认臂仍是 block32 / CPU final head。单步约 12–15 s。

**读结果时注意字段位置**：`argmax` 在 `requests[0].finals[0].argmax`，token 在 `requests[0].generated_ids[0]`，hidden digest 在 `requests[0].tokens[0].hidden_bf16_digest`——都不在顶层。顶层只有 `layers` / `fp4_accum` / `final_head` / `peak_device_used_bytes`。

## 2026-10-01：分支离交付线多远，会决定这个 digest 对不对得上

同一个单步命令，在 `exp/dspark-draft-probe`（比交付线 `codex/business-restart` **少 52 个提交**）上跑出来是：

- `argmax` = `55222`（一致）
- `hidden_bf16_digest` = **`3f6019eac9e2db33`**（不是 `5ed8bf88beae3cfe`）
- `peak_device_used_bytes` = `1825112064`

**这不是回归，是底座旧了。** 同一套命令在交付线 tip（`d1310f9`）上跑，四个钉死值全部复现：
`argmax=55222`、`hidden_bf16_digest=5ed8bf88beae3cfe`、`generated_ids=[55222]`、
`ordered_layer_digest_sha256=53de221f529dceafb166c788b766e22953df8105c3051d9dafb3911c57a31b42`，
且 **261 个 16 位十六进制 digest 字段逐一相同、零缺失**。

唯一差异是 `peak_device_used_bytes`：`1466499072` → `8242397184`。那是 `b620ef7` 常驻显存
默认开带来的（见 [[flash-0731-fixed-resident-vram]]），数值不受影响。

两条附带事实：
- **`hidden_bf16_digest` 不随 final-head 臂变**：在老底座上把 `COLIBRI_V4_FLASH_FINAL_HEAD_ARM`
  设成 `gpu_exact` 或不设（`canonical_cpu_streamed_bf16`），digest 都是 `3f6019eac9e2db33`。
- **比之前先确认分支与交付线的关系**：`git rev-list --count <branch>..codex/business-restart`。
  差得多就先 rebase 或直接在交付线上跑——否则会把"底座旧"误判成"改动引入回归"。

## 2026-10-01：重跑同一个输出文件名会静默假通过

引擎**拒绝覆盖已存在的结果文件**：`refusing to overwrite V4 trace artifact <path>: final path exists`，
进程 exit 101。这是证据保护，是对的。

坑在**比对脚本那一侧**：如果重跑用同一个 `COLIBRI_V4_RESULT_JSONL`，进程失败、新文件没产出，
而比对脚本照样去读那个**上一次的旧文件**，于是报出 `PASS：全部 digest 字段逐一相同`——
一个完全假的通过。我 2026-10-01 就这样被骗过一次。

**做法**：每次重跑换新文件名，并且比对前显式检查 `exit==0` **且** 产出文件非空、
是本轮生成的。别只看比对脚本的结论。

## 2026-09-16 建隔离验证台

不碰任何会话的工作区就能重跑这条基线，四件套：

- 源码 `/srv/localmoe/build/wrapper-verify`：从主仓库 clone 的 `codex/business-restart`。这是**可以安全 `git reset --hard`** 的地方，因为没有别人的在建工作。
- 热 target `/srv/localmoe/targets/wrapper-verify-target`：自带 744 个依赖产物，全量 release 构建约 25 秒（冷编译要几分钟）。
- 构建：`cd /srv/localmoe/build/wrapper-verify/rust && CARGO_TARGET_DIR=/srv/localmoe/targets/wrapper-verify-target ./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`
- 核对：`.zcode/verify_baseline.py <结果 jsonl>` 逐项打表

2026-09-16 实测（wrapper 接好之后）：构建 `24.68 s`、59 条警告、exit 0；单步推理 `19.4 s`；`argmax`/`generated_ids` = `55222`、`hidden_bf16_digest` = `5ed8bf88beae3cfe`、43 层、`block32`、`canonical_cpu_streamed_bf16`、峰值显存 `1466499072`（与 8-30 那次**完全一致**）。结果文件 `/srv/localmoe/targets/wrapper-verify.jsonl`。

对照组一并测了：同一条 `cargo oxide build --arch sm_120 -- --release -p colibri-cuda`，**不带** `RUSTC_WRAPPER` 时 exit 101、`mir_borrowck` ICE（当时报在 `final_head.rs:744`，说明 ICE 跟着代码走、与文件无关）；带上就 exit 0。见 [[cargo-check-ice-wrapper]]。

不能当基线的官方 TileLang oracle 漂移见 [[flash-0731-official-oracle-unstable]]；环境布局见 [[wsl-run-environment]]；编译门槛见 [[cargo-check-ice-wrapper]]。

## 定版不常驻基线（2026-09-16 晚钉下）

用户要求把最早那版"不常驻"固定成对照基线。`HEAD = 04bcf43` 本身就是它：全仓库没有任何一处读 `COLIBRI_V4_FIXED_RESIDENT_GIB`（那套代码在 `exp/fixed-residency`）。

- 构建 `CARGO_TARGET_DIR=/srv/localmoe/targets/localmoe-clone-sm120-target`：`19.84 s`、exit 0、**59 条警告**
- 新的基线产物（不覆盖任何旧文件）：`baseline-nonresident-1step.jsonl`（`16.02 s`）、`baseline-nonresident-8step.jsonl`（端到端 `103.38 s`）
- 旧两次不常驻 8 步是 `107.92 s` / `106.71 s`——**差异来自页缓存**：这次紧跟在 1 步后面跑，第 0 个 token `11.76 s`，冷的时候是 `14.88 / 15.07 s`。对比数字前先看页缓存状态。
- **别把两组显存缓存开关搞混**：被 revert 的实验开关是 `COLIBRI_V4_FIXED_RESIDENT_GIB`；仓库里另外有一组早就存在的 `COLIBRI_V4_DEVICE_FIXED_CACHE_GIB` / `COLIBRI_V4_HOST_FIXED_CACHE_GIB` / `COLIBRI_V4_STATIONARY_RAW_{DEVICE,HOST}_RETENTION_GIB`（`fixed_layer_cache.rs`），默认全 0，基线的"不常驻"就是它们都不设。

## 相位账遥测（只用来定位，不能当基线）

- `COLIBRI_V4_BENCH_TELEMETRY=1`（或 `COLIBRI_V4_PHASE_PROFILE=1`）放出 `phase_profile.*.bench` 子块（attention 的 `device_state_setup_ns` / `certified_host_path` 各步、router 的 `*_sync_ns` vs `*_gpu_ns`）。`phase_profile` 本身总是写出，只有 `bench` 受这个开关控制。
- 数值不受影响（1 步的四个钉死值照旧复现），但它会加同步点让计时慢约 1%（`104.24 s` vs `103.38 s`）——**只能用来定位，别当基线数字**。
- 读账脚本放在 `.zcode/`（未跟踪，不进库），从 WSL 侧用 `/mnt/c/...` 路径跑：`verify_baseline.py`（核对四个钉死值）、`phase_compare.py`（按 token 打顶层墙表）、`phase_layers.py`（每层明细）、`bench_dump.py`（打 `bench` 块）。

读法本身的规矩见 [[flash-0731-phase-profile-reading]]。

## 判"数值有没有变"的最强检查（2026-09-16 深夜）

`requests[0].tokens[i].ordered_layer_digest_sha256` 是**该 token 43 层逐层 digest 串成的链**——8 个 token 就是 344 层的中间结果，比只比最后一个 `hidden_bf16_digest` 强得多。2026-09-16 实测：交付不常驻基线、`exp/slot-reuse`、`exp/fixed-host-reuse`（**连缓存写坏的那版也包括**）四者的该字段 + 8 个 `hidden_bf16_digest` + 8 个 `argmax` + `peak_device_used_bytes` **完全一致**，即从交付路径到两条实验改动，模型输出一个 bit 没动。一次比完这些项的脚本是 `.zcode/outputs_match.py`（未跟踪，从 WSL 侧跑）。
