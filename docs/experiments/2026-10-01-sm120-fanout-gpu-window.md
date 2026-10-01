# fan-out lane 争用两笔账都到顶了：MLP 顶起不卡墙，GPU 窗口闸门不挣钱（2026-10-01）

## 结论

这条线一次关掉两件事，用的都是真跑的批，不是推断。

**一、"main_ns 那 0.26 s/token"换不到墙上的秒数，判据本身对准了不卡墙的相位。**
2026-09-30 晚交付线上（`codex/business-restart` @ `93423d8`，fan-out 默认开）的
`fanout-mlp-ab` 批重演了 fan-out 记录里的形状：稳态 `shared.main`（主线程 shared MLP 墙）
`9.9 ms/层`，关臂 `3.9 ms/层`，差 `6.0 ms/层`——乘 43 层正是 `0.26 s/token`，那个数是真的。
但相位窗口的代数是：**层墙 = attention + router + max(MLP 窗口, lane 链)**。同批里 lane 链
（`routed.lane_max`）是 `27.3 ms/层`，MLP `9.9` 完全藏在它下面——把 MLP 那笔争用全修好，
层墙一动不动。"看 `shared.diagnostics.main_ns` 掉不掉"作为判据会报成功，而 e2e 不动。

**二、能落在墙上的那份 lane 争用（lane 与 lane 抢 GPU 相位）也压不动：窗口闸门 K=3/K=2 无收益。**
给六条 lane 的"上传+内核"段加了一个计数闸门（`COLIBRI_V4_FLASH_FANOUT_GPU_WINDOW`，默认不设 =
完全不变），读盘照旧六路并发。三臂同批交错（单 token prompt、8 步、每跑前 drop_caches）：

| 臂 | 稳态 s/token（r2+r3 均值） | `lane_max` ms/层 | `shared.main` ms/层 | e2e s |
| --- | ---: | ---: | ---: | ---: |
| 不设（交付形状） | `1.815` | `24.3` | `10.3` | `30.4` |
| 窗口=3 | `1.810` | `24.3` | `10.7` | `29.4` |
| 窗口=2 | `1.925` | `26.3` | `9.2` | `31.2` |

K=3 与不设**逐项打平**（层墙、lane_max、main_ns、e2e 全在同带），K=2 还略亏（排队把省下的
争用又还回去了）。数值逐位不变：三个臂的 1 步四个钉死值全复现，8 步与本批 default 臂
**完全 IDENTICAL**（digest、43 层摘要链、argmax、连显存峰值 `8242397184` 都一样）。

所以：**候选 A（减少并发 lane 数 / 上传错峰 / lane 内核降优先级）这一族杠杆，对现在的墙
不挣钱**。层的临界路径是**最慢那条 lane 的 读→上传→内核**，其中读占约 `20 ms`（六路合计
`74–76 ms/层`）、GPU 尾巴只有 `4–6 ms`——闸门能匀的只有这条小尾巴。`main_ns` 那笔
`0.26 s/token` 只在 lane 链先被压到 `~10 ms/层` 以下才会变成墙上的秒数；到那一步之前，
压 lane 争用的正确杠杆是**把读挪出 lane 链**（路由前瞻预取）或者 **A2 缓存**（已落地待拍板），
不是动 GPU 相位。

**没声称的部分**：

- 流优先级（cudaStreamCreateWithPriority 那条）没试。机制上它作用于同一条已被本批证不绑定
  的 GPU 尾巴（把 SM 让给 MLP，lane 链只会更长或不变），没有单独成批。
- 服务面没跑（旋钮只动计时，数值面已逐位锁死；真要翻默认前补服务面，同 fan-out 翻默认时的做法）。
- 本批盘速闸门 `3.00–3.20 GB/s`，**中间档**（快档 3.5–3.75 之下、慢档 2.7–2.8 之上）。
  绝对秒数不跨批比，以上差值全部来自同批内交错。
- 每轮第一个跑的臂吃冷启动（r1 的 default `2.92`、k3 `2.29 s/token`），表里用的是同臂 r2+r3。

## 固定输入与模型

- 模型 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，48 个 `*.safetensors` 分片
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
  （与 2026-09-30 记录一致）
- 卡 RTX 5060 Ti 16 GB（`sm_120`）；cuBLAS `COLIBRI_CUBLAS_EXACT_LIB=…/libcublas.so.12.8.5.5`
- 常驻显存与 fan-out 都是代码默认开，三个臂只差窗口变量

## 运行证据

构建：树 `/srv/localmoe/build/rename-run-to-step`（detach 到 `4b0c6dc`，`exp/lane-contention`），
`CARGO_TARGET_DIR=/srv/localmoe/targets/rename-run-to-step-target`，
`./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`，**exit 0，18.98 s，
60 条警告**（touch 两个改动文件重编后 grep 零新增警告）。

前置批（判据前提的核对，交付线 tip、无窗口旋钮）：`/srv/localmoe/targets/fanout-mlp-ab-2026-10-01/`
的 `fan{on,off}-r{1,2}.jsonl`——fanon 稳态 `main 9.9 / lane_max 27.3 / outer 44.4 ms/层`，
fanoff `main 3.9 / join_tail 10.8–16.0 / outer 48.7 ms/层`。

本批 `/srv/localmoe/targets/fanout-window-2026-10-01/`（2026-10-01 22:47–22:55，12 次运行全
exit 0；盘速闸门三跑 `3.1954 / 3.0046 / 3.1322 GB/s`）：

- 1 步钉死值：`pin-{default,k3,k2}-1step.jsonl`，`verify_baseline.py` 三跑全 **baseline reproduces**
- 8 步三臂三轮：`{default,k3,k2}-r{1,2,3}.jsonl`，`outputs_match.py` 对本批 default 臂
  **k3/k2 全列 IDENTICAL**；对 `defaulton/default-8step.jsonl` 唯一差异是显存峰值（预期内）
- 逐 token 拆分（`token0 / 稳态 / e2e`）见上表；驱动脚本与汇总脚本在 Windows 侧
  `.zcode/run_fanout_window_ab.sh`、`.zcode/fanout_mlp_ab_sum.py`、`.zcode/lane_window_token0.sh`

臂环境（三臂相同，除标注外）：`FLASH_BASE=1`、`FLASH_TOKEN_IDS=0`、`GREEDY_STEPS=8`、
`STREAM_READ_MODE=buffered`、`FLASH_PREFETCH=1`、两个 `cublas` 臂 + `libdevice` 打分、
常驻与 fan-out 走默认。

## 代码变更

提交 `4b0c6dc`（`exp/lane-contention`，基座是交付 tip `93423d8`）：

- `env_flags.rs`：`fanout_gpu_window()`——读 `COLIBRI_V4_FLASH_FANOUT_GPU_WINDOW`，
  不设或 `0` = 关（不建闸门，交付形状），`1–6` = 六条 lane 里同时处于"上传+内核"段的条数，
  `>6` 报错。
- `layer_rows.rs`：`FanoutGpuWindow`（Mutex+Condvar 计数闸门）与它的 guard；fan-out 分支里
  每条 lane 的 worker 在**读盘之后、`upload_lane` 之前**拿闸门，闭包结束（`execute` 自带的
  同步事件之后）自动放行。读盘永远不设闸——六路并发对盘是唯一被证实有益的方向。
  worker 通过共享引用拿闸门（与 `shard`/`input` 同款），`move` 闭包不搬闸门本体。

默认路径一个分支都没加：不设变量时 `gpu_window` 是 `None`，闸门根本不构造。

## 未决事项

1. **`main_ns` 那 0.26 s/token 的正确兑现顺序**：先把 lane 链压到 `~10 ms/层` 以下
   （A2 缓存已落地默认关、待拍板；或路由前瞻预取），届时 MLP 从藏着的变成绑定的，
   争用修复才上墙。单独先动 MLP/上传/优先级是死路（本批证据）。
2. **A2 默认值拍板**（09-30 记录未决 3）现在多了一层含义：它压的正是 lane 链里那条
   `~20 ms` 的读，是"压 lane 争用"唯一还活着的杠杆。
3. 流优先级未单独成批（机制论证见上）；如果哪天 lane 链被读侧改动压短了，可以顺手补一刀。
4. 窗口旋钮留在树里（默认关、+69 行、零新警告、数值逐位锁死），作为这条线的定案探针；
   要删等 A2/前瞻预收落地后再清。
