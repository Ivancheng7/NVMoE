# 不常驻基线定版，以及一个 token 的相位账怎么读才不骗自己（2026-09-16）

## 结论

把 `codex/business-restart` 的 `04bcf43`（不常驻那版）重新构建、重跑，钉成之后的对照基线。
四个钉死值全部复现：`argmax=55222`、`hidden_bf16_digest=5ed8bf88beae3cfe`、43 层、
峰值显存 `1466499072` 字节；8 个 token 端到端 **`103.38 s`**，稳定态每个 token **`12.5 s`**。

这一轮真正的收获不是这几个数字，是**相位账该怎么读**。发动机自己记的 `phase_profile` 里，
顶层那几段是互不重叠的墙（代码里有闭合等式
`top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns`，且执行时校验），
而 `diagnostics` 里的子项关系完全不一样：有的在墙里面，有的是后台线程记的、跟墙无关。
**上一份记录把两类混着用了，结论错了三处**，逐条列在下面。

稳定态一个 token（12.46 s）按顶层的墙拆开：

| 项 | 每层 | 每个 token | 占 |
| --- | --- | --- | --- |
| routed 专家 | `143.7 ms` | `6.18 s` | 50% |
| shared（见下面第 3 条更正） | `68.9 ms` | `2.96 s` | 24% |
| fixed 固定参数搬进显存 | `43.2 ms` | `1.86 s` | 15% |
| attention | `31.7 ms` | `1.36 s` | 11% |
| router | `2.3 ms` | `0.10 s` | 1% |
| 合计 | | `12.46 s` | |

装进盘里的字节是每个 token `11.2 GB`（固定 `6.71` + routed `3.45` + final head `1.06`），
合计 `0.9 GB/s`——**这不是瓶颈，下面第 2 条会说明为什么。**

### 三处更正

1. **旧记录里的"分项"和"总量"出自两次不同的运行。** 那张表的分项（setup `29.18 s`、
   shared `25.93 s`、attention `10.47 s`）来自 `0731-sm120-8step.jsonl`（端到端 `107.92 s`），
   而拿来对照的"实测 10.6 秒"是**常驻那次**的稳定态。分项自己那次是 `12.65 s`。
   数字没错，配错了对。
2. **"固定参数的读被预取线程和计算重叠掉了"不成立。** 预取藏住的是*读盘*那一段（后台
   每层花 `151.2 ms` 读 137 MB，约 0.9 GB/s），但固定参数每层仍有 `43.2 ms` 落在关键路径上
   （其中 H2D `23.2 ms`，即 137 MB / 6.9 GB/s，剩下约 20 ms 是缓冲搬运）。常驻显存拿掉的
   正好是这 `43.2 ms`×43 = `1.86 s/token`——**这就是那 15% 的全部来源，不多不少。**
   （对照：常驻那次稳定态 `10.30 s`，不常驻 `12.44–12.65 s`，差 `2.1–2.3 s`。）
3. **`shared.wall_ns` 不是 shared expert 的计算时间。** 那段计时的起点在**启动预取线程之前**，
   终点在 `worker.join()` 之后（`layer_rows.rs:943` 起、`:996` 止），所以它量的是
   "shared expert 计算 与 routed 专家读盘 重叠之后的墙"。常驻显存只钉固定参数、不钉 routed
   专家，所以这个数在常驻之后基本不动（最后一个 token：`69 ms/层` → `62 ms/层`）——
   **不能据此说"它不是在等数据"。**
4. 顺带一条：`routed.diagnostics.read_ns` / `alloc_ns` 是**预取线程**自己记的
   （`expert.read_ms` / `expert.alloc_ms`），不在关键路径上；把它们和 `setup_ns` 相加是重复计算。

### 最大的单项：routed 槽位准备 `84.3 ms/层`

`routed.diagnostics.setup_ns` 是 `ExpertSlot::new(...)` 这个调用的**主机墙上时间**
（`layer_rows.rs:1092–1096`），每层 6 个专家各一次，每次约 `14 ms`，乘 43 层是 **`3.62 s/token`，
占一个 token 的 29%**。它的形状是固定开销，不是数据：稳态那一个 token 的 43 层里
最小值 `79.7 ms`、最大值 `88.5 ms`，波动 9%。

读 `expert_slot.rs:44-169` 能看到这 14 ms 大概花在哪：每造一个槽位就新建
**8 块 pinned 主机缓冲 + 14 块设备缓冲 + 5 个 CUDA event**（`block32` 这条臂下不建 fp8 那 4 块），
并在函数体里**同步整条 stream 两次**（`:85` 的 `weight_h2d_end.synchronize()`、
`:129` 的 `complete.synchronize()`）。

**没声称的部分**：14 ms 里分配、event、同步各占多少**没有测**，上面只是读代码得到的位置；
那 22 块缓冲在槽位析构时是否真的释放也没有确认（`Drop`/`cudaFree` 都不在这份源码里，
应该在 cuda-oxide 那侧）。代码里留着一句当初的注释
（`expert_slot.rs:50`）："先一次一个专家，等实测到多专家流量再说"——现在实测有了，
但**还没有改**。这次记录只跑不改。

## 固定输入与模型

- 模型目录 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- safetensors **48 个**，目录总字节 **166898773853**
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- 卡 RTX 5060 Ti 16 GB（`sm_120`）；cuBLAS `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`，
  SHA-256 `18a5df540d156e9aff0363776fefab7884b614f122847b51c3eb1da772048e5c`，`cublasGetVersion` `120805`

## 运行证据

- 源码 `/srv/localmoe/build/localmoe-clean-git`，`HEAD = 04bcf43`，工作区干净
- 构建 `cd rust && CARGO_TARGET_DIR=/srv/localmoe/targets/localmoe-clone-sm120-target ./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`
  → `19.84 s`、exit 0、**59 条警告**
- 二进制 `/srv/localmoe/targets/localmoe-clone-sm120-target/release/colibri-cuda`
- 共同环境（"臂"属于实验条件，原样记录）：
  `COLIBRI_V4_FLASH_BASE=1`、`COLIBRI_V4_FLASH_TOKEN_IDS=0`、`COLIBRI_V4_STREAM_READ_MODE=buffered`、
  `COLIBRI_V4_FLASH_PREFETCH=1`、`MHC_LINEAR_ARM=cublas`、`ROUTER_LINEAR_ARM=cublas`、
  `ROUTER_SCORE_ARM=libdevice`、`COLIBRI_DSV4_MODEL_DIR=…`、`COLIBRI_CUBLAS_EXACT_LIB=…`。
  `COLIBRI_V4_FLASH_FP4_ACCUM_ARM` 与 `COLIBRI_V4_FLASH_FINAL_HEAD_ARM` **故意不设**，
  用来顺带验证默认臂仍是 `block32` / `canonical_cpu_streamed_bf16`（结果文件里两项都相符）。
- 结果文件（都在 `/srv/localmoe/targets/`）：

| 文件 | 步数 | 端到端 | 说明 |
| --- | --- | --- | --- |
| `baseline-nonresident-1step.jsonl` | 1 | `16.02 s` | 四个钉死值复现 |
| `baseline-nonresident-8step.jsonl` | 8 | **`103.38 s`** | 本节基线 |
| `bench-telemetry-1step.jsonl` | 1 | `16.08 s` | 开 `COLIBRI_V4_BENCH_TELEMETRY=1`，钉死值同样复现 |
| `bench-telemetry-8step.jsonl` | 8 | `104.24 s` | 同上，用来拆 attention 内部 |

- 每 token（`outer`＝43 层墙之和，单位秒）：

| token | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outer | 11.76 | 12.88 | 12.71 | 13.08 | 12.09 | 12.20 | 11.88 | 12.51 |
| routed | 6.14 | 6.15 | 6.19 | 6.22 | 6.21 | 6.16 | 6.17 | 6.18 |
| shared | 2.26 | 3.47 | 3.17 | 3.42 | 2.57 | 2.67 | 2.30 | 2.96 |
| fixed | 1.84 | 1.83 | 1.89 | 1.90 | 1.86 | 1.89 | 1.90 | 1.86 |
| attention | 1.36 | 1.28 | 1.31 | 1.39 | 1.32 | 1.34 | 1.36 | 1.36 |

  `routed` 和 `fixed` 稳到小数点第二位，`shared` 在 2.3–3.5 之间来回——它是上面第 3 条那个
  "重叠后的墙"，所以会随读盘快慢抖。
- `final_heads` 8 次合计 `3.271 s`（最后一次 `386 ms`，其中流式读权重 `45 ms`）；first head `515 ms`。
- 进盘/过 PCIe 的字节：`fixed_source_bytes=53.672 GB`、`fixed_h2d_bytes=53.671 GB`、
  `routed_source_bytes=routed_h2d_bytes=27.594 GB`、`final_head_source_bytes=8.473 GB`。
- **页缓存会影响数字，对比时要注意。** 这次 8 步紧跟在 1 步之后跑，页缓存是热的，第 0 个 token
  `11.76 s`；之前两次（`0731-sm120-8step.jsonl` `107.92 s`、`resident-off.jsonl` `106.71 s`）
  第 0 个 token 分别是 `14.88 / 15.07 s`，冷一些，整条请求差 3–4%。
- **开 `COLIBRI_V4_BENCH_TELEMETRY=1` 会让计时略变**（`104.24` vs `103.38`），因为它会加同步点。
  它只用来定位，不能当基线数字；数值结果不受影响（1 步那次的四个钉死值照旧）。
- attention 内部（`bench-telemetry-8step.jsonl`，最后一个 token 43 层的合计）：

| 子步骤 | 合计 | 每层 |
| --- | --- | --- |
| `compressor` | `0.689 s` | `16.0 ms` |
| `indexer` | `0.219 s` | `5.1 ms` |
| `mhc` | `0.060 s` | `1.4 ms` |
| `wo_b` / `q_b_projection` / `commit_topk` / `kv_projection` / `q_a_projection` | 各 `0.045–0.058 s` | 各约 `1 ms` |

  这些子步骤每一项都同时记 `sync_wall_ns` 和 `gpu_event_ns`，两者**几乎相等**
  （如 compressor `0.6892` vs `0.6863`），也就是主机每一步都停下来等 GPU 做完——
  一层十二来个小步骤，每个都同步一次。`device_state_setup_ns`、`device_state_reset_calls`、
  临时显存分配计数在这一路径上都是 0。

## 代码变更

**没有改动任何推理代码**，这次只跑和读。新增的是四个读结果文件的脚本，
放在 `.zcode/`（未跟踪，不进库）：`phase_extract.py`、`phase_compare.py`、`phase_layers.py`、
`bench_dump.py`。`phase_compare.py` 按 token 打印上面那张顶层墙表，`phase_layers.py` 打每层明细，
`bench_dump.py` 打被 `COLIBRI_V4_BENCH_TELEMETRY=1` 放出来的 `bench` 块。

关于"不常驻是默认"这件事本身：`HEAD` 上没有任何一处读 `COLIBRI_V4_FIXED_RESIDENT_GIB`，
那套代码在 `exp/fixed-residency` 分支上。仓库里另外有一组**早就存在**的显存缓存开关
（`COLIBRI_V4_DEVICE_FIXED_CACHE_GIB`、`COLIBRI_V4_HOST_FIXED_CACHE_GIB`、
`COLIBRI_V4_STATIONARY_RAW_DEVICE_RETENTION_GIB` 等，见 `fixed_layer_cache.rs`），
它们默认全 0，`resident_on2` 那次也没有设——**别把它们和实验开关搞混**。

## 未决事项

1. **`ExpertSlot::new` 每次重建整套工作区（每 token 258 次，`3.62 s`）。** 先加一次测量，
   把 14 ms 拆成"分配 / 建 event / 两次 stream 同步 / 其余"，再决定改成每层复用还是池化。
   这个数是本地测出来的，动它之前要先把 `bench` 那套遥测接上 `shared` 和 `routed` 的槽位路径。
2. **`shared` 那段 `68.9 ms/层` 没有内部计时。** `LayerStageTelemetry` 里 shared 只有 `shared_ns`
   一个字段，既分不出"shared 自己的计算"也分不出"等 worker 的 join"。要能答"常驻之后
   还剩 64 ms 是什么"，必须先加子项。
3. **attention 的 `compressor`（`16.0 ms/层`）与 `indexer`（`5.1 ms/层`）都是同步步骤**，
   占 attention `31.7 ms` 里的 21 ms。没查为什么每个小步骤都要同步回主机，
   也没查 compressor 那 16 ms 的 GPU 时间是否真的在算。
4. **`fixed` 每层那 `43.2 ms` 的里子没拆开**：`h2d_ns` 占 23.2 ms 是清楚的，剩下约 20 ms
   到底是等预取槽位还是搬缓冲，没测。这关系到"常驻之外还有没有办法削掉它"。
5. **serve 路径和长 prompt 这次没碰。** 本记录只覆盖 `GREEDY_STEPS` 1 和 8 的 CLI 路径；
   "用户能用真实输入从服务拿到多个 token"这条线仍然没测（prompt 每个 token 约 17–19 s，
   几百字的对话要等一小时以上）。
6. **`exp/fixed-residency` 上的常驻改动不要合并回交付分支**（15% 换 6.56 GB 峰值显存，
   挤的是路由专家和 KV 的位置）。这次重新验证的结论与 A/B 那次一致，不再复议。
