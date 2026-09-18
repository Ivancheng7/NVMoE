# attention 的 18 个同步点搬到设备侧（2026-09-18）

## 结论

把交付解码路径（`run_b2_layer` → `run_real_layer_rows`）里 attention 那 18 步的**中间结果留在显存**，
只保留真正需要主机值的三处（compressor / indexer 的输入、commit、最终输出），18 步从"每步一次 H2D +
一次 D2H + 两次同步"变成一次串行入队。两条结论：

1. **数值零变化**：8 步真实 prompt（token 0）在两臂下 `generated_ids` 都是
   `[55222, 33832, 42549, 42549, 42549, 42549, 42549, 127384]`，`stop_reason = max_new_tokens`；
   8 个 token 的 `collapsed_bf16_digest` / `logits_f32_digest` / `normalized_bf16_digest` 两臂**逐位相同**。
2. **attention 相位从 `19.54 ms/层` 降到 `13.34 ms/层`**（344 层-趟求和 `6.72 s → 4.59 s`，带遥测）。
   省掉的就是那 16 个小步的主机往返开销。墙钟同向但噪声大（见"没说声称的部分"）。

这条不是"给设备 attention 解锁"——设备 attention（`run_device_attention_row`）只接在层静止/full61
那条面上，交付 CLI 走的是另一条。**这次是纯移植**：把主机路径里已有的算子换成它们的
`*_device_into` 版本，不写新 kernel。

**这次没有声称的部分**（别照着脑补）：

- **服务路径没量**：serve 不写 `phase_profile`，所以这条收益在服务帧里是多少不知道。
  服务与命令行已同批量过、一样快（memory `flash-0731-serve-path-speed`），但没有单独的 attention 拆分。
- **compressor / indexer 没动**：它们两个（`1.79 s` + `0.69 s`，带遥测）仍然是主机实现、仍然
  吃主机输入，所以设备链里还有一次 D2H 去喂它们。这两个是 attention 里现在最大的一块，但**不是**
  这次改动的目标。
- **墙钟数字噪声大**：三次配对（同二进制、只切 `COLIBRI_V4_ATTENTION_DEVICE_CHAIN`）
  省 `+6.28 / +3.49 / −1.57 s`（8 步）。中位数 `3.49 s / 8 步 = 0.44 s/token`，但有一轮反转，
  这台机的墙钟不适合当判据。**判据用相位账**。
- **带遥测会加同步点**：上面的 attention 相位是 `COLIBRI_V4_BENCH_TELEMETRY=1` 下量的，
  历史量级约 1% 偏慢；两臂同批、同遥测，差值可用，绝对值不要当基线。
- **`profile` 字段在设备链臂是 `None`**：逐步账（`certified_host_path`）只由主机路径写，
  设备链一开它就是 `null`——这正好是"主机路径不再执行"的直接证据，但也意味着**设备链上不再有
  18 步的内部拆分**，只有 attention / commit 两个总数。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- cuBLAS：`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`
- 源码树：`/srv/localmoe/build/localmoe-clean-git`，分支 `exp/attention-device-chain`
  （基于交付分支 `codex/business-restart` 的 `0e732e4`）
- 机器：WSL2、RTX 5060 Ti 16 GB（`sm_120`）

## 运行证据

两个臂共用同一份二进制（`/srv/localmoe/targets/attn-device-chain-target/release/colibri-cuda`），
只切一个环境变量：`COLIBRI_V4_ATTENTION_DEVICE_CHAIN=0` 走旧主机路径，不设（默认）走设备链。

命令（`COMMON` 是交付默认配置，与 memory `flash-0731-baseline-run-recipe` 一致）：

```bash
cd /srv/localmoe/build/localmoe-clean-git
COMMON="COLIBRI_DSV4_MODEL_DIR=/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  COLIBRI_CUBLAS_EXACT_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5 \
  COLIBRI_V4_FLASH_BASE=1 COLIBRI_V4_FLASH_TOKEN_IDS=0 COLIBRI_V4_GREEDY_STEPS=8 \
  COLIBRI_V4_STREAM_READ_MODE=buffered COLIBRI_V4_FLASH_PREFETCH=1 \
  COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas \
  COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice COLIBRI_V4_FLASH_FP4_ACCUM_ARM=block32 \
  COLIBRI_V4_FLASH_FINAL_HEAD_ARM=gpu_exact"
BIN=/srv/localmoe/targets/attn-device-chain-target/release/colibri-cuda
# 带遥测（相位账）：
env $COMMON COLIBRI_V4_BENCH_TELEMETRY=1 COLIBRI_V4_ATTENTION_DEVICE_CHAIN=0 \
  COLIBRI_V4_RESULT_JSONL=/srv/localmoe/targets/attn-chain-off-tele.jsonl $BIN
env $COMMON COLIBRI_V4_BENCH_TELEMETRY=1 \
  COLIBRI_V4_RESULT_JSONL=/srv/localmoe/targets/attn-chain-on-tele.jsonl $BIN
```

结果文件：

| 文件 | 臂 | 遥测 |
| --- | --- | --- |
| `/srv/localmoe/targets/attn-chain-off-8step.jsonl` | 主机路径 | 关 |
| `/srv/localmoe/targets/attn-chain-on-8step.jsonl` | 设备链 | 关 |
| `...-off-8step-r2/-r3`、`...-on-8step-r2/-r3` | 三次配对 | 关 |
| `/srv/localmoe/targets/attn-chain-off-tele.jsonl` | 主机路径 | 开 |
| `/srv/localmoe/targets/attn-chain-on-tele.jsonl` | 设备链 | 开 |

拷机确认：`layers = 43`，8 个 `finals[]` 每项 `status: PASS`。

**钉死值（两臂逐项相同）**：`generated_ids[0..8] = [55222, 33832, 42549, 42549, 42549, 42549,
42549, 127384]`、`stop_reason = max_new_tokens`；逐 token 摘要
`collapsed_bf16_digest = [7933f6f28bc01f11, 9f41308235d526dc, aec5c709904df23f, 4a6637bc7b224ac6,
0418527bedc01e0c, 4b1ac938a4cb541c, 1f7317cba2d1a429, 8220a035b03360a2]`；
`logits_f32_digest` / `normalized_bf16_digest` 同样逐条对齐。

**相位指标（带遥测）**：

| 指标 | 主机路径 | 设备链 |
| --- | --- | --- |
| `top_level["attention"].wall_ns` 求和（344 层-趟） | `6.720 s` | `4.588 s` |
| 每层 | `19.54 ms` | `13.34 ms` |
| `bench.attention.certified_host_path` | 18 步逐项齐全 | `null`（主机路径未执行） |

主机路径那 18 步的逐项求和（去杂项后的主要项，`344` 层-趟）：

```text
compressor           1.790 s   (5.20 ms/layer)   <- 保留在主机，未改动
indexer              0.692 s   (2.01 ms/layer)   <- 保留在主机，未改动
mhc                  0.586 s   (1.70 ms/layer)
wo_b                 0.454 s   (1.32 ms/layer)
q_b_projection       0.441 s   (1.28 ms/layer)
kv_projection        0.379 s   (1.10 ms/layer)
q_a_projection       0.378 s   (1.10 ms/layer)
kv_rope_quantize     0.350 s   (1.02 ms/layer)
wo_a                 0.296 s   (0.86 ms/layer)
commit_topk          0.200 s   (0.58 ms/layer)
sparse_score         0.194 s   (0.56 ms/layer)
hc_post              0.168 s   (0.49 ms/layer)
（其余 6 项，合计约 0.75 s）
TOTAL               6.672 s
```

16 个小步（去掉 compressor / indexer）= `4.19 s`。设备链把它们的花销基本吃掉：attention 相位降
`2.13 s`（344 层-趟）。

## 实现（供复核，不是结论）

- 新模块 `rust/colibri-cuda/src/attention_device_chain.rs`：`AttentionDeviceScratch`（一次性预分配
  全部中间缓冲，随请求循环存活）+ `run_real_layer_attention_rows_device`（同序、同参数的 18 步）。
- 接线点：`run_real_layer_rows` 新增 `attention_scratch: Option<&mut AttentionDeviceScratch>`
  末参；`run_b2_layer` 同样新增末参；`run_v4_b2_generate_arm` 里用 `attention_device_chain_enabled()`
  建一次 scratch。其余 15 个调用点传 `None`。
- 只能跑单行（`ATTENTION_DEVICE_CHAIN_MAX_ROWS = 1`）：prompt chunk 是位置内置的单行循环，
  算子从缓冲长度反推行数，多行会让校验错位。
- cache / topk staging 随 Csa4 上下文增长按需扩容，不预分配请求上限。
- 诊断面（`COLIBRI_V4_FLASH_PARITY_DUMP_DIR`、`COLIBRI_V4_WAVEFRONT_ATTENTION_BENCH`、
  `COLIBRI_V4_AUDIT_LAYER`）在设备链上会报错要求 `=0`：它们要读主机中间值，设备链里没有。
- 开关默认开、`=0` 走旧路径（与常驻 / prompt chunk 同一惯例）。

## 构建

用 `./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`。
**不要**直接用 `cargo build`：那样不会设 CUDA 后端的 `-Zcodegen-backend`，`#[cuda_module]` 的
artifact anchor 对象不生成，链接会报
`undefined symbol: cuda_oxide_artifact_anchor_246e25db_colibri_cuda_0_0_0`。
这条在已知能出二进制的树上也复现，与本次改动无关（2026-09-18 踩到并确认）。
