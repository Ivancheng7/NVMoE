# 三件已量改动合进交付分支：attention 设备链 + routed 读锁页 + final head 并行（2026-09-18）

## 结论

把三件**已在各自实验分支上量过、数值逐位不变**的改动合进交付分支 `codex/business-restart`，
在合并树上真构建、真跑 8 个 token 收口。三件事：

1. **数值逐位不变。** 合并树的 8 步产物对交付参照 `defaults-pc-8step.jsonl` 逐项比：8 个
   `hidden_bf16_digest`、8 条 `ordered_layer_digest_sha256` 链、8 个 `argmax`、`generated_ids`、
   `stop_reason`、每 token 43 层——除了**峰值显存**（下面单说），**全部 MATCH**。1 步的四个钉死值
   （`argmax=55222`、`5ed8bf88beae3cfe`、43 层、`block32` / `canonical_cpu_streamed_bf16`）原样复现。
2. **同批交错 A/B（同一二进制，只切两个开关）**：`attention` `18.6 → 13.5 ms/层`，
   `final head` 稳态 `440 → 102 ms/token`。这两项各自与它们在实验分支上单独量到的增益一致
   （attention 18.59→13.50 对记录里的 19.54→13.34；final head 438→103 对记录里的 468→110），
   说明三条改动**改的是不同的相位，叠加后不互相打架**。
3. **routed 读锁页那条在这棵树上没有开关**，所以没法同批切换；用 `routed.diagnostics.alloc_ns = 0.00`
   验证它确实生效（旧路径是每个张量 `vec![0; N]`，Slot 复用记录里约 `4.94 ms/层`）。它的增益要引
   自己的记录，不在这次的同批数里。

**这次没有声称的部分**（别照着脑补）：

- **服务路径只跑了命令行钉死值那套以外的冒烟**：三跑 `max-tokens 8` 的真实 prompt，token id
  三跑一致、与合并前逐字符相同，ITL 与命令行稳态对得上（见下"服务面"一节）。但**服务面不写
  `phase_profile`**，所以三个改动在服务帧里各占多少没拆；服务面的**峰值显存也没单独采**。
- **`final head` 的 e2e 收益被噪声盖住。** 稳态每 token 省 `~338 ms`，但整请求墙两臂只差
  `36.67/33.88`（全开）对 `37.84/37.86`（关掉）——`1–3 s` 的差落在跑间离散里。判据用
  `phase_profile.final_heads[].outer.wall_ns`，不用整请求墙。
- **峰值显存多了 `2 MiB`**（`1823014912 → 1825112064`）。`+2097152 字节`恰好等于 2 MiB，是
  attention 设备链那套 scratch；`final-head` 只多几个主机侧 `Vec`（不进这个数）。这是代价不是数值。
- **routed 读锁页的 `+4.3 ms/层` 是引用的**，本次只验到 `alloc_ns=0`，没在合并树上单独切臂量它
  （它没有开关，切不了）。
- **`read_ns` 这一列在合并树上偏高**（`59–68 ms/层`），比磁盘天花板记录里的 `35.9 ms/层` 高。
  两跑同开探针、同一批内一致，不影响 A/B，但**没查原因**，别当绝对数用。
- 没走 `run-identified.sh` 之外的东西；二进制 SHA-256 见下，结果文件里没有源码指纹（同惯例）。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- `48` 个 safetensors 分片，合计 `166886535336` 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- 机器：WSL2、RTX 5060 Ti 16 GB（`sm_120`）
- cuBLAS：`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`

## 运行证据

### 合并

在隔离 worktree（`Desktop\localmoe-deliver`）里操作，目标是交付分支：

```text
1) business-restart --ff-only--> exp/routed-pinned-replay   # 14ddffb，fast-forward
   （该分支已含 attention 设备链 6c66759，所以不用单独合 attention）
2) merge --no-ff exp/final-head-parallel                     # 合并提交 dc22c8e
```

`routed-pinned` 是 FF、`final-head` 无冲突（只碰 `v4_state.rs`）。合并前给交付 HEAD 打了备份 ref
`backup/business-restart-20260918`（指向 `0e732e4`）。

### 构建

源码树 WSL `/srv/localmoe/build/deliver-merge-0918`（`git clone` 自主仓库，checkout `codex/business-restart` @ `dc22c8e`）：

```bash
cd /srv/localmoe/build/deliver-merge-0918/rust
CARGO_TARGET_DIR=/srv/localmoe/targets/deliver-merge-0918-target \
  RUSTC_WRAPPER=.../rust/rustc-human-wrapper.sh \
  ./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda
# Finished `release` profile [optimized] in 52.21s，62 warnings，exit 0
```

二进制：`/srv/localmoe/targets/deliver-merge-0918-target/release/colibri-cuda`
SHA-256 `b924f8762420f34332e823a87f6026df48cf38122578ad78b7283ec48d80ae20`

### 数值

结果文件（WSL 路径，不在仓库里，只追加不覆盖）：

| 文件 | 说明 |
| --- | --- |
| `/srv/localmoe/targets/deliver-merge-0918/1step.jsonl` | 1 步，无遥测 |
| `/srv/localmoe/targets/deliver-merge-0918/8step-tele.jsonl` | 8 步 + `COLIBRI_V4_BENCH_TELEMETRY=1` |
| `/srv/localmoe/targets/deliver-merge-0918/ab-{all1,min1,all2,min2}.jsonl` | 同批交错四跑 |

1 步：`verify_baseline.py` 7 项全 `OK`，峰值 `1825112064`，`request_e2e_ns = 14175941409`。

8 步对参照 `defaults-pc-8step.jsonl`（`outputs_match.py`）：

```text
8step-tele.jsonl   DIFFERS: peak device bytes
```

逐项拆开（脚本比对）：`hidden digests` / `layer chain` / `argmax` / `ids` / `stop` / `layers`
**六项全部 MATCH**，只有 `peak_device_used_bytes` 不同（见结论）。

### 同批交错 A/B

同一二进制，四跑交错（全开 → 减开 → 全开 → 减开）。减开臂 = `COLIBRI_V4_ATTENTION_DEVICE_CHAIN=0`
+ `COLIBRI_V4_FINAL_HEAD_THREADS=1`；全开臂 = 两个都不设。其余环境与状态页第二节配方逐字相同
（`FP4_ACCUM_ARM` / `FINAL_HEAD_ARM` 故意不设）。

| 跑 | `attention` ms/层 | `routed` ms/层 | `final head` 稳态 ms/token | 峰值显存 | e2e |
| --- | --- | --- | --- | --- | --- |
| `all1` | **13.50** | 18.58 | **103.7** | `1825112064` | 36.67 s |
| `min1` | 18.59 | 18.52 | 438.7 | `1823014912` | 37.84 s |
| `all2` | **13.44** | 18.53 | **100.6** | `1825112064` | 33.88 s |
| `min2` | 18.87 | 18.55 | 441.9 | `1823014912` | 37.86 s |

- `attention` 省 `5.1–5.4 ms/层`；`final head` 稳态省 `337–338 ms/token`。两项在两臂各自稳定
  （全开离散 ≤0.5%，减开 ≤0.7%）。
- **四跑的数值全 MATCH**（对照 `all1` 的 hidden/argmax/ids）。
- `routed` 两臂一样——它没有开关，`alloc_ns` 四跑全 `0.00`（验证读锁页在跑）。

### 服务面（合并后补跑）

合并树上第一次跑服务面（`COLIBRI_V4_FLASH_SERVE=1`，客户端
`rust/colibri-cuda/tools/flash_serve_client.py`，真实 prompt `"你好，你是谁？"`、
`--max-tokens 8`）。三跑交错，每跑一个新引擎进程：

| 跑 | TTFT ms | ITL ms | 客户端 request | 进程 wall | 峰值 |
| --- | --- | --- | --- | --- | --- |
| `r1` | `20010.166` | `2915.367` | 41.1 s | 42.3 s | — |
| `r2` | `22907.994` | `3273.154` | 46.5 s | 47.8 s | — |
| `r3` | `20097.093` | `3357.649` | 44.5 s | 45.5 s | — |

- `ITL` 在引擎里的定义是 `(最后一个 boundary − 第一个) / (n−1)`——即**稳态每字间隔**，和命令行的
  稳态 `s/token` 直接可比。三跑 `2.92–3.36 s/字`，命令行同树稳态 `2.86–3.07 s/token`
  （`8step-tele` / `ab-all1` / `ab-all2`），在跑间离散之内。
- **三跑的 8 个 token id 完全一致**：`58572,42549,42549,109865,42549,42549,42549,42549`——
  与合并前 `prefetch-gate-serve` 那次逐字符相同，说明合并的三个改动没动服务面的数值。
- **`TTFT 20–23 s` 比命令行的 token 0（`11.6–15.0 s`）多约 `8–12 s`**：这是真实 prompt 的预填充
  （命令行用 `TOKEN_IDS=0`，单个 token 的 prompt）。这段没拆，见状态页"下一步"。
- 结果：`/srv/localmoe/targets/deliver-merge-0918-serve/serve-r{1,2,3}.log`（WSL 路径）。
- **服务面仍不写 `phase_profile`**，所以上面这些是它给出的全部；逐相位账还是只有命令行那一份。

## 代码变更

这次**没有新写推理代码**，只是把三笔已提交的改动合进交付分支：

- `6c66759` / `c9631c1`（attention 设备链，新模块 `attention_device_chain.rs`）
- `575d36a` → `95157fc`（routed 读锁页 `RoutedReadLane` / `upload_lane`）
- `42d94e0`（final head 按输出神经元切 6 线程，`v4_state.rs`）

合并本身：`business-restart` 先快进到 `14ddffb`，再 `--no-ff` 合并 `exp/final-head-parallel`
得到 `dc22c8e`。备份 ref `backup/business-restart-20260918`。

## 未决事项

1. **服务面逐相位账**：三跑冒烟过了（token id 一致、ITL 对得上），但 serve 不写 `phase_profile`，
   三个改动在服务帧里各占多少还是黑的。首字 `TTFT 20–23 s` 里那段预填充（比命令行 token 0
   多 `8–12 s`）也没拆。
2. **`routed.diagnostics.read_ns` 为什么偏高**（`59–68` 对磁盘天花板记录的 `35.9 ms/层`）。同批内
   一致、不影响 A/B，但要查清楚再引用绝对值。
3. **峰值显存那 `+2 MiB`** 是设备链 scratch，确认无泄漏（请求结束是否释放）没做。
4. `final head` 线程数默认 6 是拍的，没扫 4/8/12（引 `exp/final-head-parallel` 的未决事项）。
5. 状态页的"下一步"清单据此更新：attention 16 小步、routed restage、final head 并行三条已划掉。
