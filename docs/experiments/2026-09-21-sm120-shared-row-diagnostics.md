# sm_120 shared 行自带读盘/MLP 分解（2026-09-21）

## 结论

给相位账里的 `shared` 行补上了它自己的分解字段，改的是**记账**不是**计算**：交付 CLI 默认配置
跑 1 步，四个钉死值全部复现；跑 8 步，8 个 `hidden_bf16_digest`、8 条 43 层摘要链、8 个 argmax
与常驻默认开的参照 `defaulton/default-8step.jsonl` **逐位相同**。

补出来的数把旧表那句话坐实了：稳态 `shared` 墙 `14.66 ms/层` 里，**shared MLP 自己只有
`4.09 ms/层`**，其余 `10.57 ms/层` 是"等最慢那个读盘 worker"的 join 尾巴；六个 worker 的
读墙合计 `56.16 ms/层`，与 `routed.diagnostics.read_ns` 的 `56.11 ms/层` 相等——**那笔读只归
`routed` 一次，`shared` 行以前看着占四分之三，是因为它把别人的账套在自己框里**。

**没有声称的部分**：这是一次记账改动，不含任何加速；`@` 既没动读盘路径（读侧 09-20 已判到顶），
也没动数值。新增字段只在 `phase_profile` 里，绝不参与 `ordered_layer_digest_sha256`（那条链由
`B2LayerDigest` 的 hidden/route/attention/KV 摘要构成，与相位 JSON 无关）。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- safetensors：48 个分片，模型目录合计 `166898773853` 字节
- `config.json` sha256：`6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` sha256：`98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- 推理参考（未参与本次比对，仅登记）：`inference/model.py` sha256
  `c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f`、
  `inference/kernel.py` sha256 `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`

## 运行证据

构建（WSL 交付树 `/srv/localmoe/build/deliver-merge-0918`，`codex/business-restart` @ `0e515f7`）：

```bash
CARGO_TARGET_DIR=/srv/localmoe/targets/deliver-merge-0918-target \
  ./rust/run-identified.sh build --arch sm_120 -- --release -p colibri-cuda
# Finished `release` profile [optimized] target(s) in 20.44s
```

cuBLAS 实际加载：`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`
（`COLIBRI_CUBLAS_EXACT_LIB`）。

盘速门（跑前量的两档判据）：`dd iflag=direct bs=1M count=2048` 取 `model-00005`，
`2147483648 bytes copied, 1.21589 s, 1.8 GB/s`——**落在慢档**（历史快档 `3.66`、慢档 `2.7`；
这次只有 `1.8`）。因为本次是**逐位记账验收**、不看收益，慢档不构成作废理由；但**本批的
绝对时间数字（下面那张相位表）不可拿去跨时段比收益**，只看字段间的相对关系。

两次运行（`COLIBRI_V4_FLASH_BASE=1` + `FLASH_TOKEN_IDS=0` + `GREEDY_STEPS` + `buffered` 读 +
两个 `cublas` 臂 + `libdevice` 打分 + 常驻显存**默认开**，不设任何常驻变量）：

| 运行 | 步骤 | 结果文件 | exit | 钉死值 |
| --- | --- | --- | --- | --- |
| `a3-1step` | 1 | `/srv/localmoe/targets/a3-shared-diag/a3-1step.jsonl` | 0 | 全复现 |
| `a3-8step` | 8 | `/srv/localmoe/targets/a3-shared-diag/a3-8step.jsonl` | 0 | 8 摘要全等 |

- `a3-1step`：`argmax=55222`、`hidden_bf16_digest=5ed8bf88beae3cfe`、43 层、`block32`、
  `canonical_cpu_streamed_bf16`、峰值显存 `8175288320`——`verify_baseline.py` 判 `baseline reproduces`。
- `a3-8step`：`outputs_match.py` 对 `defaulton/default-8step.jsonl` 判 **IDENTICAL**
  （8 digest + 8 层链 + 8 argmax + 峰值显存）。

新增字段实测（稳态 = token 1–7，301 条层记录，`phase_profile.layers[*].top_level.shared`）：

| 字段 | 含义 | 稳态均值 |
| --- | --- | --- |
| `wall_ns` | 窗口墙（MLP 与读盘取最慢） | `14.66 ms/层`（`0.630 s/token`） |
| `diagnostics.main_ns` | 主线程 shared MLP 墙 | `4.09 ms/层` |
| `diagnostics.join_tail_ns` | MLP 返回后等 worker 的尾巴 | `10.57 ms/层` |
| `diagnostics.read_worker_max_ns` | 最慢那条 lane 的读墙 | `14.26 ms/层` |
| `diagnostics.read_worker_sum_ns` | 六条 lane 读墙求和 | `56.16 ms/层` |
| `diagnostics.read_workers` | 参与读的 lane 数 | `6` |

交叉核对：`read_worker_sum_ns`（`56.16`）与同批 `top_level.routed.diagnostics.read_ns`
（`56.11 ms/层`）一致——**同一笔读，在两处各记一次，量值对得上**，证实读盘确实只归 `routed`，
`shared` 行的墙主要是等它的 join 尾巴。token 0 第 0 层样本：`wall=22.01 ms`、
`main=3.86 ms`、`join_tail=18.15 ms`、`read_worker_max=21.50 ms`、`read_worker_sum=125.18 ms`。

## 代码变更

`rust/colibri-cuda/src/`，提交 `0e515f7`（交付分支 `codex/business-restart`）：

- `layer_rows.rs`：把 shared 窗口**本来就在量**的两个数（主线程 MLP 墙 `shared_probe_mlp_us`、
  六条 worker 的读墙 `shared_probe_worker_{sum,max}_us`）在 `shared_ns` 之后写进 `stage`。
  这些计时**今天就已经无条件采集**，`COLIBRI_V4_FLASH_SHARED_PROBE=1` 只控制那行 `eprintln!`，
  所以这是纯搬运，热路径没加计时。
- `flash_arms.rs`：`LayerStageTelemetry` 加 `shared_main_ns` / `shared_read_workers` /
  `shared_read_worker_sum_ns` / `shared_read_worker_max_ns`，并加进 `AddAssign` 的 `fold!`
  （chunked prompt pass 折叠多个位置时一起累加）。
- `b2_generate.rs`、`phase_profiles.rs`：两个相位 JSON 的 `shared` 行在 `wall_ns` 旁多一个
  `diagnostics` 对象（含 `join_tail_ns = wall - main`）。
- `full61_b1.rs`：full61 那条平铺记录同样补四个字段。

为什么不改数值：相位 JSON 不进 `ordered_layer_digest_sha256`；四个新字段只被序列化，不参与任何
计算路径。测试用的 `LayerStageTelemetry` 字面量都带 `..default()`，加字段不会破坏它们。

## 未决事项

1. **本次盘是慢档 `1.8 GB/s`**（历史快 `3.66`/慢 `2.7`）。上面那张相位表用于字段关系核对没问题，
   但**别拿它的绝对时间跨时段比**。要报收益得在快档重跑一批。
2. `shared` 行的诊断现在还只在 `phase_profile`（命令行默认出；服务面要设
   `COLIBRI_V4_FLASH_SERVE_PHASE_JSONL`）。服务面在这条路径上还没跑过——与 `现状.md` §四.1 同源。
3. 交付分支 `tests.rs` 里 `flash_b1_layer_phase_profile_json` 还有两处**旧的 6 参数调用**
   （函数已 7 参数，`719b5f2` 加 `chunk_positions` 时没同步）。这是本次之前就有的，`cargo test`
   因此编不过；本次没动它，验收走的是真构建 + 真跑（`run-identified.sh`）。要不要顺手修，待定。
4. A3 只是把账摊开，没有加速。按 colibri 对比账，下一步是 A2（按专家身份的缓存）与 A1（读盘与
   专家 kernel 重叠）。
