# 2026-09-17：final head 流式 CPU 臂按输出神经元并行（sm_120）

## 结论

把层循环外那个 0.45 s/token 的 final head（词表 129280 × 4096 的 bf16 点积扫）按输出
token 切段分给 6 个线程后，**稳态每 token 从 468 ms 降到 110 ms，省 358 ms/token**；
8 步整请求 45.9 s → 36.7 s。数值上，新代码 `threads=6` 与 `threads=1`、以及交付分支
原版串行代码的 8 步产物**逐位一致**（344 层逐层 digest 链、8 个 token 的
hidden/logits/collapsed/normalized digest、argmax 全部相同），钉死基线值
`argmax=55222`、`hidden_bf16_digest=5ed8bf88beae3cfe` 原样复现。

逐位一致不靠测：`canonical_final_head_row_dot` 是每个输出神经元自己的有序 fold，
神经元之间完全独立，按 token 切段不改变任何一次点积的加法顺序；argmax 由主线程在
拼好的 logits 上跑与原版完全相同的升序严格 `>` 扫描，最低 token id 平局规则原样保留。

没声称的部分：只改了 `final_checkpoint_head_cpu_timed` 这一条 CPU 路径（CLI base 与
serve 共用）；`gpu_exact` 臂没动，`final_checkpoint_logits_from_normalized_cpu`（DSpark
臂用的同款循环）也没动。token 0 的 853→98 ms 是页缓存冷热差，不属于本次改动的功劳。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`（ext4 原生）
- revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 48 个 safetensors 分片，共 166,886,535,336 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`

## 运行证据

代码在 `exp/final-head-parallel` 分支，提交 `42d94e0`（父提交 `20fbecc`，即 2026-09-17
的交付分支 HEAD）。

构建（WSL clone `/srv/localmoe/build/localmoe-clean-git`，checkout `exp/final-head-parallel`）：

```
cd /srv/localmoe/build/localmoe-clean-git/rust
CARGO_TARGET_DIR=/srv/localmoe/targets/final-head-par-target \
  ./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda
# Finished release in 54.26s，60 warnings，exit 0
```

cuBLAS：`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`（与基线一致）。

同一批内背靠背两轮，同一二进制，只改 `COLIBRI_V4_FINAL_HEAD_THREADS`；其余臂与
基线配方相同（`STREAM_READ_MODE=buffered`、`PREFETCH=1`、MHC/ROUTER_LINEAR=cublas、
ROUTER_SCORE=libdevice、`GREEDY_STEPS=8`、`FLASH_TOKEN_IDS=0`；`FP4_ACCUM_ARM` 和
`FINAL_HEAD_ARM` 故意不设）。脚本 `.zcode/final-head-ab.sh`（未跟踪），结果文件：

- `/srv/localmoe/targets/final-head-par-t1.jsonl`（threads=1，request_e2e 45.94 s）
- `/srv/localmoe/targets/final-head-par-t6.jsonl`（threads=6，request_e2e 36.66 s）

final head 每步墙（`phase_profile.final_heads[].outer.wall_ns`，ms）：

| token | t1 | t6 | delta |
| --- | --- | --- | --- |
| 0 | 853.0 | 97.6 | +755.4 |
| 1 | 640.0 | 106.4 | +533.6 |
| 2 | 482.6 | 128.0 | +354.6 |
| 3 | 417.2 | 97.4 | +319.8 |
| 4 | 405.0 | 109.6 | +295.4 |
| 5 | 417.0 | 100.0 | +317.0 |
| 6 | 426.4 | 116.1 | +310.3 |
| 7 | 489.4 | 116.0 | +373.3 |
| 稳态均值（除 0） | 468.2 | 110.5 | **+357.7** |

比对脚本 `.zcode/final_head_ab_check.py`（未跟踪），判定项：8 个
`ordered_layer_digest_sha256`、8 组 hidden/logits/collapsed/normalized digest、
argmax、`generated_ids`，t1 vs t6 **全部逐位一致**；t6 另与交付分支原版串行代码的
`/srv/localmoe/targets/baseline-nonresident-8step.jsonl`（2026-09-16 产物）比对，
同样逐位一致。两轮峰值显存相同（`1823014912`），`final_head_source_bytes` 均为
1,059,061,760（读的总字节没变，只是分了 6 份读）。`generated_ids` =
55222, 33832, 42549×5, 127384。

## 代码变更

全部在 `rust/colibri-cuda/src/v4_state.rs` 的 `final_checkpoint_head_cpu_timed`：

- 505 个权重 chunk 按 `COLIBRI_V4_FINAL_HEAD_THREADS`（默认 6，1–16，越界回 6）切成
  连续段，`std::thread::scope` 里每个 worker 用自己的缓冲 `pread` 自己的段——
  `ShardFile` 读路径是 `&self` + `file.read_at`（pread），没有共享游标。
- 每个 worker 把自己段的 logits 写进私有 `Vec`，join 后主线程拼回
  `base.logits_f32`，再做一次与原版相同的升序严格 `>` 扫描得出 argmax。
- `read_ns`/`streamed_source_bytes` 按 worker 汇总，语义不变。

DSpark 臂用的 `final_checkpoint_logits_from_normalized_cpu` 是同款循环但不在交付链上，
这次没改。

## 未决事项

1. **合并进交付分支**：代码在 `exp/final-head-parallel`（`42d94e0`），按分支管理规则
   cherry-pick / 合并进 `codex/business-restart` 并复跑基线核对四个钉死值后生效。
   合并后预期整 token 约 3.70 → 3.34 s（常驻配置下）。
2. **服务路径未量**：serve 与 CLI 走同一个 `final_checkpoint_head_cpu_timed`，预期同幅
   受益，但本次没单独跑 serve。
3. **默认线程数 6 是拍的**：6 物理核上 6 线程实测 4.2×；没扫 4/8/12 线程，也没看它
   与后台预取线程抢核的影响——两轮 read_ns 都在 50–200 ms 抖，说明读与算已有重叠。
4. **DSpark 臂的同款循环**（`final_checkpoint_logits_from_normalized_cpu`）若将来上
   交付路径，可以照抄这次的切法。
