# sm_120 上官方 oracle 与我们的首个分歧层（2026-09-16）

## 结论

换到 tilelang 0.1.14 之后，官方 full-43 oracle 在 sm_120 上**稳定了**：同一输入跑两遍，
`tools/compare_official_run_repeatability.py` 报 623 个字段逐项相同，两遍 argmax 都是 `55222`。
我们自己的引擎在同一台卡上和它给出同一个 token（`55222`，`hidden_bf16_digest` = `5ed8bf88beae3cfe`）。

但两边**不是逐位相同**。逐层比下来，第一个分歧在 **layer 8 的路由权重**（`route_weight_bits`）：

| 层 | 和我们比 |
| --- | --- |
| 0–7 | **逐位一致**——路由 id 与权重、hidden、KV、主/索引压缩器状态全对 |
| 8 | 只有 `route_weight_bits` 和由此产生的 `hidden_digest` 不同；KV 和压缩器状态**仍然一致** |
| 9–42 | 路由 id、路由权重、hidden、KV、压缩器状态全部不同（第 8 层分叉的下游后果） |

这条把旧印象改了：**routed FP4 专家不再是 sm_120 上的分歧来源**。0–7 层里 MoE 的 routed 专家
（hash router 的 0–2 层和 learned router 的 3–7 层都有）逐位一致，第 8 层也是"先路由权重、再 hidden"，
KV 和压缩器到第 8 层还是对的。

没声称的部分：**没有判断哪一侧是对的**，两边 token 相同、中间量不同，这次只定位分歧起点；
"逐位一致"只覆盖 layer 0–7，其余层不声称；这次也没有重跑 sm_89 证书。

## 固定输入与模型

- 模型目录 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`（两边结果文件里的 `model_revision` 都是它）
- safetensors 48 个，`shard_file_stat_total_bytes` = `166886535336`
- 官方记录里的源文件 SHA-256：`config.json` `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`、
  `model.safetensors.index.json` `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`、
  `inference/model.py` `c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f`、
  `inference/kernel.py` `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`

## 运行证据

官方那侧（tilelang 0.1.14 旁路，官方 venv 一个字节没动）：

- 脚本 `/srv/localmoe/targets/run_tl014_full43.sh`：`PYTHONPATH=/srv/localmoe/toolchains/tl014-only`，
  `CPATH=/srv/localmoe/compat-headers:/srv/localmoe/toolchains/glibc243-cuda13-math-overlay-v1/include`，
  解释器 `/srv/localmoe/toolchains/flash-reference-py312/bin/python`，`--end-layer 42 --final-head`
- 结果 `/srv/localmoe/targets/0731-official-full43-tl014-a.jsonl`（SHA-256
  `70138c0a45bad9a4168e5b686bbe09e65ea1ff3799c5950b663c3e7db17c790f`，80 s，exit 0）
  与 `-b.jsonl`（31 s，exit 0）；两遍都是 43 层、`full43_oracle=true`、`final_head_argmax_token_id=55222`
- 记在结果里的环境：torch `2.10.0+cu128`、tilelang `0.1.14`、safetensors `0.7.0`、
  `free_vram_bytes=15894315008`、`available_ram_kib=15176508`

我们这侧（当前 HEAD）：

- 构建 `cd /srv/localmoe/build/localmoe-clean-git/rust && CARGO_TARGET_DIR=/srv/localmoe/targets/delivery-8step-target ./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`
  （51.7 s，exit 0）
- 二进制 `/srv/localmoe/targets/delivery-8step-target/release/colibri-cuda`
  （SHA-256 `a08bb7c331ea31e7acdcbd22e2e8e770260176492b377053f0f4c2ce997c7495`）
- 单步结果 `/srv/localmoe/targets/0731-sm120-1step-head.jsonl`（SHA-256
  `86b567ffbe702a4af012a293e6bdb6ffad4dcb2c40398d2e992bb58ffec9af58`）：43 层、`argmax=55222`、
  `hidden_bf16_digest=5ed8bf88beae3cfe`、峰值显存 `1466499072`、`fp4_accum=block32`、
  `final_head=canonical_cpu_streamed_bf16`
- 臂：`mhc_linear=cublas`、`router_linear=cublas`、`router_score=libdevice`；cuBLAS 实际加载
  `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`

比对（两遍都用仓库里的工具，不是临时脚本）：

```bash
python3 tools/compare_v4_flash_0731_full43.py \
  --official /srv/localmoe/targets/0731-official-full43-tl014-a.jsonl \
  --rust-result /srv/localmoe/targets/0731-sm120-1step-head.jsonl \
  --release-binary /srv/localmoe/targets/delivery-8step-target/release/colibri-cuda \
  --profile sm_120-block32 \
  --expected-oracle-sha256 70138c0a45bad9a4168e5b686bbe09e65ea1ff3799c5950b663c3e7db17c790f \
  --release-sha256 a08bb7c331ea31e7acdcbd22e2e8e770260176492b377053f0f4c2ce997c7495
# FAIL: layer 8: route_weight_bits mismatch
```

差别图是用一个临时脚本打的（`first_divergence.py`，按上面的字段集逐层比），结论：
`identical layers (8): [0, 1, 2, 3, 4, 5, 6, 7]`，`first divergence: (8, 'route_weight_bits')`。

## 代码变更

`tools/compare_v4_flash_0731_full43.py`：把写死的 `EXPECTED_ARGMAX = 33832` 换成具名 profile。
`33832` 是 sm_89 那台笔记本上的封存真值，在这台机器上会让工具报假失败，而最顺手的"修法"就是改这个数字。
现在 `sm_89-certified`（argmax 33832，参考环境 tilelang 0.1.8）和 `sm_120-block32`（argmax 55222，
参考环境 tilelang 0.1.14，rust 侧 `fp4_accum` 必须是 `block32`）两个 profile 并存，`--profile` 必填；
profile 里声明的参考环境和 rust 臂与实际记录不符时拒绝比对。测试同步更新
（`tools/test_compare_v4_flash_0731_full43.py`，6 个用例，含 3 个新用例）。提交 `b1dbed9`。

## 未决事项

1. **第 8 层是哪个算子先差，没定位。** 下一个切口：用官方的 cutpoint 机制只抓 layer 8 的 router
   输入/输出（`--cutpoint-dir ... --cutpoint-layer 8`），看是 linear、score 还是归一化先分叉。
2. **谁对没判。** 两边 token 相同、中间量不同，token 相同不能当作判据。
3. **为什么从第 8 层开始，没有解释。** 0–7 层里既有 hash router（0–2）也有 learned router（3–7），
   层号本身不解释这个分界。
4. **0.1.14 能不能在 sm_89 上重生那份 43/43 证书，没试。** 换 tilelang 会改变参考值本身，
   那份证书是 0.1.8 时代的产物，所以这是并存第二套环境而不是替换。
