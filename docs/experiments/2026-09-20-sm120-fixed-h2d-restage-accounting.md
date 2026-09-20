# fixed.h2d_ns 少算了一段：固定层槽位的 restage 没进计时（2026-09-20）

## 结论

交付分支上**只有一处纯记账错误**，不改推理行为：固定层设备槽位（`7cdd0b1` 落地的那条）
把整层固定权重从宿主负载 restage 进复用设备缓冲，但这段拷贝发生在
`load_b2_raw_runtime_from_host_ref_with_weights` 自己那个 H2D 计时器**开始之前**，
所以 `phase_profile` 里 `fixed.h2d_ns` 只记到那笔很小的 raw `wo_a` 上传。后果：

- **`fixed.h2d_ns` 报的速率高得物理上不可能。** 实测同一二进制、同一 8 步：
  槽位开 `host_resident_hit` 那 301 层合计 `46962 MB / 1947 ms = 24.12 GB/s`；
  槽位关是 `46962 MB / 9296 ms = 5.05 GB/s`（同一批字节数一字不差，见下表）。
  这台机的 PCIe H2D 实测上限是 `14.27 GB/s`，`24.12` 超了 69%。
- **那段时间没有消失，是被算到别处去了**：槽位开时 `window.residual_ns` 里藏着
  `4051 ms`，槽位关时只有 `1.8 ms`。残差本来就该只装"等预取线程"之外的零头。

**修法**：在 `generate.rs::load_flash_b1_layer_from_host` 里，把 `take_restaged` 和随后的
loader 调用一起夹在一个计时器里，再用这个跨度覆盖 `fixed_h2d_ns`。`take_restaged` 在
槽位空着时是 no-op（每种层型的第一个 token 那层），所以那些层的数值与原来完全一样。

**验证**：1 步四个钉死值原样复现；8 步 `host_resident_hit` 速率从 `24.12` 落到
**`8.67 GB/s`**（低于链路上限，落在槽位关那档 `5.05` 与链路上限之间），`window.residual`
从 `4051 ms` 落到 `1.2 ms`；8 个 `hidden_bf16_digest`、8 条层链、8 个 `argmax` 与合并基线
`postmerge-slot-8step-resident-on.jsonl` **逐项相同**（只差那个已知过期的峰值显存值）。

**这次没有声称的部分**（别照着脑补）：

- **不改变任何输出**：改动只在记账（多夹一个 `Instant`，不碰数据流）。逐位相同是这条的性质，
  不是靠测才成立。
- **`routed.diagnostics.read_ns` 偏高那条没碰**（那是 `现状.md` 里另一条未决项）。
- **服务面没重跑**：这条改动只在命令行帧里验了；服务面本来就没有逐相位表（见未决事项）。
- **`8.67 GB/s` 不是"新的真实带宽"**：`fixed_h2d_ns` 现在夹住的区间里可能还包含一点点
  非拷贝的簿记时间；它只是从"物理不可能"回到"物理可能"。要拿它当带宽用，还得再拆。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- upstream revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 关键文件 SHA-256：`config.json` / `model.safetensors.index.json` 与既有的
  `2026-09-18-sm120-merge-three-arms.md` 同一份，未重算（同目录、revision 相同）。

## 运行证据

- 源码树：WSL `/srv/localmoe/build/deliver-merge-0918`，`codex/business-restart` @
  `f5b991b`（本记录对应提交），基线参照是它的父 `d8c66c8`。
- 构建：`cd rust && CARGO_TARGET_DIR=/srv/localmoe/targets/deliver-merge-0918-target ./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`
  → `Finished release`、`62 warnings`、exit 0。
- cuBLAS：`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`（与既定配方一致）。
- 臂：`MHC_LINEAR=cublas`、`ROUTER_LINEAR=cublas`、`ROUTER_SCORE=libdevice`；
  `FP4_ACCUM_ARM` / `FINAL_HEAD_ARM` 故意不设（验证默认仍是 `block32` / CPU final head）。
- 结果文件（绝对路径，WSL）：
  - `/srv/localmoe/targets/h2dfix-1step.jsonl`（1 步，PASS）
  - `/srv/localmoe/targets/h2dfix-8step.jsonl`（8 步，PASS）
  - 对照：`/srv/localmoe/targets/postmerge-slot-8step-resident-on.jsonl`（合并基线）
  - 复现问题的 A/B：`/srv/localmoe/targets/fixedslot-slot-on.jsonl`（槽位开）、
    `/srv/localmoe/targets/fixedslot-slot-off.jsonl`（`COLIBRI_V4_FLASH_FIXED_SLOT=0`）

修复前后与对照，同一批字节（`46962.05 MB`）的 `host_resident_hit` 合计：

| 产物 | `fixed.h2d_ns` | 速率 | `window.residual_ns` |
| --- | --- | --- | --- |
| `fixedslot-slot-off`（槽位关） | `9295.7 ms` | `5.05 GB/s` | `1.8 ms` |
| `fixedslot-slot-on`（槽位开，未修） | `1947.4 ms` | **`24.12 GB/s`** ← 超链路 | `4050.8 ms` |
| `h2dfix-8step`（槽位开，已修） | `5418.9 ms` | `8.67 GB/s` | `1.2 ms` |

链路上限对照：`14.27 GB/s`（`.zcode/io-ceiling/h2d_rate6g.py`，2026-09-18 两次独立一致）。

判据脚本：`.zcode/verify_baseline.py`（1 步四个钉死值）、`.zcode/outputs_match.py`
（8 步 digest / 层链 / argmax 逐位比）。

## 代码变更

- `rust/colibri-cuda/src/generate.rs` `load_flash_b1_layer_from_host`：加一个计时器夹住
  `take_restaged` + loader，用跨度覆盖 `loaded.fixed.fixed_h2d_ns`。
- `rust/colibri-cuda/src/b2_runtime.rs`：把 `load_b2_raw_runtime_from_host_ref_with_weights`
  上方那段注释从"restage 层会在这里报 shard open（即计时不含 restage）"改成说明计时现在由
  调用方夹住。注释与新行为对齐，不改逻辑。

## 未决事项

1. **`8.67 GB/s` 里到底多少是拷贝**：现在这段跨度还夹着 `take_restaged` 里的长度校验等
   簿记。要把它当带宽用，得再拆成"restage 拷贝"与"wo_a 上传"两段。
2. **服务面仍然不写 `phase_profile`**（`现状.md` 未决项 1）。`openai_server.py:1736`
   主动 `child_env.pop("COLIBRI_V4_RESULT_JSONL")`，所以服务子进程根本没有结果文件可写；
   要在服务面拿到逐相位表，得先给服务一条单独的落盘路径。
3. **`routed.diagnostics.read_ns` 偏高**（合并树上 `59–68` 对磁盘天花板记录 `35.9 ms/层`），
   独立于本条。
