# 服务面终于能出逐相位账了：一个默认关的落盘口（2026-09-20）

## 结论

交付的服务面（`COLIBRI_V4_FLASH_SERVE=1`，`c/openai_server.py` 起的那条）**以前一次都没写过
`phase_profile`**。原因不在引擎：服务走的是和命令行同一条 `run_v4_b2_generate_arm`，逐层的相位
数据本来就收集在 `arm.requests[0].flash_b1_phase` 里（`flash_b1_profile` 对 `FLASH_0731` 且
单请求成立）。**是网关把结果路径删掉了**——`c/openai_server.py:1736` 在 spawn 引擎子进程前
`child_env.pop("COLIBRI_V4_RESULT_JSONL")`，而引擎写结果文件唯一的依据就是这个环境变量。
所以服务面不是"没有数据"，是"没有地方落"。

**改法（默认关，不动协议）**：给服务面加一个可选落盘口
`COLIBRI_V4_FLASH_SERVE_PHASE_JSONL=<路径>`。设了它，引擎每完成一个请求就往该文件**追加**一条
JSON，schema 与命令行结果文件对齐（同样带 `phase_profile` / `physical` / `requests` /
`peak_device_used_bytes`），所以现成的相位读脚本不用改就能读。**不设它（默认）行为与原来逐字节
相同**：不多建文件、不碰 stdio 协议、不碰生成的 token。

**验证**：

- **设了开关**：客户端一次真实 prompt 的请求 → 落盘 `1` 条记录，`phase_profile.layers` 有
  `172` 项（43 层 × 4 趟：prompt chunk 把那 12 个 prompt token 折成 1 趟，加 3 个生成 token）、
  `final_heads` `4` 项、`peak_device_used_bytes = 1825112064`（与命令行合并树同值）、
  `ttft_ms = 28714.854`、`itl_ms = 3219.937`。
- **不设开关**：同一个二进制、同一条 prompt，产出的 4 个 token 与设开关那次**逐字符相同**
  （`69146,69146,69146,69146`），且**没有创建任何 jsonl 文件**。
- 这条路没有动数值路径（只加了一个 `Option<File>` 和一段 `if enabled` 的分支），所以
  token 相同是构造上成立的。

**这次没有声称的部分**（别照着脑补）：

- **不是"服务面变快了"**。这是让它能记账，不改一行数据流。设开关那次 `TTFT 28.7 s`、
  ITL `3.22 s`，与状态页记的合并后三跑冒烟（`TTFT 20–23 s`、`ITL 2.92–3.36`）同一档；
  单次请求的 TTFT 里含进程冷启动，**别和那三跑直接比**。
- **服务面的首字那段仍没拆**。这条只把逐层相位账放了出来；`TTFT` 里"预填充 vs 冷启动"的切分
  是下一步的事（见未决事项）。
- **`prompt_tokens=12` 是 prompt chunk 折过之后的口径**：`layers` 是 172 而不是 `43 × (1+4)`，
  因为首个 prompt 趟一次覆盖了 12 个位置。读这张账要和命令行帧的 `chunk_positions` 一起读。

## 运行证据

- 源码树：WSL `/srv/localmoe/build/deliver-merge-0918`，`codex/business-restart` @ `fd4a864`
  （本记录对应提交）；构建 `CARGO_TARGET_DIR=/srv/localmoe/targets/deliver-merge-0918-target`、
  `--release`、`62 warnings`、exit 0、约 23 s（增量）。
- 客户端：`rust/colibri-cuda/tools/flash_serve_client.py`（它自己设 `COLIBRI_V4_FLASH_SERVE=1`，
  并透传环境里的 `COLIBRI_V4_FLASH_SERVE_PHASE_JSONL`）。
- 结果文件（WSL 绝对路径）：
  - `/srv/localmoe/targets/servephase-on.jsonl`（设了开关，1 条记录）
  - `/srv/localmoe/targets/servephase-on.log` / `servephase-off.log`（两跑 stdout）
  - 不设开关那次**没有** `servephase-off.jsonl`（这是断言的一部分）
- 臂：`MHC_LINEAR=cublas`、`ROUTER_LINEAR=cublas`、`ROUTER_SCORE=libdevice`；
  `FP4_ACCUM_ARM` / `FINAL_HEAD_ARM` 不设。
- 复现脚本：`.zcode/servephase_verify.sh`。

## 代码变更

- `rust/colibri-cuda/src/generate.rs`：新增 `ServePhaseSink`（读
  `COLIBRI_V4_FLASH_SERVE_PHASE_JSONL`，`None` 时一切 no-op），在服务请求循环里于
  `write_done` 之前追加一条记录。

## 未决事项

1. **把服务面的首字拆开**：现在能拿到逐层账了，下一步是用它把 `TTFT` 里的"预填充趟"
   和"进程冷启动"分开，并核对稳态 ITL 与命令行稳态 s/token 的对应关系。
2. **网关侧的开关**：`c/openai_server.py` 目前不认这个环境变量（它照常删 `RESULT_JSONL`）。
   如果要在正式服务里用它，需要让网关把这个变量显式传进子进程；当前用 `flash_serve_client.py`
   直达引擎这条验证路径不需要改网关。
