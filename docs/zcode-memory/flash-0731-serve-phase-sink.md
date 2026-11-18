---
name: flash-0731-serve-phase-sink
description: 服务面逐相位落盘口已加（fd4a864，默认关 COLIBRI_V4_FLASH_SERVE_PHASE_JSONL）；根因是网关删了 RESULT_JSONL
metadata:
  node_type: memory
  type: project
  originSessionId: localmoe-clean-a1f0d1dadf0989bf
---

2026-09-20 给交付服务面加了**默认关**的逐相位落盘口（`fd4a864`）。

**为什么服务面以前一次都没写过 `phase_profile`**：不是引擎没有数据——服务走的是和命令行
同一条 `run_v4_b2_generate_arm`，逐层相位本来就收集在 `arm.requests[0].flash_b1_phase`
（`flash_b1_profile = FLASH_0731 && 单请求`，服务面成立）。**是网关把结果路径删了**：
`c/openai_server.py:1736` 在 spawn 引擎前 `child_env.pop("COLIBRI_V4_RESULT_JSONL")`，
而引擎写结果文件唯一的依据就是这个变量。所以是"没有地方落"，不是"没有数据"。

**改法**：设 `COLIBRI_V4_FLASH_SERVE_PHASE_JSONL=<路径>`，引擎每个完成的请求往该文件**追加**
一条 JSON，schema 与命令行结果文件对齐（`phase_profile` / `physical` / `requests` /
`peak_device_used_bytes`），现成相位读脚本不用改就能读。**不设（默认）行为与原来逐字节相同**：
不建文件、不碰 stdio 协议。

**验证**（`rust/colibri-cuda/tools/flash_serve_client.py` 直达引擎，不走网关）：
设开关落 1 条记录（`layers` 172 = 43×4 趟、`final_heads` 4、peak `1825112064`、`ttft_ms 28714.854`）；
不设则**一个文件都不建**，两跑的 4 个 token 逐字符相同（都是 `69146×4`）。
复现 `.zcode/servephase_verify.sh`，记录 `docs/experiments/2026-09-20-sm120-serve-phase-sink.md`。

**坑/边界**：`prompt_tokens=12` 是 prompt chunk 折过之后的口径（首趟一次覆盖 12 个位置，
所以 4 个生成 token 只有 172 层趟而不是 43×5）。

**2026-10-01 更正 + 补齐**：上面那句「网关侧目前不认这个变量」**是错的**。网关
（`c/openai_server.py:1734`）写的是 `child_env = dict(os.environ ...)`，只 pop 掉
`RESULT_JSONL` 和 final-head dump 目录，**其余变量原样下传给引擎**。在网关进程的环境里设
`COLIBRI_V4_FLASH_SERVE_PHASE_JSONL` 就能穿透——2026-10-01 走真网关实测：
`POST /v1/chat/completions` 返 200，落一条 **207,312 字节**的记录，`phase_profile` 在里面
（172 层条目、4 个 final head、`peak_device_used_bytes=8223522816`）。**「服务不写相位账」
这一项到此关闭。**

**同一次实测还暴露一个文档缺口**：网关进程**必须**带 `COLIBRI_CUBLAS_EXACT_LIB`
（绝对路径、非符号链接、SHA-256 与钉死值一致）。不设的话每个请求 500，日志里是
`ENGINE strict cuBLAS exact mode requires absolute COLIBRI_CUBLAS_EXACT_LIB`；引擎侧没有兜底
（`cublas_exact.rs:121` 就是 `ok_or(...)`），网关自己也不设。**用 `flash_serve_client.py` 测不出这个坑**
——它自带 `--cublas-lib`。另外请求里的 `model` 必须等于网关自己的 id（默认 `glm-5.2-colibri`），
写别的名字会 404。`localmoe-wsl-run` 技能第 4 步已按这两条改正。

首字也拆了，见 [[flash-0731-serve-path-speed]]。

相关 [[flash-0731-serve-path-speed]]、[[flash-0731-phase-profile-reading]]。
