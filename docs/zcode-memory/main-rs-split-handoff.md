---
name: main-rs-split-handoff
description: main.rs 拆分已全部完成（45,540 → 559 行，40 个模块，kernels 走 include!）——剩余形态、证据位置、踩过的坑
metadata:
  node_type: memory
  type: project
  originSessionId: sess_143112a0-6216-4d90-a9b1-ad9750f379c1
---

2026-09-16 把 `rust/colibri-cuda/src/main.rs` 从 45,540 行拆到 **559 行**，业务代码全在
同目录独立文件里（40 个模块）。做法、约定、验证门槛、踩过的坑全写在仓库里的
`rust/colibri-cuda/MODULE_LAYOUT.md`（**下次先读它**，别凭记忆拼）。

main.rs 现在只剩：crate 属性、顶层 `use`、LUT 常量、两个 `const fn`、`tilelang_sum4` 宏、
40 行的模块声明块、一行 `include!("kernels_module.rs");`、`fn main()`（391 行）。

## 本会话的提交（都在 codex/business-restart 上）

`3f3bae6` → `bd92e75` → `77a1130`（前三批，25,381→14,708 行）→ `4ff6223`/`1258acc`（文档）→
`dda14ea`+`4b4332c`（一次搬完剩余 23 个主题模块，14,708→3,408 行）→ `b268b88`（kernels 走
`include!`，3,408→559 行）→ `5c5ca34`（文档）。

**`include!` 这条路以前标着「没验证过」，现在验证过了**：把 `#[cuda_module] mod kernels {...}`
整体放进 `kernels_module.rs`，main.rs 里写 `include!("kernels_module.rs");`，编译 0 error、
59 条警告，真机 261 个 digest 与基线全同。被包含的文件不能有 `use super::*;` 头。

## 证据文件（勿覆盖）

`/srv/localmoe/targets/` 下的 `0731-sm120-batch6-refactor.jsonl`、`-batch78-refactor.jsonl`、
`-bigextract.jsonl`、`-kernels-included.jsonl`——最后两次对应「23 个模块」和「kernels include!」
这两个大改动，都与基线 `0731-sm120-clean-cuda128-default-block32-r2.jsonl` 的 261 个 digest 字段
逐一相同（`argmax=55222`、`hidden_digest=5ed8bf88beae3cfe`、`peak_device_used_bytes=1466499072`）。

切块/校验脚本在 Windows 侧 `C:\Users\27236\AppData\Local\Temp\moe_split\`（`cut_batch*.py`、
`verify_batch*.py`、`cut_batch9.py`、`spike_kernels.py`、`compare_run.py`）——临时目录，
长期要用得先搬进仓库。

## 三个坑（下次别再踩）

1. **切口不能重叠**：删除按行号从后往前做，重叠会让同一段抽进两个文件（`ambiguous name`）
   并且错位多删内容。脚本里先 assert 区间互不重叠。
2. **`wsl.exe -e bash -c` 里没有 `cargo`**：跑构建必须 `-lc`，否则 `exit=127`。
3. **WSL 会自己重启**（本会话遇到一次）`/tmp` 被清空、后台长任务被杀——长跑的东西别只靠
   `/tmp` 留证据，也别把「后台任务在跑」当已完成的依据。

## 下一轮（用户 2026-09-16 决定往下走函数级）

交接写在仓库里的 `交接说明-函数级重构.md`（**先读它**；另一条线 FP4 oracle 的交接是
`交接说明.md`，别混）。要点：

- 文件级搬运能给「原文逐字节相同」的证明，**函数级给不出**，只剩行为对比（真机 261 个 digest
  + 测试套件），而它们只覆盖单步基线链跑到的路径。纪律：一次拆一个函数 → 立刻真机对 digest
  → 不一致就回退。
- 待拆清单：全 crate 1,409 个函数，>200 行的 43 个、>500 行的 7 个。最大的四个是
  `run_real_v4_layer_stationary_prefill`(3,360)、`run_real_v4_full61_b1`(989)、
  `run_real_layer_rows`(790)、`prepare_generic_device_attention_row`(678)。
- 建议从 `run_real_layer_rows`(790) 起步（在基线路径上、digest 覆盖得到）；3,360 行那块
  最好等有明确改动目标时「边改边拆」。`main()`(391) 是 dispatch 样板，最后再说。
- 另一种选择是不拆、回到交付线（连续多 token / 服务）——按 AGENTS.md，进度只看真实调用链
  能跑到哪一步，拆文件本身不推进它。

并发注意见 [[concurrent-agent-sessions]]；构建路径与旧 rsync 目录见 [[wsl-run-environment]]；
断电风险见 [[hardware-instant-power-loss]]。
