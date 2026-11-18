---
name: concurrent-agent-sessions
description: localmoe-clean 的工作树可能同时被多个 ZCode 会话改写，动手前必须先看 .zcode/plans/ 和 git log
metadata:
  node_type: memory
  type: project
  originSessionId: sess_a9e0716d-7d0a-462a-bfe0-b5dd06f16f66
---

2026-09-16 我在 `localmoe-clean` 上做计划时，发现 `.zcode/plans/` 里躺着三份同一天 14:22–14:23 的计划，来自三个不同会话：一个在把 `main.rs` 从 45,540 行拆到 1.5–1.8 万行（只搬文件、函数体不动），一个在写 WSL 技能，一个是我。

后果是实的：会话开始时的 git 快照（3 个已改文件）在我读完之前就变了——另一个会话先提交了 `f213435`，又把 4,594 行测试模块抽成 `tests.rs`，`main.rs` 里还留着半个 `mod tests_continued`；我改文档期间它又提交 `d5a1b02` 收尾（`tests.rs` + `tests_continued.rs`）。整个过程没有任何冲突报错，因为撞车是静默的。

**Why:** 同一工作树两个会话同时写，`git status` 只反映某一瞬间，靠它判断安全是不够的。

**How to apply:** 在 `localmoe-clean` 里动手前，先看 `.zcode/plans/` 有几份、`git log` 有没有新提交、目标文件是否正被别的会话改。发现并发就先问用户怎么排，别直接开工。提交时只暂存自己要动的文件，不要 `git add -A`。反之，如果另一个会话正在拆 `main.rs`，就别碰它，也别另起一个重构跟它抢。

那个重构会话的做法是**每搬完一组就提交一次**——我这一轮里它又提交了 `d5a1b02`（测试模块抽成 `tests.rs` + `tests_continued.rs`，`main.rs` 少 5,060 行）和 `bc2b1ca`（抽出 layer-stationary prefill），新模块陆续出现 `layer_ops.rs` / `device_attention.rs` / `generate.rs` / `slice_checks.rs` / `artifacts.rs` / `final_head.rs`。它的计划里写死了三条约束：`mod kernels` 必须留在 `main.rs` **内联**（cuda-oxide 的 `#[cuda_module]` 要求内联模块才看得到 kernel 签名）、不引入 `lib.rs`（crate 是 bin-only）、`main.rs` 第 1–73 行头部不能动。

**2026-09-16 用户给的排法（仍然有效）**：另一个会话在拆代码期间，**我只动文档、代码一律不碰，等拆分收尾**。

**2026-09-16 晚补充：并发的引擎跑会污染计时。** 我在 `localmoe-clean` 上量 routed 读盘时，另一个会话正在跑预填充分支（`prompt-chunk-target/release/colibri-cuda`，21:08 起，load 3.1）。那一轮八步墙 90–95 s，安静机器上同一份代码是 51–91 s；`outer/层` 从 190 抬到 250。教训：**跑计时类实验前先 `pgrep -af colibri-cuda` 和 `cat /proc/loadavg`**，发现并发就别拿那一轮的绝对值下结论（同批交错的相对值还能用）。另：`drop_caches` 会牵连别人的跑，动手前想清楚。

**同一次争用，两个实验各自都记了，方向是双向的**：预填充那边也点名了这条读盘实验的二进制。量化一下噪声地板——预填充那批里有两对**做功完全相同**的对照（同二进制、只差一个不改变行为的开关），墙钟自己就差了 **12.8%** 和 **1.5%**。想验证时别信"看上去没别人跑"：`nvidia-smi --query-compute-apps` 比 `pgrep` 更能看到对面的进程（名字不同），而 WSL 刚重启过意味着页缓存是冷的，那本身又是另一层漂移。

相关：[[hardware-instant-power-loss]]、[[wsl-run-environment]]、[[localmoe-code-shape-not-quality]]、[[flash-0731-baseline-run-recipe]]
