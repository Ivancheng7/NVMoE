---
name: v4-state-hardcoded-geometry
description: v4_state.rs 的常量块整块是 Pro profile 副本；只有 EPS/FP8_BLOCK 卡 v4.1，其余是重复真值源，别提前动
metadata:
  node_type: memory
  type: project
  originSessionId: sess_a9e0716d-7d0a-462a-bfe0-b5dd06f16f66
---

`v4_state.rs:14-34` 那组常量**整块是 Pro 的 profile 副本**，不是"忘了配置化"：`REVISION = v4_spec::PRO_REVISION`、`PINNED_SHARD_BYTES = 864_721_029_744`（正是 `PRO.shard_bytes`）、`V4_LAYER_COUNT=61`、`HIDDEN=7168`、`O_GROUPS=16`、`INDEX_TOPK=1024`，每一项都对得上 `v4_spec.rs` 里的 `PRO`。真实情况是**这个模块当初按 Pro 写，Flash 后来靠 `_for(spec, ...)` 变体接上去**，裸常量留给 Pro 路径和编译期表达式。

要分三组，只有第二组叫"配置化"：

**组 A（5 个，约 731 处引用）**：`Q_HEAD_DIM` 335、`HIDDEN` 261、`KV_WINDOW` 118、`Q_HEADS` 99、`VOCAB` 22、`INDEX_TOPK` 14——spec 里已经声明过了。修法**不是读 config，是删掉重复那一份**改用 `spec.hidden` 这种。唯一的难点是 `main.rs:37/40/8100/8101/8106` 有 5 个 `const` 拿这些常量算**编译期**缓冲区大小，改成运行时 spec 就得变成函数。

**⚠ "多分配不会出错"只对缓冲区成立，对判据不成立**（2026-09-17 晚实测到的实例）：拿 Pro 常量做**相等比较**会静默杀掉整个分支。`StrictCompressorArena` 用 `v4_state::HIDDEN`（7168）建池子，`run_layer_compressor` 又用 `input.len() == v4_state::HIDDEN` 决定用不用它——Flash 喂进来的是 `4096`，判据永远为假，**那个默认开的复用池在交付帧里一次都没跑过**，`7e07056` 报的 A/B（"两臂没差别"）其实是开 vs 开。修法是让调用方给宽度（`spec.hidden`）、让判据问池子自己（`9efd662`）；代价是 attention 白付了 `0.55 s/token`。
**How to apply:** 读代码时看到 `v4_state::HIDDEN`/`V4_LAYER_COUNT` 出现在 `==`、`filter`、`matches!`、数组定尺里，就要怀疑它在 Flash 上恒假；写新开关时别拿模型常量当判据，拿被它守住的那个对象自己的字段。
- **组 B（2 个，102 处）**：`EPS=1e-6`（26 处，且是**私有** `const`，引用全在 `v4_state.rs` 内，最好改）和 `FP8_BLOCK=128`（76 处，其中 47 处跨模块）。**这一组才是 v4.1 的阻塞项**——正好是 v4.1 改掉的值（`rms_norm_eps` 1e-6→1e-20、FP8 块 128→32×32）。
- **组 C（2 个）**：`KV_WINDOW=128`（config 里的 `sliding_window`）、`COMPRESS_RATIO=128`，v4.1 没变，不卡，以后再说。

**这个项目的既定模式不是运行时读 config**，是"在 `V4Spec` 里静态声明 + 启动时 `validate_config` 校验 config 对得上"，故意 fail-closed：config 与代码不一致就开不了机，而不是静默按 config 跑出没验证过的数值。所以组 B 的正确做法是**加两个 spec 字段 + 两条校验**，不要改成运行时读。注意 `validate_config` 现在查 16 个 key，`rms_norm_eps` 和 `quantization_config.weight_block_size` 一个都没查。

**How to apply:** 组 B 排在接 v4.1 时的第一个动作，量很小（2 字段 + 2 校验 + 102 处替换）。**组 A 别为了 v4.1 提前做**——v4.1 是 CED，attention/KV/路由本来就要重写，先改这 731 处对 v4.1 毫无帮助。2026-09-16 用户明确说这一项**暂缓**，触发条件是"`main.rs` 拆完 **且** 决定接 v4.1"。验证只有一条：改完跑单步推理，token 必须还是 `55222`——改的是值的来源不是值本身，输出必须一模一样，不一样就说明替换漏了或错位了。

相关：[[deepseek-v41-flash-facts]]、[[flash-0731-baseline-run-recipe]]、[[flash-0731-official-oracle-unstable]]
