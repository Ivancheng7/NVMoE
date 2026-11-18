---
name: terms-glossary-industry-alignment
description: 2026-10-03 建 docs/术语表.md；2026-10-09 全库把自造方言清成 0（墙/臂/相位账→wall time/arm/phase profile 等）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_1694f5e1-f4b7-4970-b96c-c9093430f909
---

2026-10-03 用户说「我们这些优化与 ai infra 的专业名词在我的理解上没有对齐」，落了
`docs/术语表.md`（三层对照）。

**2026-10-09 用户要求「把文档整理干净，自造词与生硬翻译换成行业标准词，没有标准译名的直接
用英文」——已全库执行完毕。** 覆盖 35 个 md（含 26 篇实验记录、7 份交接说明、现状.md、
README.zh-CN、MODULE_LAYOUT、AGENTS.md），约 600 处。

- 主要替换：墙/墙钟→wall time、相位账→phase profile、臂→arm、快/慢档→fast/slow tier、
  门卫/闸门/门禁→gate、天花板/地板→上限/下限、钉死值→golden values、收尾→由谁决定、
  卡点→阻塞点、撞上→碰上、吃掉→占掉、藏身→重叠窗口、砍掉→减掉、晃/抖→波动、
  塞进→放进、削掉→消掉、钱/值多少钱→开销/收益大小、花销→开销、拍板→决定、游标→cursor。
- 边界：**代码块（原始输出）不重写**、数字/哈希/路径/命令/链接一律不动；中英混排只在非
  README 文档补空格（README 保持紧凑风格）。事实不变量核对过：数字/hex/链接/fence 集合一致。
- 记录与工具在 `.zcode/scratch/`：`dejargon.py`（规则表）、`polish.py`（手工修正）、
  `factcheck.py`（不变量核对）、`dejargon.diff`（全量 diff）、`before/`（改前快照，553 KB，可删）。
- `docs/术语表.md` §一 现在是「旧写法 → 现在怎么写」的对照表 —— 读旧记录/旧记忆/旧对话时用它解码。
- 没动的：`rust/colibri-cuda/tools/*.sh` 与 `.zcode/*.sh` 的中文注释仍是旧词（工具副本，不属于
  文档）；它们的 echo 本来就是英文，所以以后跑出来的原始输出不会带旧词。

**Why:** 用户看不懂 撞墙/上墙/相位账 这类比喻，明确「不如用英文词组更容易理解」。术语表建了
但正文没照做，2026-10-09 才真正执行。

**How to apply:** 写任何中文文档/回话时按 `docs/术语表.md` 的约定：行业有标准中文词用中文
（上限/下限、稳态、常驻、锁页、开销），没有的直接用英文原词（wall time、phase profile、arm、
tier、gate、golden values），别再造比喻；一个概念全库一个写法。
[[docs-plain-register]] [[flash-0731-vs-vllm-five-stage-map]] [[plain-language-reporting]]
