---
name: own-the-roadmap
description: 用户不想一步一步给约束；照 handoff 自己排优先级、自己拍板、自己执行，只在真正的取舍上给结论
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess_81c9d7aa-77c6-42cd-9e8d-aeea419b13de
---

2026-09-17：用户说"我感觉你可以根据上一个 handoff 自己去规划接下来怎么优化更合理了，我一步一步约束感觉有点浪费自己的注意力？"

**Why:** 他关心的是项目实际往前走了多少，不是替我当调度器。每一步都回来问，花掉的是他的注意力，而我手里有比他更全的现场证据（相位账、探针结果、同一批 A/B）。这条和 [[real-runs-over-proxy-verification]] 是一套：自己排的计划要靠真跑收口，不能停在纸上。

**How to apply:** 拿到 handoff 先自己排出下一步、自己执行到底（真构建、真跑、真比对 digest），只有"改交付默认值 / 动显存预算 / 推翻已有结论"这类真取舍才专门给一条结论加理由。排期写成 `.zcode/plans/plan-<日期>-<主题>.md` 当工作底稿，做完一条就在里面标上结果和提交号。汇报先讲做了什么、拿到什么数，再讲剩下的次序——不要用"要不要我……"开头。

[[experiment-branch-hygiene]] [[plain-language-reporting]] [[flash-0731-serve-path-speed]]
