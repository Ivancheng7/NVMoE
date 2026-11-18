---
name: real-runs-over-proxy-verification
description: 用户要的是真跑一遍（真构建、真推理、真出 token），不接受用 stub 或脚本 PASS 之类的代理证据代替实测
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess_762ae631-6606-4d6a-b051-8f64f6991571
---

汇报验证结果时，用户要的是**真跑一遍**。2026-09-16 我用 stub cargo 加参数重写测试证明了 `RUSTC_WRAPPER` 接线正确，并在结尾列了「没做的：没有跑完整的真实构建来验证」外加理由（怕抢另一个会话的 CPU）——用户直接回「那你实测一下」。补跑之后才拿到决定性证据：对照组 exit 101 + ICE、实验组 exit 0 + 59 条警告、真实推理 token `55222` 与峰值显存 `1466499072` 逐字节复现。

**Why:** 这和仓库自己的标准一致——AGENTS.md 写着「什么算做完：用户能用真实输入生成多个 token……单 token fixture、合成 kernel、哈希通过、脚本 PASS，单独都不算」。代理证据只能证明机制通了，证明不了整条链没坏。

**How to apply:** 改完构建或推理路径就直接跑真的，别先把「没跑真构建」写进汇报当免责声明。担心抢资源时，改用隔离目录（[[flash-0731-baseline-run-recipe]] 里的验证台）或挑机器空闲时跑，而不是把验证降级成代理形式。

相关：[[plain-language-reporting]]、[[flash-0731-baseline-run-recipe]]
