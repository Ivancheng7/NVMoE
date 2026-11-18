# localMoE 开发规则

## Code Exploration — 代码图谱优先（仅当 MCP 已接入）

若当前客户端接入了 `codebase-memory-mcp`，任何读或改本仓库代码的任务都必须先用图谱查询，
Grep/Glob/Read 不是第一步。图谱已建索引，是主要导航来源。

若没有接入（例如当前环境未配置该 MCP，或只是普通 shell），一句话说明「图谱不可用，走文本检索」，
再用 `grep` / `glob` / `read`，不要把没跑过的图查询写进结论。

- Project identifier: run `list_projects` and use the dash-form name it returns
  for this repo's absolute path. If a query errors or returns nothing, re-read
  that value before falling back.
- Symbol / implementation lookup: `search_graph` (`query=`) or `search_code`.
- Callers, callees, impact: `trace_path`, `detect_changes`.
- Repo-level shape: `get_architecture`.
- Exact source: `get_code_snippet`, only after the graph has narrowed scope.

Grep/Glob are permitted ONLY when one of these holds, and you must say which:

  (a) a graph query with the verified project name returned nothing;
  (b) `check_index_coverage` shows the file unindexed or `parse_partial`;
  (c) cross-checking a known graph false negative. Two classes are
      documented: closure-internal call edges (e.g. `advanceAutomated`),
      and method calls whose receiver is not `this` — capture there is
      unpredictable (in one measured repo only 538 of 1169 Method nodes had
      any inbound CALLS edge). The direction itself is fine; the gap is edge
      coverage. So treat every `x.method()` conclusion as unverified —
      including `get_code_snippet(include_neighbors)`'s `callers`/`callee_names`
      and `search_code(mode="full")`'s `in`/`out` columns — and confirm it by
      grep before reporting it as fact.

## 项目目标

localMoE 在消费级 NVIDIA GPU 上跑真实的 MoE 模型，对外提供命令行或
OpenAI 兼容服务。进度只看一件事：真实推理路径能跑到哪一步。文档数量、
校验数量、实验记录数量都不算数。

运行环境是 WSL2、RTX 5060 Ti 16 GB（`sm_120`）和 CUDA Toolkit 12.8 或更新版本。

交付目标是官方 `deepseek-ai/DeepSeek-V4-Flash-0731` revision
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`。代码里 pin 的、本地有权重的、
实验跑的，都是这一份。`origin/main` 里的 `sm_89` 实现是起点，不是终点。

## 先做什么

一次只打通一条真实调用链，从权重到服务输出：

```text
真实权重 -> 模型层执行 -> 专家路由 -> KV Cache -> 连续生成 -> 服务输出
```

这条链没跑通之前：

- 不加 gate、manifest、oracle 或实验框架；
- 不为还不存在的第二个调用方设计抽象、工厂或通用框架；
- 不为了“更完整的防御”推迟能跑的代码。

## 工程约束

- 优先复用当前可工作的实现，改动保持小而直接。
- 每个新行为只配一个最小可运行检查，检查必须服务当前这条调用链。
- 性能改动要保留正确性检查和前后 benchmark，但不把 benchmark 记录系统做成产品本身。
- 模型权重只能放在 WSL 原生 ext4 块设备上；不得放在 Windows 挂载路径上。运行时会拒绝
  `/mnt/c` 和 `/mnt/d`；其他 `/mnt/*` 只要背后是块设备上的 ext4 就允许，README 示例用的
  `/mnt/colibri-native` 就属于这一类。
- CUDA kernel、stream、event 和显存所有权必须明确；不得添加隐藏同步。
- C/CUDA 可以作为迁移期间的参考实现，Rust 版本先在真实调用方中替换，再逐步删除旧路径。
- 任何说明文件使用中文，且只记录会影响后续开发的决定。
- 最终仍使用 Rust 与 `cuda-oxide` 承担 Colibri 自有 host/kernel 路径，但迁移必须沿着
  已经跑起来的真实调用链逐段发生。

## README 固定文本

除非用户明确给出替换文本，不得改动以下四个字符串：

- `Want to Try Deploying a Full 1.6T MoE Model on Your Gaming Laptop?`
- `想不想在你的游戏本上试试亲自全量部署1.6T的MoE大模型呢？`
- `Does Your Next LLM Infrastructure Really Need to Be an H100?`
- `你的下一台LLM基础设施，何必是H100`

## 什么算做完

用户能用真实输入生成多个 token，并且结果从 CLI 或服务返回，才算做完。
单 token fixture、合成 kernel、哈希通过、脚本 PASS，单独都不算。
