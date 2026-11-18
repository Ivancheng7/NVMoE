# ZCode 记忆（localmoe-clean）

> 2026-11-05 从 ZCode 迁入：`~/.zcode/cli/memories/projects/<workspace>/memory/`。
> 共 63 篇。这些是 ZCode 期间沉淀的**结论 / 坑 / 偏好**，不是会话原文；
> 会话全文在 `Desktop\ZCode迁移-20261105\sessions-text\`，原始库在同目录的 zip 里。


> **先读 `MEMORY.md`**：那是 ZCode 自己的索引，把记忆按主题串成了线，还带着「别再提议 X」这类硬约束；本文件只是逐篇的一句话摘要 + 失效标注。

读的时候注意：写于 2026 年 9–10 月，**事实以当前代码和配置为准**；
下面标 ⚠ 的几篇绑定 ZCode 自身机制，结论可能已不成立。

| 文件 | 说明 |
|---|---|
| `MEMORY.md` |  |
| `ai-infra-layer-map.md` | AI Infra 六层地图 + 两处「用户仓库里已经有」的挂钩（MTP 就是他的 DSpark/mtp1_*、NVFP4 ≈ v4_fp4.rs 的 E2M1 block32）+ 社区量化命名两套体系与读陌生引擎的五部件法 |
| `ai-infra-learning-direction.md` | 2026-10-03 用户拍板 AI infra 学法——从 nano-vLLM 框架内核入手、目标工业界标准；10-06 补算子/kernel 品味路线（三问+七套路+matmul 四步曲）与 Auto Kernel 任务背景 |
| `bandwidth-wall-decides-what-optimization-pays.md` | 判据是「贴在哪面墙」而不是模型大小——带宽墙算术（每 token 字节 ÷ 带宽）、27B@3bit vs 0.6B@常驻 的数字对照、以及分页 KV + 连续批处理为何在单用户场景失效 |
| `cargo-check-ice-wrapper.md` | localmoe-clean 里普通 cargo check 会在钉住的 nightly 上 ICE（原始树也一样），必须用 |
| `codebase-memory-mcp-install.md` ⚠ | codebase-memory-mcp 装在哪、为什么必须设 CBM_RUNTIME_DIR 和 CBM_CACHE_DIR 两个环境变量（安装步骤已废；图谱已接回 DSH，索引还是原来那份） |
| `colibri-is-our-c-engine.md` | 仓库根 c/ 就是上游 JustVugg/colibri 引擎快照，DeepSeek-V4 路径是另写的 Rust；colibri 快的机制清单 |
| `concurrent-agent-sessions.md` ⚠ | localmoe-clean 的工作树可能同时被多个 ZCode 会话改写，动手前必须先看 .zcode/plans/ 和 git log（多会话抢分支的结论仍适用，机制描述不适用） |
| `concurrent-branch-hijack-during-commit.md` | 工作树被并发会话切分支，commit 会落到错的分支上；改完先 git branch --show-current |
| `cuda-anchor-symbol-requires-run-identified.md` | 直接 cargo build 会链接失败报 cuda_oxide_artifact_anchor undefined，必须走 |
| `deepseek-line-scope-decision.md` | 2026-10-03 拍板：DeepSeek 线优化收尾；CUDA graph 对我们无用（~0.1%，且录不进去）但在 |
| `deepseek-v41-flash-facts.md` | DeepSeek-V4.1-Flash（CED 架构）已核实的事实、两处我曾说错的纠正、官方参考怎么拿，以及 2026-09-16 决定暂不移植 |
| `delivery-binary-embeds-experiment-modes.md` | 交付二进制里编着约 30 个实验模式、129 个开关；19 个模块交付文件零引用但混着共享 helper，所以清理必须按函数切不是按文件搬 |
| `disk-speed-two-tier.md` | 本机盘速在 2.7 / 3.65 GB/s 两档之间晃约 25%，是宿主机侧竞争，客户机查不到；跑性能数字前先量一次 |
| `docs-plain-register.md` | 这个项目的中文文档要写工程师记事口吻，不要管理/汇报腔；用户 2026-09-16 明确说管理腔让他"很难受 |
| `experiment-branch-hygiene.md` | 实验性代码必须放 exp/* 分支，交付分支只保留已验证状态；回滚用追加式 revert，不改写历史（多会话共享同一分支） |
| `flash-0731-attention-compressor.md` | Flash-0731 的 attention 每 token 1.0 s（新默认帧，占 27%）；构成已翻转——16 个小步 |
| `flash-0731-attention-device-chain.md` | attention 那 18 个同步点已搬设备侧（exp/attention-device-chain），数值逐位不变，相位 19.54→13.34 ms/层 |
| `flash-0731-attention-plus-pinned-stack.md` | attention 设备链 + routed 读锁页两个改动改同层不同相位，合并后各自保持增益、18 跑 digest |
| `flash-0731-baseline-run-recipe.md` | 复现 Flash-0731 sm_120 业务基线的确切环境变量、命令与期望输出（token 55222 / digest 5ed8bf88beae3cfe） |
| `flash-0731-bottleneck-map.md` | 用实测硬件上限标出的瓶颈地图：PCIe 14.27 GB/s 是结构下限（712 ms/token），哪些到顶哪些还能压 |
| `flash-0731-dspark-arm-state.md` | 本地 DSpark 臂的真实状态——接受长度 0（2026-10-01 在新交付线上重量仍 0/20）；第一处缺陷是自写 rope kernel >4 行静默清零，第二处未定；官方参考实现本机齐备 |
| `flash-0731-expert-slot-cost.md` | routed 专家槽位是每 token 最大单项；实测 90% 花在建 8 块 pinned 主机缓冲上，开销跟字节走不跟调用次数，修法是复用槽位而不是动同步 |
| `flash-0731-final-head-cost.md` | final head 是 0.45 s/token（第 4 大项、在层循环外），其中 0.38 s 是单线程标量 CPU 流式；按输出神经元并行逐位不变，是剩下最便宜的一块 |
| `flash-0731-final-head-parallel.md` | final head 按输出 token 切 6 线程已实测 468→110 ms/token（省 358 |
| `flash-0731-fixed-h2d-bandwidth.md` | 固定权重现在只剩"每层 H2D 137.4 MB"（1.01 s/token、占 27%），其中分配/释放占 414 ms； |
| `flash-0731-fixed-resident-vram.md` | 整模型固定权重常驻显存已默认开（b620ef7）：稳态 −0.88 s/token（−28%）、峰值 1.83→8.18 GB、数值逐位不变；单步钉死峰值随之改 |
| `flash-0731-is-the-delivery-target.md` | 交付目标是 Flash-0731 revision 9e165c30…；文档里长期写的 60d8d707… 是另一份 checkpoint，2026-09-16 已统一 |
| `flash-0731-official-oracle-unstable.md` | 官方 FP4 核在 sm_120 上算错，根因是钉死的 tilelang 0.1.8；换 0.1.14 已实测修好（torch 无关） |
| `flash-0731-per-token-floor.md` | 一个 token 的账——交付线 2026-10-01 实测稳态 1.998 s/token（单 token 形状、三次均值）；下面是各历史形状，别当现状 |
| `flash-0731-phase-profile-reading.md` | Flash-0731 引擎自记的相位账怎么读才对（顶层是互不重叠的墙、diagnostics 会重复、shared 是 max() 而不是自己的耗时），以及定版不常驻基线的每 token 分解 |
| `flash-0731-pinned-lane-read-landed.md` | 读盘直接进锁页已落地实验分支：routed 窗口 22.9→18.5 ms/层、alloc=0、8 digest 逐位同、八步省 |
| `flash-0731-pinned-read-green-light.md` | restage 省掉的前置量测已做：读进锁页 vs 复用缓冲无差异，go；工具与证据的分支是 exp/routed-pinned-replay |
| `flash-0731-prefill-row-cost.md` | B 线预填充事实与已量出的层墙模型 layer≈attention+router+max(MoE窗口,固定权重读)； |
| `flash-0731-routed-expert-cache.md` | routed 专家按身份跨 token 缓存（A2，默认关）：复用 63%、读盘 −57~79%，但墙只 −3.6%（读被藏在 |
| `flash-0731-routed-fanout.md` | routed 六专家并行下发（默认关）实测 −17~19%；新的收尾是最慢那条 lane + 被顶起来的 MLP，诊断不能再相加 |
| `flash-0731-routed-read-ceiling.md` | routed 专家读盘的天花板实测——盘不是瓶颈（六条并发流 3.79 GB/s vs 单线程 0.8–1.4），另一项独立开销是每次读进新申请缓冲；含分片/专家布局与证据位置 |
| `flash-0731-serve-path-speed.md` | 服务与命令行已同批量过，每 token 一样快（3.94 vs 4.04 s）；4.81 s/字 只是提交旧了，不是两条路不一样 |
| `flash-0731-serve-phase-sink.md` | 服务面逐相位落盘口已加（fd4a864，默认关 COLIBRI_V4_FLASH_SERVE_PHASE_JSONL）；根因是网关删了 RESULT_JSONL |
| `flash-0731-steady-state-vs-first-token.md` | 8 步总时长里 ~40% 是 token0 建固定参数常驻库的一次性成本；判收益要用稳态 s/token，不是步数固定的墙 |
| `flash-0731-three-arms-merged.md` | 三件已量改动已合进交付分支 codex/business-restart（dc22c8e/628c8c0/db35193）——attention 设备链 + routed 读锁页 + final head 并行；合并树真跑 8 token、数值逐位不变、同批 A/B 增益叠加 |
| `flash-0731-vs-vllm-five-stage-map.md` | 我们的引擎对着 vLLM 五步学习路线各做到哪——前向/KV/prefill-decode 做透、调度只做排队、prefix |
| `hardware-instant-power-loss.md` | 开发机（技嘉 B650M + 7500F + RTX 5060 Ti）会无预警硬件级瞬时断电，平均两三周一次，未提交工作和长任务有丢失风险 |
| `jargon-vocab-writeback.md` | jargon-decoder 讲过的英文词会写回雅思项目的单词本，落点和复习命令在哪里 |
| `kernel-course-progress.md` | AI infra 算子课进度（第 1–3 节讲完）——硬件实测常数、精度/累加/tiling 的全部关键数字、两个测量陷阱、AutoKernel 查清的问题、下节=延迟隐藏 |
| `kernel-landscape-2026.md` | 与 colibri-cuda 直接相关的 2026 年算子库/参考实现/工具链外部指针（TileKernels、sm120 |
| `llamacpp-wsl-build-recipe.md` | WSL 里编译 llama.cpp(sm_120) 的完整配方与四个坑,gcc/glibc/CUDA 版本组合是关键 |
| `localmoe-abandoned-worktree.md` | Desktop\localmoe 主仓库工作区里有一坨从未提交的废弃工作（stage 阶梯 + manifest/oracle |
| `localmoe-code-shape-not-quality.md` | localmoe-clean 引擎代码是「形状失控」而不是「水平低」——用户说它被落后 AI 污染时要用证据纠正，别顺着清理 |
| `main-rs-split-handoff.md` | main.rs 拆分已全部完成（45,540 → 559 行，40 个模块，kernels 走 include!）——剩余形态、证据位置、踩过的坑 |
| `nano-vllm-lab-wsl.md` | nano-vLLM 学习实验室已跑通——位置 /srv/localmoe/lab、Windows 侧学习路线文档、四个环境坑（Python 3.14/flash-attn 编译/HF 不通/oracle 工具链）与实测数字 |
| `ninfer-fusion-kvmem-5060ti.md` | NInfer 融合引擎(KVMem 环)在 5060 Ti 上的完整跑通记录:全套 DLL 坑、真实速度数字、与 llama.cpp 对照、draft A/B、三个猜想头横向判词 |
| `optimize-the-optimization-loop.md` | 用户 2026-09-17 指出真正要优化的是"优化流程"本身——量完就搬、插桩不复制树、状态一页纸、实验批量脚本化 |
| `own-the-roadmap.md` | 用户不想一步一步给约束；照 handoff 自己排优先级、自己拍板、自己执行，只在真正的取舍上给结论 |
| `plain-language-reporting.md` | 这个项目的用户看不懂技术行话，汇报必须用大白话加日常类比 |
| `qwen38-27b-lab-baseline.md` | Qwen3.8-27B 在 RTX 5060 Ti 16GB(WSL llama.cpp) 的实测基线:混合 GDN 架构、量化选型、tok/s、TTFT、显存、MTP 实测（能跑，深度=1 才赚） |
| `real-runs-over-proxy-verification.md` | 用户要的是真跑一遍（真构建、真推理、真出 token），不接受用 stub 或脚本 PASS 之类的代理证据代替实测 |
| `resident-on-is-the-comparison-baseline.md` | localMoE |
| `run-step-naming-convention.md` | 顶层函数前缀规约——run_ = 可单独跑的程序，step_ = 模型执行里的一步；51 个函数已按此改名 |
| `terms-glossary-industry-alignment.md` | 2026-10-03 建 docs/术语表.md；2026-10-09 全库把自造方言清成 0（墙/臂/相位账→wall time/arm/phase profile 等） |
| `v4-state-hardcoded-geometry.md` | v4_state.rs 的常量块整块是 Pro profile 副本；只有 EPS/FP8_BLOCK 卡 v4.1，其余是重复真值源，别提前动 |
| `wsl-run-environment.md` | WSL2 上 localMoE 真实推理的运行环境：源码克隆、构建必须带 RUSTC_WRAPPER、模型与 cuBLAS 路径 |
| `zcode-project-hook-trust.md` ⚠ | ZCode 的项目级 hook 被「工作区信任」闸门挡住，以及这个客户端根本没有 Grep/Glob 工具——写 hook 前必看的两条实测事实（ZCode 项目 hook 信任机制，已随卸载消失） |
