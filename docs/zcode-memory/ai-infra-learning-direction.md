---
name: ai-infra-learning-direction
description: 2026-10-03 用户拍板 AI infra 学法——从 nano-vLLM 框架内核入手、目标工业界标准；10-06 补算子/kernel 品味路线（三问+七套路+matmul 四步曲）与 Auto Kernel 任务背景
metadata:
  node_type: memory
  type: project
  originSessionId: sess_7df38066-8b46-4d73-bb93-e1c5c31669ac
---

2026-10-03 用户说做这个项目很久了，提到 Qwen3.8 / 27B 这类「推理框架」，想更深入地
学 AI infra。我给了三条路，**用户已拍板走第 3 条——从 nano-vLLM 的框架内核入手，
目标是工业界标准。** 原话：「我希望达到工业界标准，从 Nano VLM 开始，也就是从那个
框架内核开始。」用户同时点明推理框架不只是 vLLM/SGLang，还有 llama.cpp。

## ⚠️ 更正：Qwen3.8-27B 是真实模型（我最初判断错了）

我原先写「Qwen3 里没有 27B，dense 是 32B，27B 是 Gemma 3；『Qwen3.8』按 Qwen3-8B 理解」
——**这条已作废，不要再用**。用户两次坚持「是最新版本的模型」，查证后用户是对的
（QwenLM 官方 GitHub README + Ollama 标签）：

- **Qwen3.8 是真实系列**（2026-08-12 发布旗舰 2.4T-A95B；2026-08-14 发布 **27B**）。
  模型卡架构串 `Qwen3_5ForConditionalGeneration`，Apache-2.0，256K 上下文，text+image。
- **Qwen3.5**（2026-02～03）：0.8B / 2B / 4B / 9B / **27B** / 35B-A3B / 122B-A10B / 397B-A17B。
- **Qwen3.6**（2026-04）：2B / 27B / 35B-A3B。
- **Qwen3**（2025-04）：0.6B / 1.7B / 4B / 8B / 14B / **32B** dense，30B-A3B / 235B-A22B MoE。
- **27B 是 3.5 代之后新的中档 dense 档位**；32B 只存在于 2025 那一代。
- Qwen3.5 起改用 hybrid **Gated DeltaNet + Gated Attention + QSA 稀疏注意力**（不再是标准 GQA），
  这个设计一路沿用到 3.8。

**后果**：架构不同的直接结果是 **nano-vLLM / vLLM 的 `models/qwen3.py` 跑不了 Qwen3.8**
——它只实现了 2025 版标准 GQA。要跑 Qwen3.8 只能上 vLLM/SGLang/llama.cpp 的官方实现。
llama.cpp 已在 2026-09 合并 Qwen3.8-Flash-Next MTP 支持。

**教训**：我对 2026 年的模型家谱没有可靠知识（训练截止早于这些发布）。再碰到
「最新版某模型」的说法，先联网查证再下结论，别拿记忆里的旧家谱去否定用户。

## 这台机器上的环境事实

- vLLM / SGLang / transformers / torch / llama.cpp 在 **Windows python 和 WSL python 里都没装**。
- HF 缓存里只有 Qwen3-**TTS**（语音合成）、chatterbox、ModernBERT，**没有 LLM 权重**；
  且 **huggingface.co 从这台机器不通（http 000）**，下模型要走 ModelScope 或 hf-mirror。
- 整台机器是图像/语音栈（ComfyUI、sd-webui、wan2.2、lora-scripts、tts），原先没有 LLM 服务栈。
- WSL 里已有 `torch 2.10.0+cu128` + `tilelang 0.1.8` 在
  `/srv/localmoe/toolchains/flash-reference-py312`——**那是 localMoE 的 oracle 工具链，只能复制不能改**。

## 两条线的分工（别混）

- **学框架内核** → nano-vLLM + Qwen3-0.6B（判据：整个模型能进显存，调度信号才干净）。
- **跑最新模型** → llama.cpp / vLLM + Qwen3.8-27B。16 GB 卡上 27B dense：bf16 54 GB /
  Q8 ≈27 GB / Q4 16–18 GB **都塞不下**，所以**量化必须，但分层不必须**——压到 3-bit
  （IQ3_XXS ≈11 GB）就能全层放 GPU，第三方实测（2026-08）llama.cpp + MTP 约 **48–50 tok/s**
  （用户说的「5060 上很快」指这个）。它是 GGUF 混合量化路线，**和分页 KV + 连续批处理不是一回事**
  ——后者在单用户场景根本失效，理由见 [[bandwidth-wall-decides-what-optimization-pays]]。

## 路线选择的位移

我原本推荐「给现有 Rust 引擎加 paged KV + 连续批处理」，用户改选了 nano-vLLM。
两者目标一致，而且 **nano-vLLM 正好补的正是 [[flash-0731-vs-vllm-five-stage-map]] 里
第 4、5 条那两块空白**（我们只做了排队、没有连续批处理；prefix caching 与 CUDA graph 全无）。
实验室位置、环境坑与实测数字见 [[nano-vllm-lab-wsl]]。

**Why:** 用户把「跑一个 Qwen3 + vLLM」当成学 AI infra 的下一步，但那是「会用」不是「懂」；
真正能学深的材料他手上已经有了，缺的是对照表和能跑的实验室。nano-vLLM 那条正好命中
他自己那张五步表里唯二的空白格。

**How to apply:** 下次直接接着 nano-vLLM 实验室往下走，别重做环境普查、别再问「选哪条路」。
汇报一律走 [[plain-language-reporting]] 的大白话；术语首次出现给英文原词
（[[terms-glossary-industry-alignment]]）。

## 2026-10-27 更正：代码驱动 ≠ 系统学习

用户否掉了「在 nano-vLLM 里跑实验 + 逐行讲代码」这个学法。原话：「我希望的是系统学习，
你这个只是根据代码来的？」以及「你做了我没学懂有啥用」。

**差别**：代码驱动学的是「nano-vLLM 这么写了」，换框架就作废；系统学习要先给约束，
让技术能被**推导**出来，再去对照实现。

**已给的框架**（用户尚未确认）：整个 AI infra 是同一个优化问题，只有三个约束——
① 每算一次要搬多少字节（算术强度 / roofline）② 容量（什么放得下）
③ 一次能并行多少活（能不能填满机器）。所有技术都是这三条的应对：
量化→①、paged KV/prefix cache→①+②、offload/预取→②、连续批处理→③、
CUDA graph→③、投机解码→③、并行切分→③、FlashAttention→①。

**节奏要求**：一次只走一个模块，讲完等用户消化完再走下一步；不许抢跑刷实验、
不许一口气跑完再汇报。注意这与 [[own-the-roadmap]] 不矛盾——「自己排下一步」指的是
排序由我定，不是「执行速度由我定」。

**进度**：用户随后指定先学**算子**（原话「我不想看这个我想看算子的」），
第 1–3 节已讲完，见 [[kernel-course-progress]] 与交接文档
`Desktop/ai-infra-lab/交接-算子课.md`。旧的 `学习路线.md` 方法论已作废
（那是代码驱动的），但其中的实测数字仍有效。
