---
name: nano-vllm-lab-wsl
description: nano-vLLM 学习实验室已跑通——位置 /srv/localmoe/lab、Windows 侧学习路线文档、四个环境坑（Python 3.14/flash-attn 编译/HF 不通/oracle 工具链）与实测数字
metadata:
  node_type: memory
  type: project
  originSessionId: sess_7df38066-8b46-4d73-bb93-e1c5c31669ac
---

2026-10-03 按 [[ai-infra-learning-direction]] 的拍板，在 WSL 里搭好了 nano-vLLM 实验室
并**真跑通**（不是只装好）。

## 东西在哪

| 东西 | 位置 |
| --- | --- |
| nano-vLLM 源码（1450 行） | WSL `/srv/localmoe/lab/nano-vllm` |
| 实验室 Python 环境 | WSL `/srv/localmoe/lab/venv-nv`（torch 2.8.0+cu128 + flash-attn 2.8.3） |
| Qwen3-0.6B 权重 | WSL `/srv/localmoe/lab/models/Qwen3-0.6B` |
| 复现脚本 | WSL `/srv/localmoe/lab/run_first.py`；`EAGER=1` 关 CUDA graph、`EAGER=0` 开 |
| 日志 | WSL `/srv/localmoe/lab/run.log`、`flashattn.log` |
| **学习路线正文（给用户看的）** | Windows `C:\Users\27236\Desktop\ai-infra-lab\学习路线.md` |
| **AI Infra 六层地图（给用户看的）** | Windows `C:\Users\27236\Desktop\ai-infra-lab\AI-Infra-地图.md`（见 [[ai-infra-layer-map]]） |

代码编辑：VS Code 开 `\\wsl.localhost\Ubuntu\srv\localmoe\lab\nano-vllm`。

## 四个环境坑（都踩过，别重踩）

1. **仓库要求 `python>=3.10,<3.13`，WSL 默认 python3 是 3.14.4**，直接装失败。
   用这台机器已有的受管 CPython 3.12.13。
2. **huggingface.co 从这台机器不通（curl 000）**，权重走 ModelScope（实测 25 MB/s）。
   `download.pytorch.org` 直连反而有 14 MB/s——别被 pip 的慢速误导成"网络不行"。
3. **flash-attn 不能源码编译**：CUDA 12.8 与系统 gcc 15 头文件打架
   （`nvcc` 报 `__is_pointer is undefined`，`-ccbin` 也绕不过），且机器只有 15 GB 内存，
   flash-attn 巨型单个 `.cu` 编译单元极易 OOM。**走预编译轮子**：
   先把 torch 降到 2.8.0+cu128，再装
   `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
   （GitHub release `v2.8.3.post1`；ABI 要 TRUE，用
   `torch._C._GLIBCXX_USE_CXX11_ABI` 确认）。**flash-attn 没有 torch 2.10 的轮子**，
   所以降 torch 是必须的，不是可选。
4. **`/srv/localmoe/toolchains/flash-reference-py312` 是 localMoE oracle 工具链**，
   只能 `cp -a` 出来用，绝不能就地装包。实验室环境就是这么复制的。

## 实测数字（RTX 5060 Ti 16 GB、sm_120、Qwen3-0.6B）

32 个独立请求 × 每个 128 token：

| 配置 | 墙钟 | 吞吐 |
| --- | --- | --- |
| `enforce_eager=True`（无 CUDA graph） | 3.78 s | **1083 tok/s** |
| `enforce_eager=False`（CUDA graph 开） | 0.88 s | **4676 tok/s** |

**同一份代码只开 CUDA graph 快 4.3 倍**——0.6B 的 kernel 小到 launch 开销压过计算。
这正好量化了 [[flash-0731-vs-vllm-five-stage-map]] 第 5 条那个"空白"值多少。

## 顺带挖到的两条框架知识（给用户讲课时用）

- **flash-attn 的分页 KV 块大小必须是 256 的整数倍**（报错原文
  `Paged KV cache block size must be divisible by 256`）。这正是 nano-vLLM
  `config.py` 里 `assert kvcache_block_size % 256 == 0` 的来源——所以"把 block_size
  改成 64"那个实验根本做不了，只能在 256/512/768 里试。
- **nano-vLLM 禁止 `temperature=0`**（`greedy sampling is not permitted`），
  因为 `layers/sampler.py` 用 `logits/temperature` 做**指数竞速采样**
  （`probs / Exp(1)` 再 argmax，等价于类别采样但全程逐元素、GPU 批量友好）。
  生产框架不会禁，而是给 greedy 单开一条 argmax 路径。

**Why:** 用户要的是"能改能量的实验室"，不是"装好的环境"；四个坑每一个都能吃掉半小时
到几小时，且第 3 条（flash-attn 编译）看起来可行实际必失败，不记下来下次一定重踩。

**How to apply:** 用户说"继续学推理框架""接着上次那个实验室"时，直接用这里的路径和
`run_first.py`，别重搭。下一步按学习路线文档第六节那个"改-跑-量"清单走
（block_size 档位、去掉 chunked prefill 限制、加 step() 度量、关前缀缓存、复现 CUDA graph 收益）。
