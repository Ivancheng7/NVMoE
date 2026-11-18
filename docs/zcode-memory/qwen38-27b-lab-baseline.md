---
name: qwen38-27b-lab-baseline
description: Qwen3.8-27B 在 RTX 5060 Ti 16GB(WSL llama.cpp) 的实测基线:混合 GDN 架构、量化选型、tok/s、TTFT、显存、MTP 实测（能跑，深度=1 才赚）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_c911262e-6e45-4df2-9402-d13cd33270e4
---

2026-10-04 真跑([[llamacpp-wsl-build-recipe]] 的构建 + unsloth GGUF)。**2026-10-09 逐张量解剖更正**：共 **27.321B 参数 / 866 张量**（不是 27.8B），架构码是 `qwen3_5`（llama.cpp 里叫 `qwen35`）；**不是普通 dense GQA**——64 层是 16×(3 层 Gated DeltaNet 线性注意力 + 1 层全注意力)，48 层只有固定大小的循环状态、**只有 16 层有 KV**（hybrid 真正省的是 KV：34KB/token@q8_0，全注意力要 4 倍）；FFN 占 63.6% 参数；无 MoE；`config.json` 里有视觉塔但**下载的 GGUF 没有视觉张量也没有 mmproj → 当前是纯文本模型**（想看图要另下 mmproj 走 mtmd）。文件在 WSL `/srv/localmoe/lab/models/Qwen3.8-27B-GGUF/`（unsloth 仓库，含 `MTP-mtp-…-Q4_0.gguf` 旁挂件 1.37GB），另存 `config.json`。**变种解剖 + 学习清单已写成 `C:\Users\27236\Desktop\ai-infra-lab\Qwen3.8-27B-变种解剖.md`**。

**量化配方（2026-10-09 逐张量实测）**：UD-Q3_K_XL 混了 **14 种精度**（IQ4_XS 37.8% / IQ3_S 26.9% / Q5_K 8.9% / IQ3_XXS 6.9% / Q3_K 6.6% / Q4_K 4.9% / IQ2_S 3.1% / Q6_K 2.6% …，平均 ~3.85bpw）；取舍规律=门控投影压最狠（ffn_gate 只给 IQ2_XS）、注意力保精度、**循环状态参数不压**（ssm_a/ssm_dt.bias/ssm_conv1d 全 F32，360 个 F32 张量总共仅 10MB）、输出头 Q5_K 比 embedding Q3_K 精。**带宽墙验算**：13.14GB×26.5t/s = 349GB/s ≈ 448 峰值的 78%（decode 已贴墙）；prefill 867t/s 是 decode 的 33 倍，换算 ≈47 TFLOPS，贴的是算力墙不是同一面。
**16GB 卡的量化判据(实测)**：UD-Q4_K_S 14.3GiB 起就装不下(卡上空闲约 14.3GiB,Q3_K_XL 全配置已用 14.2/16.3GB);**UD-Q3_K_XL(12.23GiB)是装得下的最高质量档**,8K ctx + q8 KV + FA 全开 OK。

**基线数字(UD-Q3_K_XL,WSL,build 0504396)**:
- llama-bench:pp512=868 t/s,tg128=26.25±0.13 t/s(≈ 显存带宽 448GB/s 的 72%,llama.cpp 正常水平)
- 服务面(llama-server,OpenAI 兼容,90 token 中文提问、256 输出):**TTFT=0.72s**,稳态 25.8-25.9 t/s,连发三次波动 <0.1 t/s;加载 12.2GiB 约 15 秒
- 速度档 **UD-IQ3_XXS(10.17GiB):tg128=29.53±0.21 t/s(pp512=849)**,比 Q3_K_XL 快 12.5%,质量再降一档;SHA256 校验过(c0b7c303…)

**MTP 不是死路（2026-10-09 推翻旧结论）**：sidecar 能加载、能出草稿，`LLAMA_ARG_SPEC_TYPE=draft-mtp` + `--spec-draft-model MTP-…gguf`；**但默认深度是负收益**——同 prompt 256 token 对照：不开 MTP **25.57 t/s**、`--spec-draft-n-max 1` **35.63 t/s(+39%，接受率 0.752)**、n-max=2 → 10.18、默认 n-max=3 → 10.24（接受率 0.522）。1→2 之间每链成本从 49ms 跳到 218ms，形状像撞到另一条代码路径（待读源码）。两个参数坑：`--spec-type` 不吃 `=` 形式、写成空格形式会把后面的 `-ngl` 吃掉（用环境变量 `LLAMA_ARG_SPEC_TYPE` 绕开）；**`-np 4`（多 slot/kv-unified）+ MTP 会在加载期崩** `vector::_M_range_check`（已隔离：`-c 8192 -np 4` 崩、`-c 8192 -np 1` 正常，触发条件是 slot 数不是上下文）。日志：`/srv/localmoe/lab/mtp-*.log`。这是 [[flash-0731-dspark-arm-state]] 同族机制在社区栈的现状——官方栈 vLLM 支持，但 FP8 27.8B 装不进 16GB。

**WSL vs Windows 原生对照(2026-10-04 深夜同批,同模型同参数,同日构建)**:原生 b11388(pp512=899.7、tg128=**27.14±0.08**) vs WSL master 0504396(pp512=883.6、tg128=**26.58±0.02**)→ **WSL 慢约 2%**。原生栈已完整落在 `C:\Users\27236\llama-win\llama-bin\`(主包+三个 cu13 DLL,经本机 7897 代理 32 路分片拉完),模型副本在 `llama-win\models\`。结论:2% 不构成迁移理由,留 WSL;追求极限可切原生。代理通道备忘:ghproxy 只有 12KB/s,GitHub 走 `curl -x http://127.0.0.1:7897 --ssl-no-revoke` 可用,并发分片能线性扩到 ~400KB/s。
