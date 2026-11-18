---
name: ninfer-fusion-kvmem-5060ti
description: NInfer 融合引擎(KVMem 环)在 5060 Ti 上的完整跑通记录:全套 DLL 坑、真实速度数字、与 llama.cpp 对照、draft A/B、三个猜想头横向判词
metadata:
  node_type: memory
  type: project
  originSessionId: sess_c911262e-6e45-4df2-9402-d13cd33270e4
---

2026-10-05 真跑成功([[qwen38-27b-lab-baseline]] 的对照引擎)。仓库 `C:\Users\27236\ninfer-fusion-kvmem`(上游 NInfer v0.11.0 下游改造,主打 KV 环=小显存跑长上下文+投机解码),引擎 `ninfer-serve-120a.exe`(sm_120a 专用,官方 SHA256 e3e0486a…,在 `C:\Users\27236\llama-win\`),模型 `Ternary-Bonsai-2-27B-ninfer-v3.ninfer`(8.87 GiB,PQ2 完整档:text+mtp+dflash2+proposal+vision,底座 Qwen3.8-27B 三元量化,来自 HF `WaveCut/Ternary-Bonsai-2-27B-NInfer-v3`,SHA256 cdc4810b…逐位吻合)。魔搭 shensanshu 仓**只有源码无权重**。

**量化解剖与速度墙(2026-10-06 对话结论)**:
- 量化两层出身:底座 Ternary Bonsai 2 27B(PrismML)= Qwen3.8-27B 架构不动、端到端**训练成三值** {-1,0,+1}(不是训后量化);.ninfer 制品(WaveCut 转)按组件分级装箱:64 层文本+embedding+LM head 三值 t2_g128_fp16,Hadamard 旋转(块 1024)已折进权重;GDN 线性注意力 A/B/卷积/norm **BF16/FP32 不量化**(状态是累加写的、误差滚雪球,`--gdn-state-fp16` 砍半是选项非默认);MTP 草稿 Q8、DFlash2 适配器 Q4;proposal 头复用 LM 头三值行只留 13.1 万高频 token;视觉塔原版 BF16 0.63GB 不看图不加载。LM 本体 5.9GB(FP16 54GB),比 FP16 小 9.3 倍、14 项基准保 98.2%(IQ2_XXS 同体积只 72.6)
- PTQ1_0/PQ2_0 是两种**装箱格式**不是 PTQ:PTQ1_0 紧塞 1.75 bit/权(省带宽)、PQ2_0 每 2-bit 槽 2.13 bit/权(预填快);我们跑 PQ2 系
- KV 也量化 `--kv-dtype k8v4`:键 FP8(整行 scale)+值 NVFP4(每 16 一组),1.008→0.785 字节/值;粗 scale 档(整行 fp8/int8)会"只吐 2 token 就停"——离群通道凶,细分组是硬要求
- **普通 decode 已贴显存带宽墙**:5060 Ti 448 GB/s,~8GiB 权重/token ≈ 18-20ms → 55-65 tok/s 天花板,实测 54-69 = 已到顶。仅剩两条路:换 PTQ1_0 装箱(字节 −18%)或赌投机接受率(347-517 tok/s 就是这杠杆全开)
- localMoE 搬运判词:锁页读盘/专家并发那套**没处安**(NInfer 全模型常驻显存、无盘上流);能搬的只有方法论(先量贴哪面墙)。我们的 KV 量化**没东西可压**——2026-10-06 查码核实:KV 历史在显存 128 行 ring(见 [[flash-0731-vs-vllm-five-stage-map]]),总量 ~7.8 MB、不过 PCIe
- 概念澄清(用户易混):Hadamard 旋转 ≠ RoPE——RoPE 转的是"表示位置",Hadamard 是固定 ±1 符号表折进权重、把离群值摊匀好压,不含位置信息;三值训练到顶(log₂3≈1.585 bit 是地板)但装箱还剩 PTQ1_0 一档(2.13→1.75 bit,字节 −18%),打完这条路就真到底;投机解码是"只赚不赔"结构(猜错只浪费很便宜的草稿,猜对把 8GB 权重过一遍的成本摊给多个 token)

**5060 Ti 实测(build unknown,36 SMs 校准 52 routed keys)**:
- 启动 65 秒(含 61s 路由校准,二次启动会快),权重 8.16 GiB 载入 2.4s,`runtime 1.98 GiB | free 5.53 GiB`,KV 池 17,920 tokens(280/4096 页)+主机 KV 16 GiB
- **普通中文生成:decode 54-69 tok/s**(dflash2 接受率仅 5-8%,不可预测文本投机倒贴但整体仍远快于 llama.cpp)
- **可预测语料(数数):347-517 tok/s(接受率 78%)**;TTFT 67-310ms;同会话追问 cache 复用 61%+
- 对照:同卡 llama.cpp UD-Q3_K_XL 26.6 tok/s(WSL)/27.1(原生)→ **NInfer 普通生成快 2-2.6 倍**,长上下文多轮快更多(TTFT 70ms vs 720ms)
- 口径警告(作者文档):571.9 tok/s 是"数数字"峰值,别当通用值

**全套 DLL 坑(引擎依赖 = CUDA 3 + FFmpeg 5 + libcurl + MSVC 运行库)**:
1. CUDA 3 件(cublas64_13/cublasLt64_13/cudart64_13)从 llama.cpp 预编译包白嫖
2. FFmpeg 5 件引擎要 avcodec-**63**/avformat-63/avutil-61/swscale-10/swresample-7 = **FFmpeg 9.x**;msys2 的 ffmpeg-9.0.2 包能抽出但二级依赖 30+ 个(大部分已抓齐在 deps-pkg/,少数包如 x264 新版只出静态 exe 不出 dll);**最终解法=BtbN `ffmpeg-master-latest-win64-gpl-shared.zip`(88812913 字节,bin/ 7 件 DLL 自洽整套)**,抽到 llama-win 根目录后 avcodec-63 等全部加载成功
3. libcurl-x64.dll:Git for Windows 的 `libcurl-4.dll` 改名可用,但要把它依赖的 brotli 等带上或让 `C:\Program Files\Git\mingw64\bin` 进 PATH(启动脚本已加)
4. 退出码 -1073741515(0xC0000135)=缺 DLL;python ctypes WinDLL 逐个加载是定位利器
5. 大文件分片下载的坑:ghproxy/代理会把**最后一片截短**,逐片验尺寸不够,**必须拼后验 SHA256**(引擎第一次就是被截了 49.9MB 还骗过逐片校验)

启动脚本 `C:\Users\27236\llama-win\run-ninfer.ps1`(白皮书 argv:KV_WINDOW=16384/RETRIEVE=8192/RING=1/HOST_PAGEABLE=1/REUSE_HOSTBACKED=1,--kv-capacity 17920 --kv-dtype k8v4 --spec dflash2 --draft-tokens 12 --lm-head-draft,端口 8091)。服务面 OpenAI 兼容 `/v1/chat/completions`。风险:初期版本 B01-B20 bug 账本未关完,作者只在 4080S 端到端验过。

**256K 优化配置与长上下文实测(2026-10-05 下午,`run-ninfer-256k.ps1`)**:
- 优化配置:池 65536(运行时 2.98 GiB,卡上剩 4.56 GiB)+ `--host-kv-mib 24576` + `--gdn-state-fp16` + 显式 `NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL=64`(白皮书长题主臂口径);路由标定缓存后启动仅 4.2s
- **40K 针测:turn1 命中**(45,459 tok 预填 47.5s ≈ 957 tok/s,TTFT 47.5s),**turn2 复用 99.9%、TTFT 176ms**——中篇完全健康
- **22.9 万 token 长题能跑但针会 miss**:turn1 预填 763s(池超 3.5×,大量主机页回搬),turn2 复用 100%、TTFT 83s;但 45% 位置的针 turn1/turn2 都没答出(答案漂移成数字)。与 bug 账本一致:B05(query span > MAXQ=256 触发 tail 回退)+ B08(层 0 pre-RoPE 段永不入索引)是已知检索局限,作者 94.7K 主臂(3/3)是 63 页小池跑的,大池+深超池未验
- **B01 铁律**:超池时 max_tokens 必须 ≤2048(社区实测 8192 会崩引擎);并发必须 1(B10)
- 上限判词:16GB 5060 Ti 上 256K 上下文**能装能跑**(权重 8.16+运行时 3+主机层走内存),**但中段检索可靠性在深超池未证**,可依赖的是:最近 32K 窗口 + 会话复用 + 尾部内容;要稳就按 40-64K 用

**代码任务实测(2026-10-06,5 题程序化验证,`llama-win/code-bench.py`)**:5/5 通过(两数之和/快排/JSON/字符串/斐波那契),**整体 191.5 tok/s、单题 162-234**,dflash2 接受率 64-83%(中文才 5-8% —— 代码结构可预测,投机吃满),引擎 decode 口径 290-374 tok/s。写代码是这颗模型在 16GB 卡上的甜点区;用户方向:消费级后训练产品的商业可行性(参照 MiniMax H3 玩法)。

**draft-tokens A/B(2026-10-06,同批同题,temp=0)**:draft=12 代码 205-207 tok/s、中文 59.5;draft=4 代码 137、中文 66。代码题接受率 4 深浅=89% vs 70%,但 12 深窗口的绝对收益更大(代码选 12);中文题 12 深接受率掉到 5-9%(尾部全扔),4 浅=12-32%,中文几乎不赚,深浅差异小 → **结论:代码任务保 draft=12,中文没差,现配置(12)就是对的,不换**。rev_words/fib30 两个 FAIL 是提示词歧义(输出格式不同),非模型问题。

**三个猜想头的横向判词(2026-10-06,z-lab 官方同条件 H200 对比 + 本机实测)**:DFlash2 接受长度 4.4-5.5 > MTP 3.9-5.0 > **DSpark 3.3-4.4(三者中最弱)**;HumanEval 提速 DFlash2 3.1× > MTP 2.2× ≈ DSpark 2.3×。**不换 DSpark(退步),不退回 MTP(NInfer 侧它多占 0.8GiB 预留且慢 40%)**;DFlash2 是当前最强猜头,现配置已是它。MTP 的价值在 llama.cpp 侧(n-max=1 才 +39%,见 [[qwen38-27b-lab-baseline]])。
