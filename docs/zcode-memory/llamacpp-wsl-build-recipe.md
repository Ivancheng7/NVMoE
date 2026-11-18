---
name: llamacpp-wsl-build-recipe
description: WSL 里编译 llama.cpp(sm_120) 的完整配方与四个坑,gcc/glibc/CUDA 版本组合是关键
metadata:
  node_type: memory
  type: project
  originSessionId: sess_c911262e-6e45-4df2-9402-d13cd33270e4
---

2026-10-04 在 WSL 实验室(/srv/localmoe/lab/llama.cpp,构建 0504396)编译 llama.cpp 成功,带 CUDA、CMAKE_CUDA_ARCHITECTURES=120。这批环境(Ubuntu 25.x,gcc 15.2 默认,gcc-14 可用,glibc 2.42,CUDA 12.8/13.1 都装在 /usr/local)踩过的坑:

1. CUDA 12.8 的 nvcc 默认挑 gcc 15 → 报 "unsupported GNU version"。**必须 `export CUDAHOSTCXX=/usr/bin/gcc-14`**(只传 `-DCMAKE_CUDA_HOST_COMPILER` 在 cmake 4.2 的编译器探测阶段不生效;CUDAHOSTCXX 环境变量才有效)。
2. gcc 版本过了之后还有一层:**glibc 2.42 在 mathcalls.h 新增/改了 cospi/sinpi/rsqrt(共 6 个含 float 变体)声明,与 CUDA 头文件里不带 noexcept 的旧声明冲突**(12.8 撞 cospi/sinpi/rsqrt,13.1 撞 rsqrt,错误都是 "exception specification is incompatible")。解法:给 `/usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h` 里那 6 个声明补 ` noexcept(true);`(备份在 math_functions.h.bak-glibc242)。CUDA 13.1 躲不开这个坑。
3. 新版 CLI 参数改名:`-fa` 吃 `on|off|auto` 不再吃 1;llama-bench 没有 `-c/--ctx-size`(上下文随 -p/-n 推);草稿参数是 `--spec-draft-*`(--draft-max 已移除);llama-server 安全警告默认打。
4. **Qwen3.8 系(qwen35 arch)的投机解码在 build 0504396 上是好的——2026-10-09 更正**:当时判成"坏的"是**用法问题**。挂独立 MTP GGUF 时,`-np 1`(单 slot)能正常加载并出草稿,`-np 4`(多 slot,kv-unified)才崩 `vector::_M_range_check`(已隔离:触发条件是 slot 数、不是上下文大小);`-md 主文件` 自草稿那个用法本身就错。**用法与实测数字见 [[qwen38-27b-lab-baseline]]**;主 GGUF 内嵌的 blk.64.nextn 草稿头默认确实被当 unused tensor 忽略,要显式挂 sidecar。
5. **`--spec-type` 参数解析有坑**:不吃 `=` 形式(报 invalid argument),写成空格形式会把紧跟的下一个 flag 当取值吃掉(报 unknown speculative type);**用环境变量 `LLAMA_ARG_SPEC_TYPE=draft-mtp` 最省事**。

从 Windows 调 WSL 的注意:后台任务的 stdout 是 wsl.exe 的 UTF-16 混合流,Read/iconv 都读不利索;**让所有输出落到 WSL 侧文件再 cat**,别依赖任务日志本身。
