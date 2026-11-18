# NVMoE 架构与执行路径说明

本文档说明 `DeepSeek-V4-Flash-0731`（revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`）在消费级单卡（RTX 5060 Ti 16 GB，Blackwell `sm_120`，WSL2 + CUDA 12.8+）上的真实推理链路、显存与磁盘所有权，以及代码目录职责。

## 1. 一条真实调用链

项目对外提供两个入口，共用同一套 43 层精确执行内核（`run_v4_b2_generate_arm`）：

```text
WSL ext4 真实权重
  -> 固定权重显存常驻池 + 6 路并发专家流式加载（pread -> 锁页缓冲 -> 异步 H2D）
  -> 43 层前向（mHC -> 设备侧 MLA/CSA/HCA 注意力链 -> 专家路由 -> Shared MLP + 6 Routed Experts）
  -> KV Cache 与压缩状态更新
  -> 6 线程流式 BF16 Final Head 归约
  -> CLI 结果输出（COLIBRI_V4_FLASH_BASE=1）或 OpenAI 兼容服务流式返回（COLIBRI_V4_FLASH_SERVE=1）
```

### 入口分工

| 入口 | 代码位置 | 职责 |
| --- | --- | --- |
| CLI 生成与基准 | `rust/colibri-cuda/src/generate.rs` (`run_real_v4_flash_base`) | 按给定 token 序列执行贪婪解码，输出生成结果、逐层摘要链与相位耗时 JSONL |
| 本地推理服务内核 | `rust/colibri-cuda/src/v4_serve.rs` + `generate.rs` (`run_real_v4_flash_serve`) | 常驻进程，通过 stdin/stdout 帧协议接收请求，跨请求复用固定权重显存池与锁页缓冲 |
| OpenAI 兼容 HTTP 网关 | `c/openai_server.py` (`--flash-base`) | 提供 `/v1/chat/completions`、`/v1/completions` 与 `/v1/models`，管理排队与 SSE 流式输出 |

---

## 2. 显存预算与数据搬运设计（16 GB 单卡）

1.6T 级 MoE 全量检查点无法整包放进显存，瓶颈不在算力（注意力内 18 个小矩阵乘合计仅占 ~0.3% 耗时），而在**每生成一个 token 需要搬运多少字节**：

1. **固定权重整模型常驻显存（默认开，`COLIBRI_V4_FIXED_RESIDENT_GIB=10`）**
   - 43 层的非路由固定张量合计约 `6.35 GB`。首次走完 43 层后直接留在显存（`B2RawRuntime`），后续每个 token 及后续服务请求的 `fixed` 读取和上传开销归零。
   - 16 GB 卡上开启后峰值显存约 `8.24 GB`，预检逻辑会按当前空闲显存自动留出 `2 GiB` 安全余量；设为 `0` 可退回流式模式（峰值降至 `1.83 GB`）。
2. **路由专家 6 车道并行下发（默认开，`COLIBRI_V4_FLASH_ROUTED_FANOUT=1`）**
   - 每层选出 6 个专家（共约 `3.45 GB/token` 需从 ext4 NVMe 读取），每个槽位拥有独立工作线程、独立锁页内存槽位（直接 `pread` 进 `cuMemHostAlloc` 缓冲，省去主机侧二次拷贝）与独立 CUDA Stream，读盘、PCIe 上传和 FP4 专家核计算在车道间互相重叠。
3. **设备侧注意力链（默认开，`COLIBRI_V4_ATTENTION_DEVICE_CHAIN=1`）**
   - 注意力子步中间张量常驻设备 scratch，消除逐步主机同步；压缩器采用预分配复用池（`COLIBRI_V4_FLASH_COMPRESSOR_ARENA=1`）。
4. **并行流式输出头（默认 6 线程，`COLIBRI_V4_FINAL_HEAD_THREADS=6`）**
   - 层循环外的 `1.01 GiB` 词表投影按输出神经元独立切分多线程归约，保证与单线程标量归约逐位一致（bit-exact）。

---

## 3. 数值可复现基线（Golden Values）

任何性能优化或重构都以不改变前向数值为前提。在 `sm_120` + CUDA 12.8 下，输入 `token_id = 0` 的贪婪解码固定产出：

- **第 1 步**：`argmax = 55222`，`hidden_bf16_digest = 5ed8bf88beae3cfe`，`layers = 43`，`fp4_accum = block32`，`final_head = canonical_cpu_streamed_bf16`
- **连续 8 步 `hidden_bf16_digest` 链**：
  `5ed8bf88beae3cfe, 316e5495ca424b39, a521547b2586e458, 1963dfeaaf30264c, b970da7913a57a1c, 12c6b45cecf9d53b, e0f441542fa53991, 69ed859929ee9042`

校验工具位于 `rust/colibri-cuda/tools/verify/`：
- `verify_baseline.py`：校验单步或首步 golden values；
- `outputs_match.py`：逐位比对多次运行的 8 个 hidden digest、43 层摘要链与 argmax；
- `attention_share.py`：汇总每 token 各相位耗时。

---

## 4. 目录导览

| 目录 | 说明 |
| --- | --- |
| `rust/colibri-cuda/` | DeepSeek-V4-Flash-0731 的 Rust + `cuda-oxide` 主引擎（模块清单见 `rust/colibri-cuda/MODULE_LAYOUT.md`） |
| `rust/colibri-cuda/tools/` | 启动脚本（`run_flash_cli.sh`、`run_flash_serve.sh`）、服务客户端与 A/B 验证工具 |
| `c/` | 上游 Colibri C/CUDA 引擎与 `openai_server.py`（OpenAI 兼容 HTTP 网关） |
| `tools/download_modelscope_0731.py` | ModelScope 官方版本权重断点续传与 SHA-256 校验下载器 |
| `docs/experiments/` | 从基线到 ~1.98 s/token 的完整实测实验记录与消融数据 |
| `web/` & `desktop/` | 本地聊天前端与桌面端外壳 |

---

## 5. 当前已知边界（后续演进入口）

1. **上下文窗口上限 128 token**：当前 `prompt + max_new_tokens - 1 <= 128`，超长输入会显式返回 `CONTEXT_WINDOW_EXCEEDED`。
2. **多轮对话 Prefix Caching 尚未接入**：当前服务面跨请求保留了固定权重显存池，但每次请求仍使用全新 KV 上下文；由于 MLA 压缩后 KV 体量很小，后续可按对话前缀缓存 KV 状态以消除多轮重算。
3. **存储路径约束**：权重目录必须位于 WSL 原生 ext4 块设备上，运行时会主动拒绝 `/mnt/c`、`/mnt/d` 等 Windows 挂载盘路径（9p 文件系统延迟会导致流式专家加载失速）。
