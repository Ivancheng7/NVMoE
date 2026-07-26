[English](README.md) | [简体中文](README.zh-CN.md)

# 想不想在你的游戏本上试试亲自全量部署1.6T的MoE大模型呢？

你的下一台LLM基础设施，何必是H100

| **1.6T参数** | **12GB显存** | **快2.228倍** |
| --- | --- | --- |
| 完整V4-Pro检查点留在本地 | GPU工作集保持有界 | exact完整请求开发结果 |

![exact完整请求加速](docs/assets/readme/19-full-request-speedup.svg)

目标是让通常需要数据中心硬件的模型能够在个人电脑上推理。本地SSD可以保存完整检查点。每个token都需要让当前路由选中的专家权重按正确顺序穿过很小的工作集。localMoE保持模型算术并优化这条数据流。

## 核心思路：保持模型算术并优化数据流

MoE为每个token稀疏激活一部分参数，完整专家集合在整个请求中保持可寻址。V4-Pro让问题更严苛：每个生成token串行经过61层，每层都有固定工作，router决定下一步要传入哪些专家字节。

localMoE把它视为带语义约束的放置问题。一次请求经过一条有界路径：

`SSD -> RAM decode -> pinned host staging -> GPU lease -> canonical reduction -> transactional state commit`

![有界的SSD、RAM与VRAM执行路径](docs/assets/readme/02-execution-path.svg)

运行时在GPU处理当前payload时准备下一份payload。layer-stationary状态为每个模型层保留持久状态，为每个请求保留一份scratch空间。cache lease在worker持有slot期间保护该slot，某层成功完成后才会提交持久状态。引擎由此减少I/O、decode、H2D流量和空转时间。路由选择、归约顺序和可观察状态继续遵循规范约束。

![检查点容量与笔记本内存层级](docs/assets/readme/01-capacity.svg)

| 已记录的V4-Pro压力测试 | 数值 |
| --- | ---: |
| GPU | NVIDIA GeForce RTX4080Laptop，`12,282MiB`显存 |
| 主机内存 | `32GB` RAM |
| 检查点 | 64个固定分片，共`864,721,029,744B` |
| 已执行路径 | 完整61层Rust规范机制路径 |

## 当前证据

README生成器为同一P5/N3配置人工冻结了两项`n=1`完整请求计时。两项计时相除得到页面展示的`2.228x`开发比较。记录的元数据列出同一固定V4-Pro权重、prompt IDs、贪心输出、61层调度和内部状态审计约束。

| exact P5/N3完整请求 | 墙钟时间 | 输出速率 | 变化 |
| --- | ---: | ---: | ---: |
| exact基线 | `198.95s` | `0.0151tok/s` | `1.00x` |
| 当前exact引擎 | `89.28s` | `0.0336tok/s` | **`2.228x`** |

冻结计时输入对应完整请求延迟降低`55.13%`。每个请求输出3个token。独立的full61账本审计把记录工作负载的路由源读取与路由H2D从`64.223GB`降到`29.620GB`，减少`53.9%`。

该比较的公开artifact是冻结的[README生成器](docs/assets/readme/generate_cost_figures.py)与[benchmark ledger](docs/assets/readme/cost-figures-provenance.json)记录。两项计时的source identity不同，因此结果属于描述性开发证据。它验证记录的Rust规范路由、归约和事务式状态边界。持续聊天服务吞吐量需要独立基准，官方PyTorch full61一致性也有独立验收门槛。

## 通向家用推理的路径

V4-Pro是一个刻意严苛的存储与状态压力测试。它暴露固定payload、稀疏专家并集、持久状态流量和输出头穿过小型内存层级的成本。同一调度模型可以在更小的MoE上继续评估，那些项会更适合家用硬件。

DeepSeek-V4-Flash是第一个计划目标。下列Flash区间是README生成器与benchmark ledger中冻结的说明性预测输入。当前仓库的Flash实跑次数为0，实测性能仍待获得。

![预测的Flash家用硬件范围](docs/assets/readme/20-flash-projection-envelope.svg)

| 未来V4-Flash配置 | 预测提交速率 |
| --- | ---: |
| 冷H1，12GiB | `0.103至0.125tok/s` |
| 固定权重常驻H1，16GiB | `0.313至0.658tok/s` |
| 固定权重常驻H2，24GiB，`q=0.8` | `0.548至1.153tok/s` |
| 固定权重常驻H5，32至48GiB，`q=0.8` | `0.500至1.052tok/s` |

这些区间是说明性规划输入。未来测量需要确定固定常驻、草稿成本、回滚、实际专家并集、调度跨度和接受概率`q`。每个模型家族还需要各自的metadata、路由语义、KV layout和验收门槛。

## 为什么用Rust

Rust让运行时约束在代码中保持可见。move-only lease表达cache所有权，带类型FP4-FP8 layout生成字节预算，有界worker和CUDA stream让I/O与GPU工作重叠，有序归约保持规范专家顺序，transactional commit保护持久状态。GEMM速度仍然取决于kernel和硬件。

## 快速开始

当前路径面向WSL2/Linux、CUDA12.x、LLVM/NVPTX、libclang和Ada `sm_89`GPU。模型权重必须存放在WSL原生ext4块设备。环境约束会拒绝将`/mnt/c`和`/mnt/d`用于模型存储。

```bash
git clone https://github.com/andyzpb/localMoE.git
cd localMoE/rust
cargo oxide doctor
cargo fmt --check
cargo check --locked -p colibri-cuda
./run-identified.sh test --arch sm_89
```

## 许可证与灵感来源

代码采用[Apache License 2.0](LICENSE)。模型权重及相关材料遵循各自发布条款。保留的上游代码和素材继续遵循原有署名与许可条款。

localMoE受[JustVugg/colibri](https://github.com/JustVugg/colibri)启发，探索exact流式MoE执行在现有家用硬件上能够走到多远。
