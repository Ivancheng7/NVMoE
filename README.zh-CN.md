[English](README.md) | [简体中文](README.zh-CN.md)

# 想不想在你的游戏本上试试亲自全量部署1.6T的MoE大模型呢？

你的下一台LLM基础设施，何必是H100

| **1.6T参数** | **12GB显存** | **快3.58倍** |
| --- | --- | --- |
| 完整810GiB模型目录 | exact有界GPU工作集 | 198.948s降至55.550s |

![exact完整请求加速](docs/assets/readme/19-full-request-speedup.svg)

localMoE从WSL原生ext4卷运行完整DeepSeek-V4-Pro检查点，只在RAM和VRAM中保留有界工作集。在实测游戏本上，已验收的P5/N3候选完成完整请求的几何中心为`55.550s`，记录的exact基线为`198.948s`。跨来源谱系的描述性加速为`3.58倍`，墙钟时间降低`72.1%`。

最终候选通过密封ABBA门槛，每个配置`n=2`。所有arm输出相同的3个token ID和相同的规范化BF16摘要序列。

## 12GB显存面对的1.6T难题

MoE为每个token稀疏激活一部分专家。V4-Pro仍需串行经过61层，每层都包含固定工作和router选中的专家payload。完整专家集合始终保持可寻址，GPU只能容纳其中很小的工作集。

localMoE把这个约束变成分层执行路径：

`ext4 SSD -> CPU read and decode -> pinned RAM -> retained VRAM -> canonical GPU reduction -> transactional state commit`

![有界的SSD、RAM与VRAM执行路径](docs/assets/readme/02-execution-path.svg)

GPU消费当前层时，运行时同步准备后续payload。immutable expert cache让每个prefill专家并集只加载一次，host retention让更多固定payload绕开存储读取，device retention继续削减H2D路径。整个请求持续遵循规范路由顺序和状态转移。

![检查点容量与游戏本内存层级](docs/assets/readme/01-capacity.svg)

## 密封实验支持的结论

3个ABBA门槛都使用P5/N3 full61 exact执行。每个配置在冻结的`A1, B1, B2, A2`顺序中都有2个完整arm，所有完整arm都进入对称估计。

| 密封机制门槛 | 候选变化 | 对称E2E比值 | 注册计时门槛 |
| --- | --- | ---: | ---: |
| immutable expert cache | 从关闭cache到32个slot | `0.7007` | TTFT `0.7901`，hot total `0.6328` |
| host retention | `12GiB`到`13GiB` | `0.8946` | TTFT `0.8862`，hot total `0.9039` |
| device retention | `3.5GiB`到`4.0GiB` | `0.9260` | TTFT `0.9268`，hot total `0.9248` |

cache门槛把物理专家加载次数从`2,562`降至`1,576`，路由存储读取与路由H2D流量分别减少`34.603GB`。host retention提升至`13GiB`后，固定权重源读取再减少`2.563GB`。device retention提升至`4.0GiB`后，固定权重源读取减少`1.709GB`，固定H2D流量减少`1.568GB`。

密封N6 device-retention overlay把实测每token时间从`17.8409426951s`降至`16.5992631988s`，变化为`-1.2416794963s/token`（`6.96%`）。每token减少`1,708,974,176B`权重源读取和`783,919,152B`H2D流量，同时保持引擎声明的exact语义。该估计中每个配置`n=2`。配对漂移把结果范围限定在已记录的P5/N3设置，不能据此外推通用吞吐。

## 实测游戏本上的预期时间

P5/N3表示prompt中的5个token和随后输出的3个token。最终验收配置使用`13GiB`host retention和`4.0GiB`device retention。

| P5/N3测量项 | 几何中心 | 推导速率 |
| --- | ---: | ---: |
| TTFT | `27.631s` | 首个输出token |
| 2个decode interval | 合计`27.713s` | `0.0722tok/s` |
| 完整请求E2E | `55.550s` | `0.0540 emitted tok/s` |

decode速率只来自2个实测interval，证据范围限于短序列。持续生成仍需更长的基准测试。E2E数值覆盖完整记录请求，包括TTFT和后续2个decode interval。

实测起始配置为：

| 资源 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU，`12,282MiB`显存 |
| 主机 | `32GB`RAM |
| WSL2 | 约`24GiB`可见内存 |
| retention | `13GiB`主机预算和`4.0GiB`设备预算 |
| 模型存储 | WSL原生ext4，模型目录至少需要`810GiB` |

这组配置提供当前实测参考点，通用最低配置仍需独立验证。还应为构建产物和实验结果预留额外磁盘空间。

## 此处exact的含义

`exact`指Rust引擎声明的规范约束。所有验收通过的arm在p4、p5和p6输出token `115287`，规范化BF16摘要依次为`519e76cfedc01504`、`7017c61decbc4151`和`3bed9be0e5ebd27c`。路由顺序、归约、KV更新、lease释放和transactional state commit也通过对应的记录门槛。

官方PyTorch完整模型一致性、长上下文行为、多prompt文本质量、sampling和持续吞吐仍有各自的验收目标。头条`3.58倍`来自跨来源谱系比较，证据性质为描述性结果。上方3个ABBA比值承担配对机制主张。

## 为什么使用Rust

Rust把运行时约束落实在语言边界。move-only lease让cache slot持续拥有明确owner，直到GPU工作释放资源。RAII在所有退出路径关闭pinned memory、CUDA event和device buffer。带类型的FP4与FP8 layout让字节预算保持显式。有界channel与CUDA stream重叠存储读取、CPU decode、H2D和GPU执行，同时保持规范归约顺序。

资源调度器的所有权规则因此可以直接在代码中审计，GEMM性能继续由CUDA kernel和GPU决定。

## 快速开始

当前路径面向WSL2/Linux、CUDA 12.x、LLVM/NVPTX、libclang和Ada `sm_89`GPU。模型权重必须存放在WSL原生ext4块设备。环境约束会拒绝将`/mnt/c`和`/mnt/d`用于模型存储。

```bash
git clone https://github.com/andyzpb/localMoE.git
cd localMoE/rust
cargo oxide doctor
cargo fmt --check
cargo check --locked -p colibri-cuda
./run-identified.sh test --arch sm_89
```

代码采用[Apache License 2.0](LICENSE)。模型权重遵循各自的发布条款。

## 灵感

localMoE受[JustVugg/colibri](https://github.com/JustVugg/colibri)启发，并沿着这条路线继续推进家用硬件上的exact全量MoE推理。
