[English](README.md) | [简体中文](README.zh-CN.md)

# localMoE：想不想在你的游戏本上亲自试试部署全量1400B级MoE？

localMoE在配备12GB显存的RTX 4080 Laptop GPU上运行完整DeepSeek-V4-Pro检查点。固定模型为1.6T参数MoE，使用官方混合FP4-FP8权重，模型文件约864.7GB。64个分片留在本地SSD，Rust运行时把有界工作集在RAM和VRAM之间移动。

当前P6开发轨迹走完全部61层，在`14.789027100 s/token`内生成token `115287`，即`0.0676177 token/s`。full61审计在P4至P6返回`PASS`，认证GPU argmax没有触发CPU回退。这是一条开发轨迹，`n=1`，没有置信区间，缓存状态与major fault条件没有控制。

![三个同一计时估计量下的P6 full61开发吞吐结果](docs/assets/readme/18-p6-throughput.svg)

图中每个条件各有一条P6开发轨迹，使用同一个计时估计量：full61下一个token墙钟时间。按这一描述性比较，combined path比pre-pipeline轨迹快`1.1605x`。它不能建立因果效应、持续吞吐或上游C++对比。

## 当前可运行范围

| 记录配置 | 数值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 Laptop，`12,282 MiB`显存 |
| 主机内存 | `32 GB`RAM |
| 检查点 | 64个固定分片，模型文件共`864,721,029,744 B` |
| 模型路径 | 完整61层DeepSeek-V4-Pro Rust执行 |
| 当前P6轨迹 | `14.789027100 s/token`，token `115287`，full61审计P4-P6 `PASS` |

实现执行记录提示词和贪心token步所需的精确、窄范围全模型路径。通用聊天、采样、长上下文、持续热稳定、服务行为和官方参考一致性仍待评估。

## Rust路径的实际机制

运行时把数据移动关系直接写进代码。

- 显式资源所有权让设备与主机缓冲区随请求生命周期持有。
- pinned双缓冲worker在当前层计算时准备固定层输入。
- metadata驱动的有界readahead为下一层发出建议读取；建议路径无法运行时会记录skip。
- 认证GPU argmax在区间足够确定时接受证书，其他情况使用规范CPU路径。
- Colibri自有CUDA kernel使用`cuda-oxide`实现。

这些机制让运行时在不放松记录执行顺序和容量检查的条件下放置远大于笔记本内存的检查点。

## P6基准

下表三项均使用同一个P6 full61下一个token计时估计量。每个条件`n=1`，仅作描述，没有CI。

| P6条件 | 每token秒数 | 每秒token数 |
| --- | ---: | ---: |
| Pre-pipeline | `17.162779092` | `0.0582668` |
| Pinned pipeline | `15.852466305` | `0.0630817` |
| Combined path | `14.789027100` | `0.0676177` |

## 未来目标

DeepSeek-V4-Flash支持和家用平台1400B级部署目标属于未来工作，本README没有对应实测结果。每个新模型家族都需要自己的metadata、容量账本、语义检查和基准证据。

## 复现当前路径

环境需要WSL2/Linux、CUDA 12.x、LLVM/NVPTX、libclang和Ada `sm_89`GPU。检查点必须存放在WSL原生ext4文件系统。环境约束会拒绝把`/mnt/c`和`/mnt/d`用于模型存储。按照[full61环境约束](rust/colibri-cuda/tools/V4_FULL61_ORACLE.md)设置`COLIBRI_DSV4_MODEL_DIR`、`COLIBRI_ORACLE_PYTHON`和`COLIBRI_CUBLAS_EXACT_LIB`。

```bash
git clone https://github.com/andyzpb/localMoE.git
cd localMoE/rust
cargo oxide doctor
cargo fmt --check
cargo check --locked -p colibri-cuda
./run-identified.sh test --arch sm_89
```

[机制实验记录](docs/experiments/v4-pro-mechanisms-2026-07-24.md)包含精确模型运行命令和故障关闭预检。仓库也提供用于复核的紧凑[证据封装](docs/assets/readme/data/evidence-v1.json)。

## 仓库结构

```text
rust/colibri-cuda/          Rust + cuda-oxide V4-Pro prototype
rust/colibri-cuda/tools/    oracle, trace, and evidence tools
docs/assets/readme/         public figures and compact evidence artifacts
c/                          retained upstream C/CUDA oracle and baseline
```

## 许可证

代码继续采用[Apache License 2.0](LICENSE)。模型权重及相关材料遵循模型发布页面上的条款。保留的上游代码和素材继续遵循原有署名与许可条款。

## 灵感来源

Inspired by [JustVugg/colibri](https://github.com/JustVugg/colibri) and pushing further.
