[English](README.md) | [简体中文](README.zh-CN.md)

# 想不想在你的游戏本上试试亲自全量部署1.6T的MoE大模型呢？

你的下一台LLM基础设施，何必是H100

| 全量检查点 | exact优化链 | 本机结果 |
| --- | --- | --- |
| 1.6T Flash-0731 FP4权重 | 稳态帧上派生6.50倍 | 16GB RTX 5060 Ti上0.505tok/s |

![sm_120稳态每token链路](docs/assets/readme/25-sm120-steady-chain.svg)

localMoE让单张消费级显卡跑起全量权重的DeepSeek-V4-Flash-0731检查点，把内存移动当成推理工作。模型放在WSL原生ext4卷上。每一层把固定tensor与6个routed专家带进一个有界工作集，CUDA用完才释放lease。

下面所有数字都来自同一台机器：16GB的RTX 5060 Ti，计算能力`sm_120`，WSL2 + CUDA 12.8。稳态每token从钉死基线的`12.873s`降到整模型常驻的`2.304s`，再降到routed fan-out成为默认之后的`1.98s`。这是`6.50倍`的降幅，由**在不同时间分别测量的三段**派生而来，不是单个匹配区块。按当前默认约`0.505tok/s`。

| 阶段 | 变化 | exact计时结果 | 怎么量的 |
| --- | --- | --- | --- |
| `C-D-D-C` | 整模型固定权重常驻显存 | 稳态每token`3.183s`降到`2.304s`，配对中位数`1.378倍`，`4/4`输出完全相同 | 一批4轮交错，同一二进制，只切一个开关 |
| `E` | routed专家改成六条流并行下发，不再排一条队 | 稳态每token`2.39s`降到`1.98s`，`−17%`，输出完全相同 | 两批匹配（默认先跑、关先跑），每臂3轮，同一二进制，只切一个开关；峰值`+64MiB`，服务面重复同样的8个id、每字`−0.464s` |
| `A` | 钉死的不常驻基线 | 稳态每token`12.873s`，整请求`103.381s` | 来自更早一棵树的锚点，不是当天同批的基线 |

有两对是真匹配区块——常驻那一对和fan-out那一对——所以它们的比值是量出来的。`6.50倍`这个头条数字把它们和更早树上的基线接在一起；每段各自都立得住，但端到端那个比值是派生的，不是在单次运行里观测到的。下面那张图画的是常驻那条链，所以它最后一根柱子是常驻默认，不是今天的默认。

精简的[运行台账](docs/assets/readme/sm120-steady-ab.csv)记录了每一次跑的值、峰值显存和对应的结果文件。常驻那批跑在`codex/business-restart`的`8e162a6`上；结果文件按惯例不带二进制指纹，所以那些数字是从`/srv/localmoe/targets/residentfill/`下的文件重新推出来的，不是仓库里的某个校验和。

![sm_120常驻显存的取舍](docs/assets/readme/26-sm120-residency-tradeoff.svg)

把整套固定权重钉在显存里要付`+6.35GB`——峰值从`1,825MB`涨到`8,175MB`——换来每个稳态token省`0.879s`。匹配批里8次跑产出的token逐字节相同，所以这是纯提速。16GB的卡上还剩约8GB空闲；再要让别的东西常驻，先看这个预算。

## 一个请求，一条经过检查的CUDA生命周期

Rust把请求的生命周期变成明确对象。请求拥有CUDA stream、event、pinned staging buffer、cache slot、已经检查的cuBLAS handle及其workspace。排队的GPU操作会持续持有lease。少量有状态CUDA对象因此能跨43层保持热态，流式权重仍然有界。

同一条规则现在也管权重本身。43层的固定tensor在第一次读到之后被`B2RawRuntime`捕获，之后每个token直接从显存取；命中时`fixed_source_bytes`与`fixed_h2d_bytes`都记0，所以台账上直接看得出省了多少。填缓存那一遍预取保持开，只在下一层已经常驻时才跳过，所以第一个token不比流式慢。

服务面遵循同样的规则。它跑的是和命令行同一条`run_v4_b2_generate_arm`循环，常驻的门槛覆盖`FLASH_BASE`和`FLASH_SERVE`两个面。

## sm_120上的exact Flash-0731执行

exact路径钉住了控制路由和状态的数值工作：FP8 MMA累积、有序归约、RMSNorm行为、cuBLAS compressor调用、KV更新以及final head。在这台机器上，对token ID`0`走一步greedy，每次改动之后都复现同一组值：

```
argmax                 55222
hidden_bf16_digest     5ed8bf88beae3cfe
layers                 43
fp4_accum              block32
final_head             canonical_cpu_streamed_bf16
```

8步请求复现下面这条digest链，项目自己的检查脚本会逐个token比对：

```
5ed8bf88beae3cfe, 316e5495ca424b39, a521547b2586e458, 1963dfeaaf30264c,
b970da7913a57a1c, 12c6b45cecf9d53b, e0f441542fa53991, 69ed859929ee9042
```

上面表里的每一项优化，都是先让这8个digest、8条43层摘要链和8个argmax与上一阶段逐位相同，才算数。

有一条要如实说清：`sm_120`上官方TileLang FP4核不可复现。同一份权重和输入，两次跑的`routed_sum_f32`有`510/4096`个元素不同（其中`440`个相对误差超1%，最大`30.7倍`）；一个`M=1, N=2048, K=4096`的最小复现例报`2048/2048`个元素错误，跑间差`6.912e+01`。猜的原因是TileLang 0.1.8发出的是`sm_100`代码，而消费级`sm_120`缺它面向的那条tensor core指令——这个猜测没有验证。所以上面那组钉死值是**本引擎自己的回归门**，不是"在`sm_120`上对上了官方reference"的证书。把官方oracle钉到稳定的TileLang版本之后两边的token都到`55222`，但两个实现仍在第8层的`route_weight_bits`首次分叉，谁对谁错都还没判。[归档的sm_89 README](docs/archive/README.sm89-12gb-rtx4080.md)里那张`43/43`对上reference的证书是在另一张卡上测的，不能当成这里也过了。

## DSpark提交经过验证的前缀

DSpark先提出一个block，再算base target，随后提交被接受的前缀。状态transaction在提交前携带base和3个MTP状态digest。

`sm_120`上第一次真跑DSpark，整条路都执行到了——权重、状态transaction、三个MTP阶段——结果是`30`个proposal里接受`0`个，输出与base greedy逐位相同，但比直接解码慢大约`16倍`。排查抓到一个真bug：`rope_v4_bf16`这个核超过4行就静默返回，而DSpark的draft是5行，所以`q`和`query_kv`一直被清零。把上限抬高之后`q_nz`从`0`变成`163840`，可是接受率仍是`0`，说明还有第二处缺陷。现在的卡点是修完那条之后`confidence`的幅值从约`1`跳到`10^3..10^4`，指向`collapsed`的量级。

独立的官方DSpark oracle在这里跑不起来：它的生成脚本要合并过的`model{i}-mp{mp}.safetensors`分片，而本地是48分片的HF布局，166GB本来也塞不进16GB的卡。草稿没跑对之前，谈"一趟验证N行"还早。

## 现在一个token花在哪

稳态、当前默认，从常驻那批的相位账量出来。每一项是什么、来历如何：

| # | 项 | 每token | 里面是什么 |
| --- | --- | --- | --- |
| 1 | 专家路径：`routed`行，六条lane互相重叠 | `0.58s` | 每条lane自己读、自己上传、自己在自己的流上跑专家，一层的收尾是最慢那条lane。读盘仍贴着盘上限；原来那趟restage拷贝已经没了。**每条lane的`setup`/`h2d`/`kernel`是重叠后的和值，不能相加** |
| 2 | `attention` | `0.58s` | compressor `0.20s`与indexer `0.08s`仍是**主机**实现；16个小步已搬到设备侧 |
| 3 | `shared` | `0.46s` | 现在只剩MLP自己。六条lane把它顶起来约`9.9ms`一层：上传约`4.5ms`（抢PCIe）+ 内核约`5.4ms`（抢SM）；但最慢那条lane仍比它长，所以它**不卡人** |
| 4 | `router` | `0.13s` | |
| 5 | `final head`（层循环外） | `0.10s` | 每token 1.01GiB权重，从1个线程改成6个 |
| 6 | `fixed` | `0` | 默认常驻；设`COLIBRI_V4_FIXED_RESIDENT_GIB=0`回到约`0.81s` |

稳态的墙是`1.98s`。表里每行是token 1–7的相位均值，且每个token的第一层被排除，所以各行相加会略小于墙。

这个账是"搬字节"的账，不是算力的账：attention里18个GEMM只占`0.2–0.3%`。能走的只有两个方向——少搬，或者和别人重叠。**地板取两边的较大值，不是相加**：每个token必须从盘上搬的3.45GB按实测`3.66GB/s`值`0.94s`，而非读侧（shared MLP + attention + router + 专家kernel）约`1.4s`，所以地板约`1.3–1.4s`每个token；而且实测已经到过它附近：fan-out默认在真prompt上量到`1.549s`，把lane内核摘掉的消融臂量到`1.352s`。

这张表有两点要讲清楚，否则会误导。`shared`那一行以前是一个套住两件事的墙——shared MLP在主线程跑，同时6个worker在读routed专家——那`0.69s`里约四分之三属于专家路径；fan-out成为默认之后那些读都在lane里面了，所以这一行是MLP自己扣掉抢PCIe/抢SM的代价之后的样子。另外`final head`以前被记成`~0.13s`，那是只读了它的盘读部分；整步的墙是`outer.wall_ns`那个数，它是一块真实的第四大项。

## 从一张16GB的卡出发

现在卡就是预算。因为6.71GB权重住进了显存，`fixed`归零，峰值`8.24GB`，还剩约8GB。下表是实测的稳态帧和剩下的余量，不是外推。

| 配置 | 稳态每token | 峰值显存 | 16GB卡上剩余 |
| --- | ---: | ---: | ---: |
| 当前默认（常驻 + fan-out 都开） | `1.98s` | `8,242MB` | `~8GB` |
| fan-out关（`COLIBRI_V4_FLASH_ROUTED_FANOUT=0`） | `2.39s` | `8,175MB` | `~8GB` |
| 常驻关（`COLIBRI_V4_FIXED_RESIDENT_GIB=0`） | `3.183s` | `1,825MB` | `~14GB` |

前两行来自2026-09-30的同一批（单token prompt，两个顺序各跑一遍）；第三行来自常驻那批，所以三个绝对值不是同一个区块。

还剩两个已知的天花板。输入上限是128个token（`prompt + 生成 − 1`），超过121个token的prompt直接被拒（`CONTEXT_WINDOW_EXCEEDED`），首字还按每个prompt token约`2.6s`涨。另外服务面只在显式开`COLIBRI_V4_FLASH_SERVE_PHASE_JSONL`时才写相位账，所以它的逐相位拆分目前只有**一个prompt**的实测，不是普遍结论——不过两个面在稳态上已经量到一样快。

## 运行

Flash路径面向WSL2/Linux、CUDA 12.8、Blackwell`sm_120`，模型目录必须在WSL原生ext4块设备上。运行时会拒绝`/mnt/c`和`/mnt/d`下的模型路径。

```bash
git clone https://github.com/andyzpb/localMoE.git
cd localMoE/rust

export COLIBRI_DSV4_MODEL_DIR=/mnt/colibri-native/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb
export COLIBRI_CUBLAS_EXACT_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

cargo oxide doctor
./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda
```

在`sm_120`上跑真实的Flash base路径。`FP4_ACCUM_ARM`和`FINAL_HEAD_ARM`故意不设，用来确认默认臂仍是`block32`与`canonical_cpu_streamed_bf16`。

```bash
env \
  COLIBRI_V4_FLASH_BASE=1 \
  COLIBRI_V4_FLASH_TOKEN_IDS=0 \
  COLIBRI_V4_GREEDY_STEPS=8 \
  COLIBRI_V4_STREAM_READ_MODE=buffered \
  COLIBRI_V4_FLASH_PREFETCH=1 \
  COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas \
  COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas \
  COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice \
  COLIBRI_V4_RESULT_JSONL="$PWD/flash-result.jsonl" \
  "$PWD/target/release/colibri-cuda"
```

检查`flash-result.jsonl`里的`layers: 43`、`hidden_bf16_digest: 5ed8bf88beae3cfe`，以及`generated_ids`开头是`55222, 33832, ...`。`status: PASS`只表示43层跑完，不表示任何oracle gate通过。整模型常驻显存默认开；设`COLIBRI_V4_FIXED_RESIDENT_GIB=0`回到流式，峰值回到`1,825MB`。这些数字背后的结果文件在仓库之外，放在`/srv/localmoe/targets/`，所以计时数字无法只从受版本控制的文件重新推导。模型使用遵循检查点许可证，localMoE代码采用[Apache License 2.0](LICENSE)。

## 灵感

localMoE受[JustVugg/colibri](https://github.com/JustVugg/colibri)启发，并继续推进家用硬件上的可用全量权重MoE推理。12GB RTX 4080 Laptop上那套`sm_89`测量保存在[docs/archive](docs/archive/README.sm89-12gb-rtx4080.zh-CN.md)。
