---
name: flash-0731-official-oracle-unstable
description: 官方 FP4 核在 sm_120 上算错，根因是钉死的 tilelang 0.1.8；换 0.1.14 已实测修好（torch 无关）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_e49d8c7c-e0c8-42dd-b0eb-31745dc99281
---

在 `sm_120`（RTX 5060 Ti）上，官方 TileLang FP4 核**不是稳定 oracle**：已保存的官方 full-43 运行分别给出 token `67900`（v4）、`55222`（v5）和 token `0`（r9），对应的 hidden SHA-256 也各不相同；官方 L0 routed FP4 的中间量重复运行同样漂移。

因此不能把单次 `67900`，也不能把某个随机中间快照，当成唯一数值真值。当前可复现的业务基线是 Rust 的 `block32` 路径——同环境连续两次完整运行都得到 token `55222`、hidden digest `5ed8bf88beae3cfe`。在有稳定 FP4 专家 oracle 之前，不做 routed FP4 的逐元素精确声明。

**Why:** 这条决定了"什么能算通过"的口径。把不稳定的官方核当真值，会得到时好时坏的假失败，进而去改本来正确的代码。

**How to apply:** 报 `sm_120` 结果时只报告 runtime GPU exact 实际输出，并写明未声称 FP4 逐元素精确；不要为了让官方比对过而调整断言。来源：`docs/experiments/2026-08-28-sm120-flash-0731-adaptation.md`。

另外：README 上那份招牌证书（43/43 层全匹配、argmax `33832`、三个 digest）是 **sm_89**（12GB RTX 4080 Laptop）上的封存结果，不是当前 sm_120 环境的结果，逐位相等本来也只对那台机器 + 那个 binary + 那套 cuBLAS 成立。2026-09-16 已把这点写进 `README.md` / `README.zh-CN.md`，并注明 sm_120 上可复现的是 token `55222`。顺带：结果 JSONL 里的 `status: PASS` 只表示 43 层跑完了，**不表示 oracle 门通过**。

同一天还发现 README 原来有一条无 artifact 支撑的声明（DSpark 账本 SHA-256 `34734f…e553`，仓库里没有任何匹配文件），已删除；`docs/assets/readme/flash-0731-abba.csv` 里的数字仍是手写死在 `generate_flash_0731_figures.py` 的 `ROWS` 表里的，`--check` 只能验证 SVG 与 CSV 互相对得上，验证不了数字本身。

**2026-09-16 口径修正：核是坏的，不是求和顺序。** 本会话之前有一条结论说"最大相对差 `2.6e-4`、漂移是浮点求和顺序、不是核坏了"（依据是 `0731-official-r7.jsonl` 与 `r8.jsonl` 两次前缀运行，37 个字段）。**这条已被推翻，不要再照它走。** 两层证据都在 `docs/experiments/2026-09-16-sm120-fp4-oracle-diagnosis.md`：

1. **模型级**：抓 layer 0 的 23 个 cutpoint 张量比两次运行，**21 个逐位相同**——含整条注意力路径、`route_ids`、`route_weights`、共享专家。唯一不同的是 `routed_sum_f32`（routed 专家的 FP32 累加器）：4096 个元素里 510 个不同，其中 440 个相对误差超过 `1e-2`，**最大 30.7 倍**。纯归约顺序只会差到 `1e-7` 量级。
2. **核级（决定性，且不依赖任何参考）**：`tools/repro_fp4_gemm_sm120.py` 把问题缩到一次 `kernel.fp4_gemm` 调用。**同一输入连续四次，四次互相差到 `49.19`**，而该结果整体幅值只有 `13.0`；四次的 2048 个输出元素全部错，误差达参考幅值的 4.6–5.4 倍。对照实验把权重每字节设成 `0x22`（两个半字节相同，字节内顺序无关）、缩放设成精确 `1.0`，并断言了这两个解码，排除参考实现这一侧的嫌疑。

**为什么之前会得出相反结论**：r7/r8 那对比的是 37 个字段，而 `routed_sum_f32` **不在那批字段里**，所以恰好越过了分叉点。1e-4 量级的浮点顺序漂移确实存在，但它不是这个 bug；两者是两件事。

**How to apply：** 报 `sm_120` 结果时只报 runtime GPU exact 的实际输出，写明未声称 routed FP4 逐元素精确。**不要**为了让官方比对过而调断言或加容差了事——官方核在这张卡上每次给不同的、且幅值级错误的结果，容差没有意义。来源：`docs/experiments/2026-09-16-sm120-fp4-oracle-diagnosis.md`。

仍然成立、但这轮没动的两点：(a) token 是 argmax，是不连续函数，小扰动也会翻 token；(b) 官方 oracle 缺 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 与 `torch.set_float32_matmul_precision("highest")` 两个确定性开关（旧仓库 `localmoe` 的脚本有），而且 `torch.use_deterministic_algorithms` 管不到 TileLang 生成的 kernel。这两条在这个 bug 面前都救不了场，但将来若要用容差口径比对平滑中间量还会用到。

**2026-09-16 继续查：把"为什么"缩到了两个可以证伪的点上。**

- **不是 arch 选错。** tilelang 0.1.8 在这张卡上 `determine_target("auto")` 自己给出
  `-arch=sm_120a`。`docs/experiments/2026-09-16-sm120-fp4-oracle-diagnosis.md` 里
  "合理猜测是按 sm_100 生成代码"那条**已被实测证伪**，别再把 sm_100 codegen 当成原因。
- **不是 FP4 张量核缺失。** 官方 `fp4_gemm` 根本不用 FP4 张量核：它把 FP4 经 fp32 cast
  成 FP8，再做 FP8×FP8 的 `tl::mma_sync<kFloat8_e4m3, kFloat8_e4m3, kFloat32, 16, 8, 32>`
  （即 `mma.sync m16n8k32`）——这是 sm_120 上合法且正确的指令。所以"sm_120 没有 tcgen05/
  原生 FP4 所以算错"这条推理**不成立**，我们这条链没碰 FP4 张量核。
- **坏在搬运/同步，不在算术。** 对照实验里整块权重恒定（全 `0x22`）、scale 精确 1.0，
  任何"读错位置"都读不出不同值，结果**仍然每次都不同**。所以坏的是激活侧共享内存、
  累加器或输出落盘，与权重解码无关。
- **上游有一模一样的 bug 类。** tile-ai/tilelang#2699：sm_120 上 ThreadSync 没把
  `st.bulk` 记成共享内存写、漏插 `__syncthreads`，症状"能编能跑、静默错误且**非确定**"
  （他们复现是 20/20 次与参考不同），**0.1.12 上仍存在**，PR#2700 才修；
  `racecheck` 查不出。**注意这条是同类不是同一个**——不同 kernel，我们没验证过它就是我们
  那个缺的 barrier，只验证到"症状签名完全一致"。
- 官方 `requirements.txt` 钉死 `tilelang==0.1.8`。上游 0.1.13（2026-08-03）才加
  `gemm_mma_sm120` / `GemmMMASm120BlockScaled`（`cuda.mma.blockscaled`，warp 级
  block-scaled MMA）并带"a broad set of correctness fixes"。

**2026-09-16 定案：换掉 tilelang 就好了，而且只需要换这一个。** 已实测（三组对照，
同一个 `tools/repro_fp4_gemm_sm120.py`、同一台卡）：

| 组合 | 结果 |
| --- | --- |
| tilelang 0.1.8 + torch 2.10.0+cu128（官方原样） | 全错，四次互不相同 |
| tilelang 0.1.14 + torch 2.10.0+cu128 | 四次**逐位相同**，误差降到参考幅值的 0.2% |
| tilelang 0.1.14 + torch 2.14.0+cu130 | 逐位稳定，**数字与上一行完全相同** |

**torch 换不换对结果毫无影响，唯一变量是 tilelang。** 两个专家几何都验了
（N=2048/K=4096 与 N=4096/K=2048），都是 `run-to-run max_abs = 0.000e+00`。
残留的 `n(rel>1e-3)` 只是 FP8 舍入对上 fp32 参考的正常差距（max_abs 0.5 对幅值 239），
不是缺陷。所以之前"sm_120 上没有可用官方数值真值"的结论**只对 tilelang 0.1.8 成立**。

**怎么跑（不碰官方环境）：** 官方 `flash-reference-py312` 是钉死的 venv，**不要动**——
sm_89 证书和 full-43 参考都靠它。用 `pip install --target` 装到旁路目录再靠 `PYTHONPATH`
遮住它，torch 仍用官方那份：

```
P=/srv/localmoe/toolchains/flash-reference-py312/bin/python
$P -m pip install --target /srv/localmoe/toolchains/tl014-only --no-deps tilelang==0.1.14
$P -m pip install --target /srv/localmoe/toolchains/tl014-only --no-deps --upgrade "apache-tvm-ffi>=0.1.11,<0.1.13"
PYTHONPATH=/srv/localmoe/toolchains/tl014-only \
CPATH=/srv/localmoe/compat-headers:/srv/localmoe/toolchains/glibc243-cuda13-math-overlay-v1/include \
$P /tmp/repro.py --inference-dir /srv/localmoe/sources/deepseek-v4-flash-git/inference --runs 4
```

（`/srv/localmoe/toolchains/tl014-only` 现成可用，180MB。）注意 `apache-tvm-ffi` 必须一起升
——0.1.14 要求 `>=0.1.11,<0.1.13`，官方环境里是 `0.1.8.post2`。CPATH 那两行不是可选项，
少了会因 glibc 2.43 与 CUDA 13.1 的 `rsqrt` 声明冲突而编译失败；warm 的
`/root/.tilelang/cache` 会掩盖这个问题（换了 tilelang 版本缓存键就变了，必然重编）。

**注意**：换 tilelang 会改变参考值本身，sm_89 那份 43/43 证书是 0.1.8 时代的产物，
所以这是**并存第二套环境**，不是替换。

**2026-09-16 收官：full-43 已在 sm_120 上 5/5 逐位可复现。** 五趟
`--end-layer 42 --final-head` 全部 token `55222`、`final_hidden` fnv1a64 `23839e7574811bde`、
`logits` fnv1a64 `9ad29ad11f5a1c67`；用
`tools/compare_official_run_repeatability.py` 互比，**623 个字段全一致**（两两 identical）。
对照：旧版 r9/r10 的**第 0 层** `hidden_bf16` 就分叉——所以之前"sm_120 上不存在可用官方数值
真值"这句话，**只对 tilelang 0.1.8 成立**。单跑 30~80 秒（含 167GB 权重 mmap 流式加载）。

注意 token 相同**不能**当证据：r10 也是 `55222`，但和 A 的第 0 层就不同。**必须比整条记录。**

**2026-09-16 按用户要求把版本闸门整个删掉了：** 已提交 `2ea35d6`（分支
`codex/business-restart`），WSL 克隆已 `git pull --ff-only` 到该提交。
`v4_flash_0731_prefix_oracle.py` 和 `v4_full61_oracle.py` 原来都硬判
`tilelang == "0.1.8"`，而且前者还把版本**写死成字符串** `"0.1.8"` 记进 resources（一旦
绕过记录就会撒谎）。现在**两个文件都不再要求任何 tilelang 版本**，只 `import tilelang`
确认环境能用；resources 里改成写**真实版本**（如 `"tilelang": "0.1.14"`），
`tilelang_pin` / `tilelang_override` 两个字段已删除。删闸门也符合 AGENTS.md 的
"这条链没跑通之前不加门禁"。

跑 full-43 用 `/srv/localmoe/targets/run_tl014_full43.sh <tag>`（只需 `PYTHONPATH` 指到
`tl014-only` 和 CPATH 两样），**从克隆里跑**。六趟（a~f）互比全部 identical。

**还没做**：把 Rust 实现跟这份新的 sm_120 参考逐层对照。Rust block32 基线也出 token `55222`，
但按上面那条，token 相等不算数——要比逐层字段。

**台账怎么变成可重生的（2026-09-16 查清卡点）：** 每行需要四样——`request_e2e_s` 耗时、`exact_gate`、`binary_sha256`、`max_rss_kib` 峰值 host 内存。前三个今天的运行自己就产出（耗时与 gate 在 result JSONL 里，二进制 hash 直接算），**只有峰值 host 内存缺**：Flash base 的 result JSON 写了 `peak_device_used_bytes`，但峰值 host RSS（`/proc/self/status` 的 `VmHWM`）只在几个 benchmark 模式里记（`main.rs` 约 21992 行附近），base 路径不写。所以修法分两半——先在 base result JSON 里补上峰值 host RSS（约三行，在 `main.rs` 里，得等拆分完），再给 `generate_flash_0731_figures.py` 加 `--from-runs <目录>` 模式，从真实 result 文件建表并当场对 gate。

现有那八个数字**补不出真实出处**：它们是在 sm_89 笔记本上跑的，原始 trace 不在仓库里。硬编一个出处等于往台账里塞假数据，别干——2026-09-16 我把 README 那条同性质的 DSpark SHA 声明删掉就是这个道理。

相关：[[wsl-run-environment]]、[[deepseek-v41-flash-facts]]
