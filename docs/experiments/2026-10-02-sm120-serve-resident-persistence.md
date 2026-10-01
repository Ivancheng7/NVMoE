# 服务面的常驻显存不再每个请求重建：第 2 个请求起每次省 4.6–5.2 秒（2026-10-02）

## 结论

服务面引擎是一个长驻进程、逐请求调用生成入口，但**整模型固定权重的常驻池和 routed 槽位
scratch 都是那次调用里的局部变量**——每个请求进来都把 6.25 GiB 固定权重重读、重上传一遍，
fan-out 车道重新锁页一遍。相位账里看得清清楚楚：旧代码三个请求每个都背
`physical.fixed_source_bytes = 6.25 GiB`、`fixed` 相位墙 `1.2–2.1 s`。

这次把这两样提升到服务循环外（生成入口加两个参数，调用方传进来；一次性调用方传空池，
行为与以前逐位相同）。新旧二进制交错两轮、每轮各起一个引擎连打 3 个请求：

| 臂 | 请求 1 TTFT | 请求 2 TTFT | 请求 3 TTFT | 请求 2/3 的 fixed 墙 | 请求 2/3 的 fixed 读账 |
| --- | ---: | ---: | ---: | --- | --- |
| 旧（每请求重建） | `16.4 s` | `18.5 s` | `19.1 s` | 每请求 `1.2–2.1 s` | 每请求 `6.25 GiB` |
| 新（常驻跨请求） | `19.3 s` | `14.4 s` | `13.6 s` | **`0.00 s`** | **`0.00 GiB`** |

**同一引擎内看（消除盘速窗口的跑间差）**：旧臂三个请求 TTFT 平坦（16.4 → 18.5 → 19.1，
没有跨请求收益），新臂从第 2 个请求起掉到 `14 s` 档。整条请求 e2e 同样是每请求
**省 `4.6–5.2 s`**（旧 31.1/31.0 对新 26.5/25.9）。省的构成：`fixed` 相位墙那一截
（1.2–2.1 s）直接归零；其余 ~3 s 是重填的 H2D 与重读在 chunk pass 里跟 lane 工作挤在
同一段墙里的部分。

**第一个请求不吃这个收益**（它本来就要付一次填充），数值逐位没动：

- 服务面 **12 次运行的 id 串全部相同**（`58572,42549,42549,109865,…`，两臂两轮各 3 请求）。
- 新二进制命令行闸门：1 步四个钉死值 **baseline reproduces**；8 步 8 个 digest/摘要链/argmax
  对 `defaulton` 参照**只有显存峰值一个差异**（判据允许的"代价不是数值"）。
- 新臂第 2、3 个请求的 `fixed residency` 打印是 `344 layer-visits served from VRAM`
  （chunk pass 也全命中），旧臂是 `301`（只有解码段命中）。

**没声称的部分**：

- 两臂固定按"新先旧后"跑，没有换序重跑。这只影响**各轮第 1 个请求**的冷读
  （新臂的第 1 请求因此偏慢 `2.3–2.9 s`，方向对它不利）；上面引用的收益全部来自
  **同一引擎内请求 2/3 对请求 1 的比较**，与跑序无关。
- 服务面多请求**并发**没验（引擎循环仍是逐请求串行）；常驻池现在跨请求占住 ~6.7 GiB 显存，
  并发预算的算法没变（预检仍按当前空闲显存夹，只数本次调用新建的捕获）。
- 第 1 个请求的填充本身没动（~1.7–2.3 s 的 `fixed` 墙还在它身上）；"引擎启动时先填满再 READY"
  是另一件事，没做。
- 预填充本身的逐行成本（每行 ~1.7 s 等于一个解码 token）一行都没改——那是 B 线的下一刀。

## 固定输入与模型

- 模型 `/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，48 个分片
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- prompt `你好，你是谁？`（8 token），`--max-tokens 8 --repeat 3`（一个引擎连打 3 个请求）
- 旧臂二进制 @ `93423d8`（交付 tip），新臂 @ `2a49dbb`（本改动），sha256 见运行证据

## 运行证据

- 构建：树 `/srv/localmoe/build/rename-run-to-step`，`run-identified.sh build --arch sm_120
  -- --release -p colibri-cuda`，两份二进制同 target 目录先后构建、零 error：
  - 旧 `14e7134b1b0e9ba868e958267f13349fea35af6f42dd0c5593c4f0c5b52f368e`
  - 新 `4efa4a7aca5cebebfcbe947ed918aba4783174836e5a4182aaef99d91956626c`
- 批：`/srv/localmoe/targets/serve-resident-ab-2026-10-02/`（2026-10-02 00:43–00:50，
  4 个服务运行 × 3 请求 + 2 个命令行闸门，全 exit 0；驱动脚本 Windows 侧
  `.zcode/run_serve_resident_ab.sh`，汇总 `.zcode/serve_resident_sum.py`）
- 前置账（动机证据）：`/srv/localmoe/targets/firsttoken-repeat-2026-10-01/three-requests-phase.jsonl`
  —— 交付 tip 上三个请求各自 `physical.fixed_source_bytes = 6.25 GiB`

## 代码变更

提交 `82b99f4` + `9de3fa3`（生命周期）+ `2a49dbb`（补回 mut），都在 `exp/lane-contention`：

- `generate.rs`：`run_v4_b2_generate_arm` 加两个参数——`resident: &mut Vec<Option<B2RawRuntime>>`
  与 `routed_slot_scratch: &mut RoutedSlotScratch<'module>`（'module 与 `module` 参数同寿命，
  `step_b2_layer` 要求）。arm 内不再自建这两样；池长度不符时按层数重置。
  捕获预算**只数本次调用新捕获的字节**（带来的池不占新预算），池里有内容时常驻视为开启
  （即使预算算出来是 0 也不把已有池当不存在）。`fixed residency` 打印改报池的真实大小。
  四个调用方（b2_generate / flash_base / serve / b4 bench）传空池 + 新 scratch，行为不变；
  **服务循环外**建一次、逐请求复用——这是唯一的行为改动。
- 数值路径零改动：池/车道的内容与请求无关（权重不变、缓冲每次用前重写），KV 与逐请求状态
  仍在 arm 内。

## 未决事项

1. **预填充的逐行成本是下一个真目标**：chunk pass 每层 8 行仍付 8 遍
   attention(~10) + router(~3.5) + MLP(~12) 单行开销（毫秒/行/层），32 行 prompt 的
   token 0 是 55 s。要么把 MoE 段按行批处理（需要保 K 序的行并行内核，逐位不变可达成），
   要么把"下一行的 attention+router"与"当前行的 lane 窗口"流水起来（纯调度，逐位不变由
   构造保证）。两者都要拆 `step_real_layer_rows` 的层内顺序，是下一场硬仗。
2. **引擎启动时先填常驻再 READY**：把第 1 个请求那 ~2 s 填充也挪出 TTFT（服务面启动多 ~2 s）。
3. 服务面多请求并发的显存预算要按"常驻池常驻不释放"重算（现状 §五已挂的那条，现在更实际）。
4. 本改动在 `exp/lane-contention`（`2a49dbb`），数值三闸门全过，**可合并交付**；合并时按
   老规矩跑一次 `verify_delivery_arm.sh` 的换序批。
