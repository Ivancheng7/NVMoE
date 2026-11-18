---
name: flash-0731-dspark-arm-state
description: 本地 DSpark 臂的真实状态——接受长度 0（2026-10-01 在新交付线上重量仍 0/20）；第一处缺陷是自写 rope kernel >4 行静默清零，第二处未定；官方参考实现本机齐备
metadata:
  node_type: memory
  type: project
  originSessionId: sess_a615c8b1-894c-4181-84b2-ced4dcf9ab31
---

2026-09-16 深夜读代码 + 核本地权重得出。**没跑过**，下面全部是代码与索引事实（该跑一次的是接受长度）。

---

## 2026-10-01：这条臂已从交付线移除（`93423d8`）

不是「先放着以后再修」——是**拆掉了**。判据不是「现在接受率是 0」，是**算术上打不赢**：
一轮花掉 1 次 anchor + 5 次逐行的 base 前向（现有验证是逐行的），最多提交 6 个 token，
所以 5 个草稿全接受也只是打平，再加 3 个草稿 stage 就是净亏。真正能改变这个账的
「一趟验 N 行」利好的是 base 路径本身，所以这条臂没有可加的东西。

**改了什么**（13 个文件，+92/−1635）：`FLASH_DSPARK` 入口删掉、改成**硬报错**
（`COLIBRI_V4_FLASH_DSPARK was removed…`，exit 101、不产出结果文件，防止老脚本静默穿透到
smoke test）；`dspark.rs` 整个删除，其中 `step_device_attention_row` 是 base 解码循环在用的，
搬去了 `device_attention.rs` 和它的两个兄弟放一起；`generate.rs` 里 569 行的 dspark 块删掉，
但 **`FlashDsparkFinalHeadGpuWorkspace` 保留**（名字带 dspark，但 base 面走 `gpu_exact` 臂时在用它）；
`v4_state.rs` 减 645 行（张量/索引合同、模型目录校验、B5 ledger 与 prepared/validated 类型）、
`device_weights.rs`/`v4_expert.rs`/`fixed_layer_planning.rs` 里的死载入器；
`routed_expert_prefix` **保留**（测试在用，`cargo build` 看不到）。

**顺带留下的那一行**：`step_rope_device_into` 现在超过 4 行就直接返回错误。那个 kernel
超限时静默 return、输出保持全零——DSpark 的草稿就是这么被清零了几个月。静默清零改成响亮失败。

**`mtp1_*`（约 2,500 行）刻意没动**：`step_mtp1_draft_candidate` 和 `mtp1_resources` 那组
字节预算也被 `layer_stationary_prefill.rs` 调用，不是 DSpark 专属。

验证：构建 0 error、61 条警告（删除前 63）；base 面**全新产出**、261 个 digest 逐字段与历史参照相同、
`argmax=55222`。调查记录、已排除嫌疑和 `confidence` 入口都留在
`docs/experiments/2026-09-18-sm120-dspark-acceptance.md`，README 两版都改成了结论式并指向它。

---

## 2026-10-01（早先）：在新交付线上重量，结论没变

把 rope 上限那一行和 prompt-text 两个小提交（`10f3daf` / `c65066b` / `181260b`）cherry-pick 到
当时的交付线 tip（`0554419`）上，建 `exp/dspark-on-delivery`——**只在 WSL 树
`/srv/localmoe/build/rename-run-to-step` 里，没推回主仓库**（重建就是那三条 cherry-pick）。
接受率测量台**不用搬**：`generate.rs` 本身就在结果里打 `proposed` / `accepted` / `committed_inputs`。

- **接受率仍是 0/20**（4 轮 × 5 proposal，退化 prompt `[0,1]`）→ 第二处缺陷在新底座上确认还在。
- **产出不变**：DSpark `generated = [33832, 109865, 119061, 62268]`，与同 prompt 的 base 逐位相同。
- **墙钟 `446.5 s` / 4 步**，与 09-18 的 `429 s` 同量级（仍约 16 倍慢）。
- **rope 上限对 base 面完全惰性**：抬到 8 之后交付面仍是 **261 个 digest 逐字段相同**、
  `argmax=55222`、43 层链一致 → 「当 bug 修」不会动交付数值（但它是数值路径，仍要单独 A/B）。
- **`confidence` 线索原样复现**：结果里 `rounds[].confidence` 五行的量级 `-8541 … +28236`，
  绝对值均值 `8749`（官方合同该是 O(1)）。这是第二处的入口，数据就在结果文件里。
- 结果文件：`/srv/localmoe/targets/dspark-on-delivery-2026-10-01/`。

**一处新观察（未定论）**：今天每轮的 `target` 行与 09-18 记录**只有 `target[0]`（锚点）相同**，
第 2 位起不同（记录 round0 `[109865,119061,120159,120159,120159,58572]`，
今天 `[109865,119061,89352,127885,62268,109865]`）。锚点链一致、产出也一致，所以不是回归；
怀疑是**修 rope 之前草稿的全零行被写进了 attention cache**，污染了当时的 verification target。

---

## 2026-09-18 更新：已真跑，接受长度 0，根因是我们自己的 rope kernel

真跑之后（`exp/dspark-draft-probe`）结论是：**DSpark 从未接受过一个草稿 token**
（4 次运行、每轮 20 proposal、`accepted=0`），产出与 base 贪心逐位同但慢 12–16 倍。
下面那条"每轮 6 趟 base 前向所以打平"的前提（接受率可能很高）**不成立**，作废。

**根因（已定位）**：`kernels_module.rs` 的 `rope_v4_bf16` 在
`input.len() / row_width > 4` 时**静默 return**（输出保持全零，不报错）。
DSpark 草稿一次喂 5 行 → `q` 和 `query_kv` 出来全零 → 草稿 attention 什么也看不到。
stage 探针实测：`cache_rows=133` 但只有 `[0,1,2]` 非零，行 128–132 全零；链上
`q_nz=0`、`query_kv_nz=0` 而 `q_rope_in`/`query_kv_rope_in` 非零。
这是**本地自写的 guard**，与任何外部代码无关。提交 `10f3daf` 把上限抬到 8，
但**明标 PROBE ONLY, NOT A FIX**（静默返回的合同本身还没定）。

**重要**：抬上限后的 `dspark-ropecap-1.jsonl`（16:25）**接受仍是 0**，所以 rope 不是
唯一的问题，后面还有一处（草稿 confidence 从 ~1 跳到 ~4×10⁴，量级也不对劲）。

**真实文本也已确认 0 接受**（`dspark-realtext-1.jsonl`，prompt `"The capital of France is"`
→ `[671,6102,294,8760,344]`，6 步 30 个 proposal 全拒）→ **不是退化 id 序列的伪影**。
需要 `COLIBRI_V4_FLASH_PROMPT_TEXT`（`exp/dspark-draft-probe` 里加的，走现成 `V4Tokenizer`）。
另：抬 rope 上限没有动交付数值——同二进制跑钉死单步基线仍是 `generated [55222]`。

**官方 oracle 不是"装了 torch 就能跑"**：环境有（上面那句对），但 `inference/generate.py`
用 `load_model(model, f"model{rank}-mp{world_size}.safetensors")` 吃**合并单文件**，
本地只有 48 分片，得自己写分片加载器；且整模型 ~166 GB，16 GB 卡放不下全量
（DSpark 只用到 `mtp.0/1/2` + `layers.40/41/42`，可缩范围）。

**外部开源支持：不需要。** 所需一切本机齐备——官方 `DSparkAttention`/`DSparkMarkovHead`/
`DSparkBlock`/`forward_spec` 就在模型目录 `inference/model.py`（45118B，
sha256 `c0c19e6c…`，随钉定 revision 一起来）；权重 4705 个 `mtp.*` 张量齐全；
可跑的 oracle 环境 `flash-reference-py312`（torch 2.10.0+cu128）+ `tl014-only` 旁路
（已实测修好 sm_120 非确定性）也在。**唯一陷阱**：老 oracle 脚本用的克隆
`sources/deepseek-v4-flash-git/inference/model.py`（38632B，07-29）**根本没有 DSpark 代码**，
要跑官方 DSpark oracle 必须指到模型目录那份。
外部现成实现（`fraserprice/dspark-vllm`、`hw-native-sys/pypto-lib#935`、DeepSpec）只能当交叉验证，
不是依赖。细节在 `docs/experiments/2026-09-18-sm120-dspark-acceptance.md`。

---

## 权重：本地那份就够了

0731 的 48 分片 checkpoint 通过 `validate_flash_0731_dspark_model_dir` 那道权重验收
（我用 Python 按它的合同表手工核过一遍，索引层面 PASS，不占 GPU）：

- `mtp.0/1/2` 分别在 `model-00046/47/48-of-00048.safetensors`（合同的 shard 是 `46+stage`，跟着
  48 分片布局走，所以是 46/47/48 不是 64 分片那份的名字）。
- 三个 stage 各有 **256 个连续专家**；`mtp.0` 多 `main_proj/main_norm`，`mtp.2` 多 `norm`、
  `markov_head.markov_w1/w2`（129280×256）、`confidence_head.proj`（1×4352）、`hc_head_*`。
- `config.json` 带 `dspark_block_size=5`、`dspark_noise_token_id=128799`、
  `dspark_target_layer_ids=[40,41,42]`、`dspark_markov_rank=256`。

跑起来需要的环境：`COLIBRI_V4_FLASH_DSPARK=1`、`COLIBRI_V4_GREEDY_STEPS>=1`、
`COLIBRI_V4_RESULT_JSONL`、**`COLIBRI_V4_FLASH_FINAL_HEAD_ARM=gpu_exact`**（不设直接报错）、
prompt 至少两个 token。入口是 `generate.rs` 的 `run_real_v4_flash_dspark`。

## 已经有的

- 块长 5 的**并行**草稿：`run_flash_dspark_stage` 一次吃 5 行，三个 MTP stage 串起来；
  markov 头 + 置信度头都在（`flash_dspark_head`）。
- 状态事务：每轮克隆前缀、逐行验证、`validate_greedy_target_bounded` 判前缀、
  `commit_prefix` / `rollback`，四个状态 digest（base + mtp0/1/2）都在结果里。
- 验证是**精确贪心比对**：草稿和 base target 逐位相同才算接受。

## 缺的（这才是重点）

1. **验证是逐行的**：`run_flash_base_input` 一次一行、跑满 43 层、每层还要
   `load_b2_raw_runtime_for` 重新读权重。一轮 = 1 次 anchor + 5 次 proposal 验证 = **6 趟完整 base 前向**，
   最多换回 6 个 token。也就是**接受率 5/5 也刚好只打平**，加上草稿那 3 个 stage 就是净亏。
   所以"跑一次看能快多少"这个问法必然得到"没快"的答案，跟接受率无关。
   想要那个乘数，必须把验证改成**一趟 N 行**（权重按趟读，不按行读）——
   而那正是 prompt-chunk / 设备 attention 那条线在做的事（`layer_stationary_prefill.rs` 已经是一层一趟多行；
   解码侧主机 attention 路径 `rows > 2` 直接报错，是硬限制）。
2. **置信度只查了有限性**（`validate_greedy_target_bounded` 里 `.any(|v| !v.is_finite())`），
   没有参与"验证几行"的决策。DSpark 论文里那半（按引擎吞吐曲线定验证长度）**没实现**。
   这半边对我们特别要紧，因为验证行数直接乘上 routed 读盘量。
3. **草稿的 MoE 还在每个专家重建 `ExpertSlot`**（`dspark.rs` 的 `flash_dspark_moe` 循环里
   `ExpertSlot::new`），也就是刚在 base 路径上删掉的那 17 ms/专家锁页，这里一轮要重付
   （3 stage × 每 stage 去重后最多约 30 个专家）。

## 一次 batched 验证的字节账（按交付实测外推）

一趟 k 行的验证：`fixed 6.71 GB`（跟行数无关）+ `routed 43 层 × 6 专家 × k 行 × 12 MiB`
（k=6 最坏 18.6 GB，专家去重会低）+ `head 1.06 GB`。k=6 约 26 GB ≈ 7.5 s 换最多 6 个 token，
对比现在 11.2 GB/token ≈ 9 s/token。这是唯一一个能把"每 token 重读一遍权重"这个乘数打掉的方向。

## 其它两条更正

- 结果文件里那 90 份 `"dspark":"disabled"` 是 base 那条写结果分支里**写死的字面量**（`generate.rs`），
  不能当"没跑过 DSpark"的证据。
- README 里唯一一次跑过的记录是 sm_89 的 B 回归：`161.55 s`、`1.124 GiB` maxRSS、8 次 base 等价计数、
  **接受 0 个 proposal**、committed one input；那份台账不在仓库里，无法从 tracked 文件复算。
  0 接受说明当时要么草稿质量为零、要么草稿管线有 bug——这跟"跑得快不快"是两个问题。
