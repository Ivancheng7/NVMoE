# 服务面的 fan-out 两臂验证，以及把它翻成默认（2026-09-30）

## 结论

`COLIBRI_V4_FLASH_ROUTED_FANOUT` 从"默认关"翻成"默认开"之前，先补上了**唯一没验过的交付面**：
服务面。同一个二进制、两臂交错各三轮，服务面用真实 prompt 文本生成 8 个 token，两臂的 id 串
逐字符相同，稳态每字 `2.231 → 1.767 s`（`−0.464`，`−21%`）。翻默认之后，命令行默认面的 1 步
四个钉死值和 8 步的 8 个 digest / 8 条摘要链 / 8 个 argmax 全部复现，**两个顺序**（默认先跑、
关先跑）都一样，唯一差异是显存峰值 `8175288320 → 8242397184`（`+64 MiB`，代价不是数值）。

**没声称的部分（必须一起读）：**

- **服务面只跑了单请求、一个 prompt、8 个 token**。多请求并发没验；fan-out 每个请求多锁约
  `64 MiB` 页锁定内存、多占约 `64 MiB` 显存，服务面并发的预算要重算（`现状.md` §四.5 已挂）。
- **没有重做"服务 vs 命令行同批对照"**（那是 `serve_vs_cli.sh` 的活）。本批服务面 `1.767 s/字`
  与命令行 `1.98 s/token` **不是同一形状**（服务面是真实 5 字 prompt，命令行是 `TOKEN_IDS=0`
  单 token 形状），两个数不能相减。
- **服务面没有 digest**，它的数值证据只有 id 串；服务面相位账这次落盘了（6 个文件），但本批
  没在第①②段之间做相位级比对。
- 命令行闸门批的"稳态"是**单 token prompt 形状**（`COLIBRI_V4_FLASH_TOKEN_IDS=0`），与 fan-out
  记录里真 32 token prompt 的 `1.913 → 1.581` 不是同一形状，绝对秒数不可跨批比。
- **A2（按专家身份的 routed 缓存）没有动**，默认仍关；它的默认值仍待拍板（见未决事项 3）。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，48 个 `*.safetensors` 分片，
  `stat` 合计 `166886535336` 字节
- `config.json` SHA-256：`6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- 卡：RTX 5060 Ti 16 GB（`sm_120`）；cuBLAS `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`
- 盘：WSL2 虚拟盘；本批现场闸门见下（`3.22–3.50 GB/s`，快档下沿）

## 运行证据

### ① 盘速现场闸门（跑批之前先量，防"慢档窗口"污染）

探针 `/srv/localmoe/targets/io-ceiling/disk_ceiling`，plan `plan-expert.tsv`，
`mode=reuse direct=0 threads=6`（`1548` 次读、`3449290752` 字节），每次跑前
`sync; echo 3 > /proc/sys/vm/drop_caches; sleep 1`：

```text
run 1  wall_gbps 3.2248   wall_s 1.069599
run 2  wall_gbps 3.4147   wall_s 1.010125
run 3  wall_gbps 3.4969   wall_s 0.986381
digest 00000000000000ba   errors 0（三跑相同）
```

按 `docs/experiments/2026-09-20-sm120-read-concurrency-ceiling.md` 的两档（快 `3.5–3.75`、
慢 `2.7–2.8`），本批落在**快档下沿**，不是慢档，判定有效。复现命令写在
`.zcode/disk_gate.sh`（Windows 侧脚本，WSL 里用 `/mnt/c/...` 调）。

### ② 服务面两臂交错批（本记录的主证据）

构建：树 `/srv/localmoe/build/merged-prompt-chunk` @ `808a7bc`（`git status --porcelain` 为
**0 行**），`CARGO_TARGET_DIR=/srv/localmoe/targets/merged-prompt-chunk-target`，
`RUSTC_WRAPPER=<tree>/rust/rustc-human-wrapper.sh`，
`cargo oxide build --arch sm_120 -- --release` → **exit 0**；

```text
binary sha256 8ab8808a441e2fb940c46c519283a072dff2c94f572a10da809f6ff0c0f7e4f4
```

驱动：`tools/serve_arm.sh`（本批新增）→ `tools/flash_serve_client.py`，prompt
`"\u4f60\u597d\uff0c\u4f60\u662f\u8c01\uff1f"`（`你好，你是谁？`），`--max-tokens 8 --repeat 1`；
两臂只差一个环境变量 `COLIBRI_V4_FLASH_ROUTED_FANOUT=0|1`，两臂都设
`COLIBRI_V4_FLASH_SERVE_PHASE_JSONL` 拿服务面相位账。客户端自己设
`COLIBRI_V4_FLASH_SERVE=1 / DSV4_MODEL_DIR / STREAM_READ_MODE=buffered / FLASH_PREFETCH=1`、
三个线性臂，`FP4_ACCUM_ARM` / `FINAL_HEAD_ARM` 故意不设（用代码默认）。

| 臂 | 轮 | 首字 s | 稳态 s/字 |
| --- | ---: | ---: | ---: |
| 关 | 1 | 30.72 | 2.352 |
| 关 | 2 | 20.72 | 2.162 |
| 关 | 3 | 20.34 | 2.179 |
| 开 | 1 | 18.64 | 1.753 |
| 开 | 2 | 17.47 | 1.770 |
| 开 | 3 | 17.60 | 1.778 |

```text
id strings: IDENTICAL across 6 runs
  ids=58572,42549,42549,109865,42549,42549,42549,42549
off steady per char: 2.231 s (2.162-2.352, n=3)
on  steady per char: 1.767 s (1.753-1.778, n=3)
on - off = -0.464 s/char
```

这条 id 串与 `现状.md` §三 里记的历史服务面串（`58572,42549,42549,109865,42549,…`）**逐字符
相同**，所以它同时说明两件事：数值没动，以及这个二进制就是那个引擎。首字也从 `20.3–30.7 s`
降到 `17.5–18.6 s`（预填充那一段也走这条路，与 fan-out 记录的方向一致）。

结果文件（WSL 绝对路径，只追加）：

- `/srv/localmoe/targets/servearm-fanout-servearm1/batch.log`
- `/srv/localmoe/targets/servearm-fanout-servearm1/serve-{off,on}-r{1,2,3}.log`（6 个）
- `/srv/localmoe/targets/servearm-fanout-servearm1/serve-{off,on}-r{1,2,3}-phase.jsonl`
  （6 个，各 1 条记录——**服务面相位账第一次有数据**，见未决事项 6）

### ③ 翻默认之后的命令行闸门（`9799a0c`）

`tools/verify_delivery_arm.sh 9799a0c <tag>`，`ARM=COLIBRI_V4_FLASH_ROUTED_FANOUT LABEL=fanout`，
两趟：默认顺序与 `ORDER=off-first`。两趟的 `git status --porcelain` 都是 0 行，`build exit=0`。

**数值（两趟、两臂、1 步与 8 步）：全部复现。**

```text
argmax               55222                        55222                        OK
fp4_accum            block32                      block32                      OK
final_head           canonical_cpu_streamed_bf16  canonical_cpu_streamed_bf16  OK
VERDICT: baseline reproduces
8step-fanout-on.jsonl                        DIFFERS: peak device bytes
8step-fanout-off.jsonl                       DIFFERS: peak device bytes
```

`outputs_match.py` 对参照 `defaults-pc-8step.jsonl` 的唯一差异就是**显存峰值**——那是槽位/车道
自己占的设备缓冲，判据里明确允许（"代价不是数值"）。8 个 digest、8 条 43 层摘要链、8 个
argmax 逐项相同。

**稳态每 token**（`TOKEN_IDS=0` 单 token 形状；定义与 `serve_vs_cli_sum.cli_steady` 一致）：

| 顺序 | fan-out 关 | fan-out 开（新默认） | 差 |
| --- | ---: | ---: | ---: |
| 默认先跑 | `2.388` | `1.986` | `−0.402` |
| 关先跑 | `2.403` | `1.977` | `−0.426` |

**峰值显存**：关 `8175288320`、开 `8242397184`（两趟一致，`+64 MiB`）。

**相位签名**（默认先跑那一趟，稳态均值，s/token）：`fixed` 两边都是 `0`（整模型常驻默认开）；
`routed` `0.79 → 0.58`、`shared` `0.68 → 0.46`，`attention` `0.57 → 0.58`、`router`
`0.12 → 0.13`、`head` `0.10 → 0.10`。也就是这次省下来的是 **routed 段**，其它相位没动——
收益出现在它该出现的相位上（`shared` 变小是因为读盘等待从 `shared.join_tail` 挪进了各条
lane 自己的链子里，不是 MLP 变快）。

结果文件（WSL 绝对路径）：

- `/srv/localmoe/targets/ab-fanout-fanoutflip/`（默认先跑）：`batch.log`、`1step-fanout-on.jsonl`、
  `8step-fanout-{on,off}.jsonl`
- `/srv/localmoe/targets/ab-fanout-fanoutflip-off/`（关先跑）：同上命名

## 代码变更

- `808a7bc`（`tools:`）新增 `rust/colibri-cuda/tools/serve_arm.sh` 与 `serve_arm_sum.py`：
  服务面的换臂验证。`serve_vs_cli.sh` 只能比"服务 vs 命令行"，没有换臂参数；这个脚本补的是
  "同一个面上的两臂"。汇总用 `serve_steady()`（从 `serve_vs_cli_sum.py` 导入），只加了按臂分组
  与 id 串比对。
- `9799a0c`（`flash:`）把 `routed_fanout_enabled()` 从"只认 `=1`"改成
  `parse_default_true_zero_one_flag`（默认开、`=0` 回退、非法值报错），调用点加 `?`；
  函数上方的注释写明默认开、收益、`+64 MiB` 代价与 `=0` 的语义。**这是唯一的行为改动**：
  闸门批之前默认关、之后默认开，别的什么都没动。

## 未决事项

1. **服务面多请求并发没验**。fan-out 每请求多占 `64 MiB` 显存 + `64 MiB` 锁页主机内存；服务面
   几路并发就乘几。常驻的预检只挡 OOM 不挡性能（`现状.md` §四.5），并发预算要重算。
2. **`serve_vs_cli.sh` 与换臂现在是两件事**。要不要合成"每个臂都两面都跑"的一条批（那才是完整
   的"翻默认之前两面都验"），没定。
3. **A2（`COLIBRI_V4_ROUTED_EXPERT_CACHE`）默认值仍待拍板**：叠在 fan-out 上只再省
   `0.07 s/token`，代价 `3.2 GiB` 锁页主机内存。本身逐位安全。
4. **fan-out 的六套槽位里有一套是浪费的**（约 `60 MiB` 设备 + `60 MiB` 锁页），见
   `2026-09-21-sm120-routed-fanout.md` 未决 5；值不值得省取决于显存档位。
5. **`/srv/localmoe/build/deliver-merge-0918` 那棵树还带着 6 个文件的未提交残留**
   （`layer_rows.rs` 等，忽略行尾差异后约 700 行），与已提交内容的关系还没逐个比对过。谁要用
   那棵树，先做这一步。
6. **服务面相位账已落盘但没读**：6 个 `serve-*-phase.jsonl` 就在上面那个目录里。下一步可以拿它
   把首字 `TTFT` 拆成"预填充趟 + 冷启动"（`现状.md` §四.1 的那一条）。
