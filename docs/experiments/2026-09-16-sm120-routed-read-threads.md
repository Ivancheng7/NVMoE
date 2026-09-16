# routed 专家读盘并行化：单独没用，配上常驻省 40 秒（2026-09-16 晚）

## 结论

把一层六个专家的读拆成六条并发流，**读盘窗口确实被打下来了**：`shared`（= 预取线程的读 + 申请 + 尾巴）
从 `76.8–80.6 ms/层` 掉到 `16.6–17.9 ms/层`，八步八个 digest 和峰值显存一个没变。

**但在当前交付配置下，总时间一点没动**：墙 `91.4/94.3 s → 91.9/93.3 s`。
读盘省下的 60 ms 原封不动出现在固定参数窗口里（`fixed` `82.2/84.4 → 151.9/154.2 ms/层`）。
盘只有一块，每层要搬的字节没变——routed 专家 80 MB + 固定参数 156 MB——时间只是换了个窗口等。

**只有把固定参数常驻打开之后，并行读才兑现**：

| 配置（每次跑前清缓存，交错各两遍） | 墙 | `outer`/层 | `shared` | `fixed` | 固定参数进盘 |
| --- | --- | --- | --- | --- | --- |
| 常驻关 + 1 线程（现状） | 91.4 / 94.3 s | 247 / 253 ms | 76.8 / 80.6 | 82.2 / 84.4 | 53.67 GB |
| 常驻关 + 6 线程 | 91.9 / 93.3 s | 248 / 250 ms | **17.2 / 16.9** | 151.9 / 154.2 | 53.67 GB |
| 常驻开 + 1 线程 | 73.1 / 73.6 s | 192 / 193 ms | 74.6 / 75.7 | 34.1 / 33.6 | 6.71 GB |
| **常驻开 + 6 线程** | **51.2 / 54.7 s** | 128 / 138 ms | **16.6 / 17.9** | 45.4 / 52.9 | 6.71 GB |

两个改动是**超可加**的：单独加线程 0 收益；单独开常驻省 18–21 s；两个一起省 40–43 s。
换成每 token：交付记录是 `8.2 s/token`（`outer` 190.8 ms/层），这组是 `5.5–5.9 s/token`。

**两个都已合进交付分支**（`0cb430e` + `f4f1beb` + `bb3976b`，并行读默认 6、常驻默认关），
合并后在交付树上重验：单步四个钉死值全同、八步默认配置 `71.66 s`、
常驻开 `48.84 s`——数字见下面"已合进交付分支"那节。

抢核不是问题：六条读线程合计只占 **2.4 个核**（`0.63 CPU-秒/GB`，几乎全在内核拷贝路径），
6 核机器上主线程和固定参数 worker 还有余量。

没声称的部分：

- **绝对墙钟不能和记录里的 `70.06 s` 直接比。** 这批每次跑前都 `drop_caches`（冷启动），
  交付那次没有；而且同一天里同一份代码的墙在 `76–94 s` 之间都出现过。能比的是**同一批交错**的相对值。
- 这批数字里有一轮（`routed-threads-*-8step-*`）撞上另一个会话在跑引擎（`prompt-chunk`，21:08 起），
  所以那一轮的绝对值作废；但它和安静那轮给出的是同一个形状（读盘窗口降、固定参数窗口升、总时间守恒），
  这条交叉验证本身有效。
- 回放用的专家 id 仍是固定种子生成的伪随机六元组，不是真实路由结果。
- 常驻打开时 `attention` 窗口从 `~31` 掉到 `~20 ms/层`，**没有查原因**，别当成 GPU 变快。

## 固定输入与模型

- 模型目录：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`，48 分片、`166898773853` 字节
- `config.json` SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- 机器：Ryzen 5 7500F（6 核 12 线程）、15 GiB 内存、`/dev/sdd` ext4（WSL2 虚拟盘）
- 分支（都在 Windows 侧仓库，WSL 克隆里也同步过）：
  - `exp/routed-read-threads` `646c07d`：在交付分支 `29419ae` 上加并行读
  - `exp/resident-parallel-read` `448875a`：`exp/fixed-host-reuse` `29009b9` + 上面那个提交（cherry-pick）
- 构建：`./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda`，**两份都是 59 条警告**
- 新开关：`COLIBRI_V4_ROUTED_READ_THREADS`（**默认 1 = 现在交付的行为**，代码在 `layer_rows.rs::routed_read_threads`）；
  常驻用既有的 `COLIBRI_V4_FIXED_HOST_RESIDENT_GIB=7`

## 运行证据

### 1. 隔离回放：并行读本身有效（`/srv/localmoe/targets/io-ceiling/cpu-accounting.jsonl`）

3.449 GB/token、1548 次 `pread`、缓冲复用、每次跑前清缓存：

| 线程 | 墙 | 吞吐 | 占核 |
| --- | --- | --- | --- |
| 1 | 2.69 / 3.23 s | 1.35 GB/s | 0.74 / 0.81 |
| 2 | 1.48 / 1.57 s | 2.2 GB/s | 1.25 / 1.30 |
| 3 | 1.18 / 1.25 s | 2.0 GB/s | 1.68 / 1.72 |
| 6 | 0.88 / 0.90 s | 3.7 GB/s | 2.45 / 2.48 |
| 12 | 1.07 / 1.08 s | 3.2 GB/s | |

（4 线程那档没列：专家数是 6，四线程会分成 2/2/1/1，慢的那两个线程决定墙，不是真的平台期。）

### 2. 再压一条固定参数流进去：抢的是盘，不是核（`stream-sharing.jsonl`）

routed 3.449 GB 与"固定参数形状"的 5.95 GB 同时读（不同区域，互不重叠）：

| 场景 | routed 墙 | routed 吞吐 | 两流合计 |
| --- | --- | --- | --- |
| routed 6 线程，单独 | 0.94 s | 3.68 GB/s | — |
| routed 6 线程 + 固定参数流 | 1.07–1.11 s | 3.14–3.22 GB/s | 2.1–2.6 GB/s |
| routed 1 线程 + 固定参数流 | 2.72–2.77 s | 1.24–1.27 GB/s | 2.8 GB/s |
| 固定参数流单独（1 线程） | 2.81 s | 2.12 GB/s | — |

并发的 routed 流只掉 13%，但两条流加起来也不到单流六线程的 3.68 GB/s——
这台盘在两股交错访问下比一股要差。

### 3. 引擎里的 2×2

结果文件（每格两遍，逐一列出）：

- 常驻关：`/srv/localmoe/targets/quiet-unset-8step-1|2.jsonl`、`quiet-6-8step-1|2.jsonl`
  （另有受干扰的一轮 `routed-threads-{unset,3,6}-8step-1|2.jsonl`，只作交叉验证用）
- 常驻开：`/srv/localmoe/targets/resident-unset-8step-1|2.jsonl`、`resident-6-8step-1|2.jsonl`

墙的汇总在 `quiet-ab-summary.txt`、`resident-ab-summary.txt`；每层平均用
`layer_averages.py` / `compare_threads.py` 从结果文件里读。

### 4. 正确性

十六次运行（受干扰轮 6 次 + 安静轮 6 次 + 常驻轮 4 次）的八步 token digest 全部相同：

```
7933f6f28bc01f11 66fb8d43b1be58ed aec5c709904df23f 4a6637bc7b224ac6
3e4fc2900b03f605 4b1ac938a4cb541c 211ce2414c01c9c6 864dfbfab49f65b5
```

`argmax` `[55222, 33832, 42549, 42549, 42549, 42549, 42549, 127384]`、
峰值显存 `1466499072` 字节也都不变。检查脚本 `.zcode/io-ceiling/check_digests.py`，
digest 字段在结果文件的 `requests[0].finals[].collapsed_bf16_digest`。

### 复现

```bash
# 隔离回放
DIGEST_STRIDE=0 disk_ceiling plan-expert.tsv reuse 0 6      # 6 线程最好一档
bash run_sharing.sh                                        # 两流共享
# 引擎（交错、每次清缓存）
bash run_quiet_ab.sh                                       # 常驻关：unset / 6 / 3 交错
bash run_resident_ab.sh                                    # 常驻开：unset / 6 交错
python3 check_digests.py "/srv/localmoe/targets/resident-*-8step-*.jsonl"
```

工具源码：`/srv/localmoe/targets/io-ceiling/src/` 与 Windows 侧 `.zcode/io-ceiling/`。

## 代码变更

`rust/colibri-cuda/src/layer_rows.rs`，两个提交：

1. 新增 `routed_read_threads()`：读 `COLIBRI_V4_ROUTED_READ_THREADS`，默认 1。
2. Flash B1 预取那段从"一个 worker 顺序读六个专家"改成"按 `chunks(chunk_size)` 起 N 个 worker，
   再按原顺序拼回 `Vec<(u16, RoutedExpert)>`"。默认 1 线程时只有一个 chunk，行为与交付一致。

没有改读取方式（仍是 buffered `pread` 到 `vec![0; N]`）、没有改数值路径、没有插桩。

### 已合进交付分支（2026-09-16 晚）

- `0cb430e` merge `exp/fixed-host-reuse`（`29009b9`，默认 `COLIBRI_V4_FIXED_HOST_RESIDENT_GIB=0` = 关；
  它下面两个 `probe:` 提交是纯计时字段、无行为改动，就是量出那些数字的那份二进制，所以一起合）
- `f4f1beb` merge `exp/routed-read-threads`（`646c07d`）
- `bb3976b` 把并行读的**默认值改成 6**（= 一个专家一条流）。常驻关时它中性
  （八步 `91.9` vs `91.4 s`，在漂移带内），所以默认可开；`COLIBRI_V4_ROUTED_READ_THREADS=1` 回到旧形状。

合并后在交付树上重验（`59` 条警告的二进制，结果文件见下）：

| 检查 | 结果 |
| --- | --- |
| 单步 `requests[0].tokens[0].hidden_bf16_digest` | `5ed8bf88beae3cfe` = 参考 `post-merge-1step.jsonl` |
| 单步 `argmax` / `layers` / 峰值显存 | `55222` / `43` / `1466499072`，全同 |
| 单步 `collapsed_bf16_digest` / `normalized_bf16_digest` / `logits_f32_digest` | `7933f6f28bc01f11` / `c56dcc4ba9ef0b03` / `0488954427e4468e`，全同 |
| 单步第 0 层 `route_ids` | `[254, 222, 245, 200, 53, 35]`，全同 |
| 八步（默认配置：并行读 6、常驻关） | `71.66 s`，八个 digest 与记录那组逐个相同 |
| 八步（`COLIBRI_V4_FIXED_HOST_RESIDENT_GIB=7`） | **`48.84 s`**，八个 digest 逐个相同 |

八步两跑的每层平均：默认配置 `outer 192.1`（`shared 15.0` / `fixed 98.0` / `attention 31.1`），
常驻开 `outer 121.6`（`shared 16.8` / `fixed 35.7` / `attention 22.3`），
固定参数进盘 `53.67 GB → 6.71 GB`。结果文件：
`/srv/localmoe/targets/postmerge-routed-threads-1step.jsonl`、
`postmerge-routed-threads-8step.jsonl`、`postmerge-resident-on-8step.jsonl`。

**交付分支现在的数字**：默认 `71.66 s`（同一天同一状态下的对照是 `91.4–94.3 s` 的常驻关单线程，
以及记录里的 `70.06 s`——后者是热缓存状态，别直接相减）；把常驻打开是 `48.84 s`。

## 未决事项

1. **常驻默认还是关的**：`48.84 s` 要显式设 `COLIBRI_V4_FIXED_HOST_RESIDENT_GIB=7` 才有。
   要不要把默认也打开，是一个"吃 6.5 GB 内存换 32% 速度"的部署决定；没定之前别把 48.84 当交付数字。
2. 常驻打开后 `attention` 窗口从 `~31` 掉到 `~22 ms/层`，原因没查。别当 GPU 变快。
3. 并行读在 3 线程时略差于 1/6（`253–259` vs `248–250 ms/层`），只有 6 线程稳定最好；
   6 线程占 2.4 核，还留有余量，但没有试过 5 线程。
4. 页缓存命中率仍然没有字段（`2026-09-16-sm120-disk-ceiling.md` 的未决事项 3）。
   这次常驻开+并行读是冷启动 `48.84 s`，记录里的 `70.06 s` 是热状态，两个数不能直接相减。
5. 常驻那个 `923 ms` 的 H2D 异常仍未解释（`2026-09-16-sm120-fixed-host-residency.md` 里记的）。
6. 回放的专家 id 可以换成真实轨迹了（`tokens[].layers[].route_ids`，见
   `2026-09-16-sm120-disk-ceiling.md` 里那条更正）。
