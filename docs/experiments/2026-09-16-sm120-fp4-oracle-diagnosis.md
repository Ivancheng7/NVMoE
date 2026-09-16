# `sm_120` 上官方 FP4 oracle 诊断（2026-09-16）

## 结论

**官方 TileLang FP4 routed 专家在 `sm_120` 上不是"归约顺序不同"，是算错了。**

同样的权重、同样的输入、同样的源码、同样的 torch/tilelang 版本、同样的可用显存，两次运行
在 routed 专家的 FP32 累加器上给出 **510/4096 个元素不同，其中 440 个相对误差超过 1%，
最大 30.7 倍**。浮点归约只是换个相加顺序的话，相对误差应该在 fp32 机器精度附近，也就是
`1e-7` 量级。实测差了两到五个数量级。

这条结论决定了一件重要的事：**`sm_120` 上不存在可用的官方数值真值，重跑也修不好。**
上一份实验记录把它写成"不是稳定 oracle"，那个说法太温和了。

## 怎么查的

分两步，都不需要新的模型运行也能先做一半。

**第一步：把已有的八次官方运行两两比对。** `/srv/localmoe/targets/` 里存着八份官方 full-43
结果，用新增的 `tools/compare_official_run_repeatability.py` 逐个字段比。结果：**七个运行的
layer 0 输出两两都不同**，每一份都是独一无二的。最终 token 出现三种值：`0`、`55222`、`67900`。
连两次给出同一个 token 的运行（r10 与 v5），layer 0 也完全不一样。所以这不是"在少数几种
结果之间跳"，是每次都不一样。

**第二步：抓 layer 0 的实际数值。** 用 oracle 的 `--cutpoint-dir` 加 `--cutpoint-layer 0`
跑两次，拿到 23 个中间张量的原始数值，再用 `tools/compare_oracle_cutpoints.py` 逐个比。

## 证据

固定输入：token `0`，`start_pos 0`，43 层模型的第 0 层。两次运行分别落在

- `/srv/localmoe/targets/fp4diag/a.jsonl` 与 `/srv/localmoe/targets/fp4diag/a-cut/`
- `/srv/localmoe/targets/fp4diag/b.jsonl` 与 `/srv/localmoe/targets/fp4diag/b-cut/`

两次运行的环境记录逐字段相同：`torch 2.10.0+cu128`、`tilelang 0.1.8`、`safetensors 0.7.0`、
`free_vram_bytes 15894315008`，`model.py`/`kernel.py`/`config.json` 的 SHA-256 也相同。
所以不存在"两次跑了不同代码路径"这种解释。

23 个张量里 **21 个逐位相同**，包括：

- 整条注意力路径：`attn_norm`、`q_a_linear`、`q_a_norm`、`q_b_logical`、`q_rope`、
  `wkv_linear`、`kv_norm`、`k_rope`、`cache_kv_row`、`attn_scores`、`attn_probs`、
  `attn_context`、`wo_a_logical`、`wo_b_hidden`、`post_attn_hidden`
- 路由：`router_pre_score_f32`、`router_post_score_f32`、`route_ids_i32`、`route_weights_f32`
- 共享专家：`shared_out_bf16`

**差异只出在一个地方：**

| 张量 | 差异 |
| --- | --- |
| `routed_sum_f32`（routed 专家的 FP32 累加器） | `510/4096` 元素不同，`max_abs=2.627e-04`，`max_rel=3.07e+01`，`rel>1e-3` 有 504 个，`rel>1e-2` 有 440 个 |
| `block_output_hidden_bf16`（该层输出） | `36/16384` 元素不同，最大 `96` 个 bf16 ULP |

输入是逐位相同的（`post_attn_hidden_bf16`、`route_ids_i32`、`route_weights_f32` 全部一致），
它们喂给 routed 专家之后输出就变了。**同样的输入，算出不同的结果，而且差得很远。**

## 对证书的影响

`sm_120` 上不能再用"与官方参考逐位相等"作为证书口径：官方参考在这张卡上每次都不一样，
而且不是舍入级的不一样。README 里那份 `43/43` 层、argmax `33832` 的证书是 `sm_89` 上的
结果，和这件事不冲突，但它也**没法在这台机器上重新生成**。

下一步（还没做）：确定 `sm_120` 上的证书拿什么当真值。

## 还没确定

1. **为什么算错，没有查。** 一个合理猜测是 TileLang 0.1.8 的核按 `sm_100`（数据中心
   Blackwell）生成代码，而消费级 `sm_120` 没有对应的第五代张量核指令，于是静默走了某条
   不正确的路径。这只是猜测，没验证过——要验证得去看生成的 PTX/SASS。
2. **差异从哪个专家开始扩散，没查。** 现在只知道 6 个 routed 专家合起来的累加器不对，
   不知道是某一个专家错还是全部。
3. **`sm_89` 那台机器上是否仍然可复现，没查。** 那份 `33832` 证书是 2026-08 之前在
   12GB RTX 4080 Laptop 上做的。如果那台机器还在，官方参考在那上面是否稳定，值得确认一次。

## 新增工具

- `rust/colibri-cuda/tools/compare_official_run_repeatability.py` —— 比较多次官方运行的
  每层每个字段，报首个分歧点和分歧字段。路由以普通数组存储，不是摘要对象，所以它对比整条
  记录而不是只比摘要，否则会漏掉路由分歧。
- `rust/colibri-cuda/tools/compare_oracle_cutpoints.py` —— 比较两个 cutpoint 目录，按
  manifest 顺序逐张量报差异；F32 给相对误差分布（`rel>1e-6 / 1e-3 / 1e-2` 的个数），
  BF16 给 ULP 距离。
