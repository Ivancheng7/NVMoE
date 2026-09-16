#!/usr/bin/env python3
"""Official DeepSeek-V4-Flash-0731 single-token layers 0..3 oracle.

This is deliberately not a full-43 oracle.  It imports the pinned upstream
``inference/model.py``, starts from the real token embedding, and constructs one
official Block at a time.  Only routed experts selected by that Block are mmaped
and copied; the previous Block is released before the next one is constructed.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path


REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
MODEL_BASENAME = f"deepseek-v4-flash-0731-{REVISION}"
SCHEMA = "colibri-v4-flash-0731-official-prefix4-v1"
FULL43_SCHEMA = "colibri-v4-flash-0731-official-full43-head-v1"
CUTPOINT_SCHEMA = "colibri-v4-flash-0731-official-l0-cutpoints-v1"
SHARD_COUNT = 48
SHARD_FILE_TOTAL_BYTES = 166_886_535_336
TENSOR_PAYLOAD_TOTAL_BYTES = 166_878_536_440
PINNED_SHA256 = {
    "config.json": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
    "model.safetensors.index.json": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
    "inference/model.py": "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f",
    "inference/kernel.py": "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2",
}
RATIOS = (
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
)
PREFIX = (
    {"layer": 0, "compression_ratio": 0, "router": "hash", "shard": 2},
    {"layer": 1, "compression_ratio": 0, "router": "hash", "shard": 3},
    {"layer": 2, "compression_ratio": 4, "router": "hash", "shard": 4},
    {"layer": 3, "compression_ratio": 128, "router": "learned", "shard": 5},
)


def execution_plan(end_layer=3):
    if not 0 <= end_layer < len(RATIOS):
        raise ValueError(f"end layer must be in 0..{len(RATIOS) - 1}")
    return tuple({
        "layer": layer,
        "compression_ratio": RATIOS[layer],
        "router": "hash" if layer < 3 else "learned",
        "shard": layer + 2,
    } for layer in range(end_layer + 1))


def run_manifest(end_layer=3, final_head=False):
    if final_head and end_layer != 42:
        raise ValueError("--final-head requires --end-layer 42")
    if end_layer == 3 and not final_head:
        return static_manifest()
    plan = execution_plan(end_layer)
    manifest = static_manifest()
    manifest.update({
        "schema": FULL43_SCHEMA if final_head else SCHEMA,
        "scope": f"single token, official base layers 0..{end_layer}"
        + (" plus official final head" if final_head else " only"),
        "full43_oracle": end_layer == 42,
        "layers": list(plan),
        "final_head": final_head,
    })
    return manifest
DTYPE_BYTES = {
    "BOOL": 1, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E8M0": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}
EXPECTED_CONFIG = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "vocab_size": 129280,
    "num_hidden_layers": 43,
    "num_attention_heads": 64,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
    "num_hash_layers": 3,
    "index_topk": 512,
    "routed_scaling_factor": 1.5,
    "sliding_window": 128,
    "compress_rope_theta": 160000,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
}
CUTPOINT_DTYPE_BYTES = {"U8": 1, "BF16": 2, "F32": 4, "I32": 4, "I64": 8}
L0_CUTPOINT_SEMANTICS = {
    "input_hidden_bf16": "Block input with four HC lanes",
    "attn_norm_bf16": "attention RMSNorm output",
    "q_a_linear_bf16": "official FP8 wq_a GEMM output; official output dtype is BF16",
    "q_a_norm_bf16": "q_norm output",
    "q_b_logical_bf16": "official FP8 wq_b GEMM output before head RMSNorm and RoPE",
    "wkv_linear_bf16": "official FP8 wkv GEMM output",
    "kv_norm_bf16": "kv_norm output before RoPE and in-place FP8 simulation",
    "q_rope_bf16": "per-head query after head RMSNorm and RoPE",
    "k_rope_bf16": "shared latent KV row after RoPE and before in-place FP8 simulation",
    "cache_kv_row_bf16": "one active logical shared latent KV row, not the full 128-row ring",
    "attn_scores_f32": "raw Q dot KV scores before softmax_scale, indexed [batch,token,head,topk_slot]",
    "attn_probs_f32": "exp(score*softmax_scale-score_max) before sink denominator normalization",
    "attn_context_bf16": "sparse attention output after inverse RoPE; logical wo_a input",
    "wo_a_logical_bf16": "grouped wo_a output before flattening",
    "wo_b_hidden_bf16": "official FP8 wo_b GEMM output",
    "post_attn_hidden_bf16": "first HC post output",
    "router_pre_score_f32": "gate projection logits before sqrt-softplus",
    "router_post_score_f32": "sqrt-softplus gate scores before selection; actual Tensor.sqrt return in HC debug mode, formula fallback otherwise",
    "route_ids_i32": "selected expert IDs in official slot order",
    "route_weights_f32": "normalized and route-scaled selected expert weights",
    "routed_sum_f32": "actual FP32 official expert-ID-order sum of routed expert contributions before shared expert",
    "shared_out_bf16": "actual shared expert BF16 output before FP32 MoE addition",
    "block_output_hidden_bf16": "L0 Block output after FFN HC post",
}
L0_CUTPOINTS = tuple(L0_CUTPOINT_SEMANTICS)
L0_HC_DEBUG_FLAT_INDEX = 16102
L0_HC_DEBUG_CUTPOINT_SEMANTICS = {
    "attn_hc_rsqrt_input_f32": "actual FP32 x.square().mean(-1, keepdim=True) + norm_eps argument passed to torch.rsqrt in L0 attention hc_pre",
    "attn_hc_rsqrt_f32": "actual torch.rsqrt output used by L0 attention hc_pre",
    "attn_hc_raw_dot_f32": "actual torch.nn.functional.linear output before multiplication by rsqrt",
    "attn_hc_scaled_mixes_f32": "actual scaled mixes observed at the official TileLang hc_split_sinkhorn input boundary",
    "attn_hc_mixes_f32": "actual scaled mixes passed to official TileLang hc_split_sinkhorn; legacy comparison alias",
    "attn_hc_scale_f32": "actual FP32 hc_scale passed to official TileLang hc_split_sinkhorn",
    "attn_hc_base_f32": "actual FP32 hc_base passed to official TileLang hc_split_sinkhorn",
    "attn_hc_pre_tilelang_f32": "actual pre returned by official TileLang hc_split_sinkhorn",
    "attn_hc_post_tilelang_f32": "actual post returned by official TileLang hc_split_sinkhorn",
    "attn_hc_comb_tilelang_f32": "actual final comb returned by official TileLang hc_split_sinkhorn",
    "attn_hc_sinkhorn_replay_trace_f32": "scalar replay stages [raw logits, row softmax+eps, col0, row1, col1, ..., row19, col19]; not a kernel hook",
    "post_attn_flat16102_branch_bf16": "actual torch hc_post branch input at output lane 3, hidden column 3814",
    "post_attn_flat16102_residual_bf16": "actual torch hc_post residual inputs in source-lane order at hidden column 3814",
    "post_attn_flat16102_branch_coeff_f32": "actual torch hc_post post coefficient for output lane 3",
    "post_attn_flat16102_residual_coeff_f32": "actual torch hc_post comb[source_lane, output_lane=3] coefficients",
    "post_attn_flat16102_branch_product_f32": "FP32 scalar replay of branch coefficient times BF16 branch value",
    "post_attn_flat16102_residual_products_f32": "FP32 scalar replay of each residual coefficient times BF16 residual value",
    "post_attn_flat16102_partial_sums_f32": "FP32 scalar replay [p0+p1, p2+p3, residual tree sum, branch+residual]",
    "post_attn_flat16102_replay_output_bf16": "BF16 RNE cast of the scalar replay total; not a torch internal hook",
    "post_attn_flat16102_torch_output_bf16": "actual BF16 output returned by official torch hc_post expression",
    "ffn_hc_collapsed_bf16": "actual collapsed branch returned by the second L0 Block.hc_pre call, immediately before ffn_norm",
    "router_input_bf16": "actual L0 ffn_norm output passed to the routed FFN gate",
    "router_gate_logits_f32": "actual torch.nn.functional.linear return from L0 Gate.forward before sqrt-softplus",
    "selected_scores_f32": "actual Tensor.gather return from L0 Gate.forward in final route slot order before normalization",
    "route_denom_f32": "same-input FP32 serial source-order replay of the inline route normalization denominator; not an official hook",
    "expert_input_fp8_e4m3": "raw U8 storage bytes of actual FP8-E4M3 expert input returned by official act_quant, in final route slot order",
    "expert_input_scale_e8m0": "raw U8 storage bytes of actual E8M0 expert-input scales returned by official act_quant, in final route slot order",
    "routed_w1_bf16": "actual BF16 routed expert w1 module returns in final route slot order",
    "routed_w3_bf16": "actual BF16 routed expert w3 module returns in final route slot order",
    "swiglu_bf16": "actual BF16 w2 module inputs after SwiGLU and route-weight multiplication, in final route slot order",
    "w2_input_fp8_e4m3": "raw U8 storage bytes of actual FP8-E4M3 w2 input returned by official act_quant, in final route slot order",
    "w2_input_scale_e8m0": "raw U8 storage bytes of actual E8M0 w2-input scales returned by official act_quant, in final route slot order",
    "routed_expert_out_bf16": "actual BF16 routed expert returns, already route-weighted, in final route slot order",
    "moe_out_bf16": "actual BF16 StreamingMoE return after FP32 routed-plus-shared addition",
    "ffn_hc_post_f32": "actual post returned by the second L0 official TileLang hc_split_sinkhorn call",
    "ffn_hc_comb_f32": "actual comb returned by the second L0 official TileLang hc_split_sinkhorn call",
}
L0_HC_DEBUG_CUTPOINTS = tuple(L0_HC_DEBUG_CUTPOINT_SEMANTICS)
ROUTER_TRACE_CUTPOINT_SEMANTICS = {
    "router_input_bf16": "actual ffn_norm return passed to the traced routed-FFN gate",
    "router_gate_logits_f32": "actual torch.nn.functional.linear return in traced Gate.forward",
    "router_softplus_f32": "actual torch.nn.functional.softplus return in traced Gate.forward",
    "router_post_score_f32": "actual Tensor.sqrt return in traced Gate.forward",
    "selected_scores_f32": "actual Tensor.gather return in final route slot order before normalization",
    "route_denom_f32": "actual Tensor.sum return used by the inline Gate.forward normalization",
    "route_denom_steps_f32": "same-input serial source-order replay prefixes; diagnostic only, not an official reduction hook",
    "route_ids_i32": "actual selected expert IDs in official slot order",
    "route_weights_f32": "actual normalized and route-scaled selected expert weights",
}
ROUTER_TRACE_CUTPOINTS = tuple(ROUTER_TRACE_CUTPOINT_SEMANTICS)
ROUTER_TRACE_RMSNORM_CUTPOINT_SEMANTICS = {
    "ffn_hc_collapsed_bf16": "actual collapsed branch returned by the second Block.hc_pre call, immediately before ffn_norm",
    "ffn_hc_pre_for_hidden124_f32": "four actual TileLang pre coefficients returned during the second Block.hc_pre call; these are the coefficients used to collapse hidden column 124",
    "router_rmsnorm_square_f32": "FFN RMSNorm FP32 square buffer from the traced official RMSNorm execution",
    "router_rmsnorm_mean_f32": "FFN RMSNorm FP32 mean of the traced official square buffer",
    "router_rmsnorm_inv_f32": "FFN RMSNorm FP32 torch.rsqrt(mean + eps) in the traced official execution",
}
ROUTER_TRACE_RMSNORM_CUTPOINTS = (
    tuple(ROUTER_TRACE_RMSNORM_CUTPOINT_SEMANTICS) + ROUTER_TRACE_CUTPOINTS
)
L2_COMPRESSOR_PRECOMMIT_CUTPOINT_SEMANTICS = {
    "main_compressor_kv_precommit_f32": "L2 main Compressor FP32 wkv projection immediately before its start_pos=0 state write",
    "main_compressor_score_precommit_f32": "L2 main Compressor FP32 wgate plus APE immediately before its start_pos=0 state write",
    "indexer_compressor_kv_precommit_f32": "L2 indexer Compressor FP32 wkv projection immediately before its start_pos=0 state write",
    "indexer_compressor_score_precommit_f32": "L2 indexer Compressor FP32 wgate plus APE immediately before its start_pos=0 state write",
}
L2_COMPRESSOR_PRECOMMIT_CUTPOINTS = tuple(L2_COMPRESSOR_PRECOMMIT_CUTPOINT_SEMANTICS)
CUTPOINT_SEMANTICS = {
    **L0_CUTPOINT_SEMANTICS,
    **L0_HC_DEBUG_CUTPOINT_SEMANTICS,
    **ROUTER_TRACE_CUTPOINT_SEMANTICS,
    **ROUTER_TRACE_RMSNORM_CUTPOINT_SEMANTICS,
    **L2_COMPRESSOR_PRECOMMIT_CUTPOINT_SEMANTICS,
}


class CutpointWriter:
    def __init__(self, directory, token_id, hc_debug_trace=False, router_trace_layer=None,
                 router_trace_rmsnorm=False):
        if sys.byteorder != "little":
            raise RuntimeError("cutpoint dumps require a little-endian host")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.token_id = token_id
        self.hc_debug_trace = hc_debug_trace
        self.router_trace_layer = router_trace_layer
        self.router_trace_rmsnorm = router_trace_rmsnorm
        self.tensors = {}

    def write_raw(self, name, data, dtype, shape, semantic="test tensor"):
        if name in self.tensors:
            raise RuntimeError(f"duplicate cutpoint: {name}")
        if dtype not in CUTPOINT_DTYPE_BYTES:
            raise RuntimeError(f"unsupported cutpoint dtype: {dtype}")
        shape = list(shape)
        elements = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                raise RuntimeError(f"invalid cutpoint shape: {name}")
            elements *= dimension
        if len(data) != elements * CUTPOINT_DTYPE_BYTES[dtype]:
            raise RuntimeError(f"cutpoint byte length mismatch: {name}")
        filename = f"{name}.bin"
        (self.directory / filename).write_bytes(data)
        self.tensors[name] = {
            "file": filename, "dtype": dtype, "shape": shape,
            "layout": "logical-row-major-contiguous", "byte_order": "little",
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "semantic": semantic,
        }

    def write_tensor(self, name, tensor, torch):
        tensor = tensor.detach().cpu().contiguous()
        dtypes = {
            torch.bfloat16: "BF16", torch.float32: "F32",
            torch.int32: "I32", torch.int64: "I64",
        }
        try:
            dtype = dtypes[tensor.dtype]
        except KeyError as error:
            raise RuntimeError(f"unsupported cutpoint tensor dtype: {tensor.dtype}") from error
        self.write_raw(
            name, raw_bytes(tensor, torch), dtype, tensor.shape,
            CUTPOINT_SEMANTICS[name],
        )

    def write_storage_bytes(self, name, tensor, expected_dtype, torch):
        tensor = tensor.detach().cpu().contiguous()
        if tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"cutpoint storage dtype mismatch: {name}: {tensor.dtype} != {expected_dtype}"
            )
        self.write_raw(
            name, raw_bytes(tensor, torch), "U8", tensor.shape,
            CUTPOINT_SEMANTICS[name],
        )
        self.tensors[name]["source_tensor_dtype"] = str(tensor.dtype).removeprefix("torch.")
        self.tensors[name]["storage_encoding"] = "raw-byte-view-of-actual-returned-tensor"

    def finish(self, required=()):
        missing = set(required) - set(self.tensors)
        extra = set(self.tensors) - set(required) if required else set()
        if missing or extra:
            raise RuntimeError(f"cutpoint set mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        manifest = {
            "schema": CUTPOINT_SCHEMA, "model_revision": REVISION,
            "scope": f"token0, official layer {self.router_trace_layer if self.router_trace_layer is not None else 0} only",
            "token_id": self.token_id,
            "layer": self.router_trace_layer if self.router_trace_layer is not None else 0,
            "tensor_order": list(self.tensors), "tensors": self.tensors,
            "kv_contract": "Flash uses one shared latent KV row for both score and value; no separate K/V arrays",
        }
        if self.hc_debug_trace:
            manifest["hc_debug_trace"] = {
                "enabled": True,
                "flat_index": L0_HC_DEBUG_FLAT_INDEX,
                "flat_index_coordinate": {
                    "batch": 0, "token": 0, "output_lane": 3, "hidden_column": 3814,
                },
                "official_boundary": {
                    "sinkhorn_input": [
                        "attn_hc_scaled_mixes_f32", "attn_hc_scale_f32",
                        "attn_hc_base_f32",
                    ],
                    "tilelang_outputs": [
                        "attn_hc_pre_tilelang_f32", "attn_hc_post_tilelang_f32",
                        "attn_hc_comb_tilelang_f32",
                    ],
                    "torch_hc_post_inputs": [
                        "post_attn_flat16102_branch_bf16",
                        "post_attn_flat16102_residual_bf16",
                        "post_attn_flat16102_branch_coeff_f32",
                        "post_attn_flat16102_residual_coeff_f32",
                    ],
                    "torch_hc_post_output": "post_attn_flat16102_torch_output_bf16",
                },
                "hc_linear_actual_hooks": {
                    "kind": "actual-official-torch-boundary-hooks",
                    "replay": False,
                    "raw_dot": {
                        "tensor": "attn_hc_raw_dot_f32",
                        "boundary": "return value of the actual torch.nn.functional.linear call in pinned Block.hc_pre",
                    },
                    "rsqrt": {
                        "tensor": "attn_hc_rsqrt_f32",
                        "input_tensor": "attn_hc_rsqrt_input_f32",
                        "input_formula": "x.square().mean(-1, keepdim=True) + norm_eps",
                        "input_boundary": "actual FP32 argument received by the wrapped torch.rsqrt call; not reconstructed",
                        "boundary": "return value of the actual torch.rsqrt call in pinned Block.hc_pre",
                    },
                    "scaled_mixes": {
                        "tensor": "attn_hc_scaled_mixes_f32",
                        "legacy_alias": "attn_hc_mixes_f32",
                        "boundary": "actual mixes argument received by the official hc_split_sinkhorn wrapper after torch multiplication",
                    },
                    "formula": "scaled_mixes = raw_dot * rsqrt",
                },
                "sinkhorn_trace": {
                    "kind": "official-source-equivalent-scalar-replay",
                    "kernel_hook": False,
                    "reason": "official TileLang kernel returns only final pre/post/comb and does not expose internal stages",
                    "source": "pinned inference/kernel.py hc_split_sinkhorn_kernel",
                    "shape": [1, 1, 41, 4, 4],
                    "rust_comparison_shape": [1, 41, 4, 4],
                    "shape_qualification": "contiguous reshape of batch and token into Rust rows only",
                    "stage_order": "raw logits, row softmax+eps, initial column norm, then 19 row/column norm pairs",
                    "sum4_order": "(lane0 + lane1) + (lane2 + lane3)",
                    "exp_boundary": "CPU torch.exp FP32 replay of TileLang T.exp formula; no claim of hidden TileLang exp instruction bits",
                },
                "post_attn_scalar_trace": {
                    "kind": "same-input-fp32-scalar-replay",
                    "torch_internal_hook": False,
                    "reason": "torch.sum does not expose products or internal partial reductions",
                    "source": "pinned inference/model.py Block.hc_post",
                    "comb_orientation": "comb[source_lane, output_lane]",
                    "sum4_order": "(p0 + p1) + (p2 + p3)",
                },
                "ffn_router_actual_hooks": {
                    "kind": "actual-official-torch-boundary-hooks",
                    "replay": False,
                    "ffn_hc_collapsed": {
                        "tensor": "ffn_hc_collapsed_bf16",
                        "boundary": "collapsed branch returned by the second actual L0 Block.hc_pre call",
                    },
                    "router_input": {
                        "tensor": "router_input_bf16",
                        "boundary": "return value of the actual L0 ffn_norm module call",
                    },
                    "gate_logits": {
                        "tensor": "router_gate_logits_f32",
                        "boundary": "return value of the actual torch.nn.functional.linear call in L0 Gate.forward",
                    },
                    "post_score": {
                        "tensor": "router_post_score_f32",
                        "boundary": "return value of the actual Tensor.sqrt call in L0 Gate.forward",
                    },
                    "selected_scores": {
                        "tensor": "selected_scores_f32",
                        "boundary": "return value of the actual Tensor.gather call in L0 Gate.forward",
                        "order": "final route slot order from the actual gather index",
                    },
                },
                "route_denom_trace": {
                    "tensor": "route_denom_f32",
                    "kind": "same-input-fp32-serial-source-order-replay",
                    "official_sum_hook": False,
                    "source_tensor": "selected_scores_f32 actual Tensor.gather return",
                    "source": "pinned inference/model.py Gate.forward inline weights.sum(dim=-1, keepdim=True)",
                    "reason": "official Gate.forward does not retain or return the inline denominator",
                    "sum_order": "slot0 + slot1 + slot2 + slot3 + slot4 + slot5",
                },
                "fp4_ffn_actual_hooks": {
                    "kind": "actual-official-torch-boundary-hooks",
                    "replay": False,
                    "route_axis": {
                        "order": "final route slot order",
                        "ids_tensor": "route_ids_i32",
                        "shape_prefix": [1, 6],
                    },
                    "expert_input_quant": {
                        "value_tensor": "expert_input_fp8_e4m3",
                        "scale_tensor": "expert_input_scale_e8m0",
                        "boundary": "actual tensors returned by pinned kernel.py act_quant for routed w1 and w3",
                        "storage": "raw U8 byte views of FP8-E4M3 values and E8M0 scales",
                        "repeated_calls": "w1 and w3 actual returns are required bit-exact; one canonical copy is stored",
                    },
                    "routed_w1": {
                        "tensor": "routed_w1_bf16",
                        "boundary": "actual routed Expert.w1 module return",
                    },
                    "routed_w3": {
                        "tensor": "routed_w3_bf16",
                        "boundary": "actual routed Expert.w3 module return",
                    },
                    "swiglu": {
                        "tensor": "swiglu_bf16",
                        "boundary": "actual routed Expert.w2 module input after SwiGLU, route-weight multiplication, and BF16 cast",
                        "route_weighted": True,
                    },
                    "w2_input_quant": {
                        "value_tensor": "w2_input_fp8_e4m3",
                        "scale_tensor": "w2_input_scale_e8m0",
                        "boundary": "actual tensors returned by pinned kernel.py act_quant inside routed Expert.w2",
                        "storage": "raw U8 byte views of FP8-E4M3 values and E8M0 scales",
                    },
                    "routed_expert_out": {
                        "tensor": "routed_expert_out_bf16",
                        "boundary": "actual routed Expert return, equal to its w2 module return",
                        "route_weighted": True,
                    },
                    "routed_sum": {
                        "tensor": "routed_sum_f32",
                        "boundary": "actual FP32 StreamingMoE accumulator before shared expert addition",
                        "sum_order": "pinned official expert-ID iteration order",
                    },
                    "shared_out": {
                        "tensor": "shared_out_bf16",
                        "boundary": "actual shared Expert return before FP32 MoE addition",
                    },
                    "moe_out": {
                        "tensor": "moe_out_bf16",
                        "boundary": "actual BF16 StreamingMoE return after routed-plus-shared FP32 addition",
                    },
                    "ffn_hc_coefficients": {
                        "post_tensor": "ffn_hc_post_f32",
                        "comb_tensor": "ffn_hc_comb_f32",
                        "boundary": "actual post and comb returned by the second official L0 hc_split_sinkhorn call",
                    },
                    "unavailable_requested_physical_intermediates": [],
                },
            }
        if self.router_trace_rmsnorm:
            manifest["router_rmsnorm_trace"] = {
                "enabled": True,
                "input": {
                    "tensor": "ffn_hc_collapsed_bf16",
                    "boundary": "collapsed branch returned by the second actual Block.hc_pre call",
                },
                "execution": "verbatim pinned RMSNorm.forward operations on that actual input",
                "outputs": [
                    "router_rmsnorm_square_f32",
                    "router_rmsnorm_mean_f32",
                    "router_rmsnorm_inv_f32",
                    "router_input_bf16",
                ],
            }
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


def shard_name(number):
    return f"model-{number:05}-of-{SHARD_COUNT:05}.safetensors"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def rss():
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator and key in ("VmRSS", "VmHWM"):
            fields[f"{key.lower()}_kib"] = int(value.split()[0])
    fields["ru_maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return fields


def storage_identity(model_dir):
    best = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        mountpoint = left.split()[4]
        filesystem, source = right.split()[:2]
        try:
            model_dir.relative_to(mountpoint)
        except ValueError:
            continue
        if best is None or len(mountpoint) > len(best[0]):
            best = (mountpoint, filesystem, source)
    if best is None or best[1] != "ext4" or not Path(best[2]).is_block_device():
        raise RuntimeError(f"oracle requires a native ext4 block device, got {best}")
    return {"mountpoint": best[0], "filesystem": best[1], "source": best[2]}


def read_safetensors_header(path):
    size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"short safetensors prefix: {path.name}")
        header_len = int.from_bytes(prefix, "little")
        if header_len <= 1 or header_len > size - 8:
            raise RuntimeError(f"invalid safetensors header length: {path.name}")
        header = json.loads(source.read(header_len))
    tensors = {}
    intervals = []
    payload_sum = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"invalid safetensors metadata: {path.name}:{name}")
        elements = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                raise RuntimeError(f"invalid safetensors shape: {path.name}:{name}")
            elements *= dimension
        start, end = offsets
        byte_len = elements * DTYPE_BYTES[dtype]
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end - start != byte_len:
            raise RuntimeError(f"invalid safetensors range: {path.name}:{name}")
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
        intervals.append((start, end, name))
        payload_sum += byte_len
    intervals.sort()
    if not intervals or intervals[0][0] != 0:
        raise RuntimeError(f"safetensors payload does not begin at zero: {path.name}")
    for left, right in zip(intervals, intervals[1:]):
        if left[1] != right[0]:
            raise RuntimeError(f"safetensors payload has overlap/gap: {path.name}")
    payload_end = intervals[-1][1]
    if payload_sum != payload_end or size != 8 + header_len + payload_end:
        raise RuntimeError(f"safetensors file length mismatch: {path.name}")
    return tensors, payload_sum, header_len


def require_tensor(headers, weight_map, name, shard, dtype, shape):
    expected_shard = shard_name(shard)
    if weight_map.get(name) != expected_shard:
        raise RuntimeError(f"index mapping mismatch: {name}")
    actual = headers[expected_shard].get(name)
    if actual is None or actual["dtype"] != dtype or actual["shape"] != list(shape):
        raise RuntimeError(f"tensor shape/dtype mismatch: {name}")


def validate_config(config):
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise RuntimeError(f"pinned config mismatch: {key}")
    if tuple(config.get("compress_ratios", ())) != RATIOS:
        raise RuntimeError("pinned compression schedule mismatch")


def validate_checkpoint(model_dir):
    model_dir = model_dir.resolve(strict=True)
    if model_dir.name != MODEL_BASENAME or str(model_dir).startswith(("/mnt/c/", "/mnt/d/")):
        raise RuntimeError("oracle model directory is not the pinned native-ext4 snapshot")
    storage = storage_identity(model_dir)
    for relative, expected in PINNED_SHA256.items():
        path = model_dir / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            raise RuntimeError(f"pinned source hash mismatch: {relative}")
    config = json.loads((model_dir / "config.json").read_text())
    validate_config(config)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    if index.get("metadata", {}).get("total_size") != TENSOR_PAYLOAD_TOTAL_BYTES:
        raise RuntimeError("index tensor payload total changed")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("checkpoint index lacks a weight_map")

    headers, seen = {}, set()
    payload_total = header_total = shard_total = 0
    for number in range(1, SHARD_COUNT + 1):
        name = shard_name(number)
        path = model_dir / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular pinned shard: {name}")
        shard_total += path.stat().st_size
        header, payload_bytes, header_bytes = read_safetensors_header(path)
        for tensor_name in header:
            if tensor_name in seen or weight_map.get(tensor_name) != name:
                raise RuntimeError(f"duplicate or misindexed tensor: {tensor_name}")
            seen.add(tensor_name)
        headers[name] = header
        payload_total += payload_bytes
        header_total += 8 + header_bytes
    if shard_total != SHARD_FILE_TOTAL_BYTES:
        raise RuntimeError(f"48-shard stat total changed: {shard_total}")
    if payload_total != TENSOR_PAYLOAD_TOTAL_BYTES:
        raise RuntimeError(f"safetensors payload total changed: {payload_total}")
    if seen != set(weight_map):
        raise RuntimeError("index and shard header tensor sets differ")

    require_tensor(headers, weight_map, "embed.weight", 1, "BF16", (129280, 4096))
    for layer in range(43):
        prefix = f"layers.{layer}"
        require_tensor(headers, weight_map, f"{prefix}.attn.wq_a.weight", layer + 2, "F8_E4M3", (1024, 4096))
        require_tensor(headers, weight_map, f"{prefix}.ffn.gate.weight", layer + 2, "BF16", (256, 4096))
    require_tensor(headers, weight_map, "layers.0.ffn.gate.tid2eid", 2, "I64", (129280, 6))
    require_tensor(headers, weight_map, "layers.2.attn.compressor.ape", 4, "F32", (4, 1024))
    require_tensor(headers, weight_map, "layers.2.attn.indexer.wq_b.weight", 4, "F8_E4M3", (8192, 1024))
    require_tensor(headers, weight_map, "layers.2.attn.indexer.compressor.ape", 4, "F32", (4, 256))
    require_tensor(headers, weight_map, "layers.3.attn.compressor.ape", 5, "F32", (128, 512))
    require_tensor(headers, weight_map, "layers.3.ffn.gate.bias", 5, "F32", (256,))
    require_tensor(headers, weight_map, "hc_head_base", 45, "F32", (4,))
    require_tensor(headers, weight_map, "hc_head_fn", 45, "F32", (4, 16384))
    require_tensor(headers, weight_map, "hc_head_scale", 45, "F32", (1,))
    require_tensor(headers, weight_map, "norm.weight", 45, "BF16", (4096,))
    require_tensor(headers, weight_map, "head.weight", 45, "BF16", (129280, 4096))

    source_ownership = []
    for layer in range(4):
        layer_prefix = f"layers.{layer}."
        layer_tensors = {
            name: info for name, info in headers[shard_name(layer + 2)].items()
            if name.startswith(layer_prefix)
        }
        sizes = {}
        for name, info in layer_tensors.items():
            byte_len = DTYPE_BYTES[info["dtype"]]
            for dimension in info["shape"]:
                byte_len *= dimension
            sizes[name] = byte_len
        expert_sizes = []
        for expert in range(256):
            marker = f".ffn.experts.{expert}."
            expert_sizes.append(sum(size for name, size in sizes.items() if marker in name))
        if len(set(expert_sizes)) != 1 or expert_sizes[0] == 0:
            raise RuntimeError(f"layer {layer} routed expert storage geometry changed")
        fixed_bytes = sum(size for name, size in sizes.items() if ".ffn.experts." not in name)
        source_ownership.append({
            "layer": layer,
            "fixed_parameter_source_bytes": fixed_bytes,
            "one_routed_expert_source_bytes": expert_sizes[0],
            "maximum_live_parameter_source_bytes": fixed_bytes + expert_sizes[0],
        })
    return model_dir, config, {
        "storage": storage,
        "regular_shards": SHARD_COUNT,
        "shard_file_stat_total_bytes": shard_total,
        "safetensors_header_total_bytes": header_total,
        "tensor_payload_total_bytes": payload_total,
        "tensor_count": len(seen),
        "prefix_streaming_source_ownership": source_ownership,
    }


def static_manifest():
    return {
        "schema": SCHEMA,
        "model_revision": REVISION,
        "scope": "single token, official base layers 0..3 only",
        "full43_oracle": False,
        "layers": list(PREFIX),
        "shard_count": SHARD_COUNT,
        "shard_file_stat_total_bytes": SHARD_FILE_TOTAL_BYTES,
        "index_tensor_payload_total_bytes": TENSOR_PAYLOAD_TOTAL_BYTES,
        "source_sha256": PINNED_SHA256,
        "value_encoding": "BF16 and FP32 tensors are emitted as exact little-endian scalar bits",
    }


def validate_resources(torch):
    version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
    if version < (2, 10) or not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError(f"official oracle requires PyTorch >=2.10 with FP4, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("official prefix oracle requires CUDA PyTorch")
    # TileLang decides which kernel gets generated, and sm_120 needs a version
    # whose codegen is correct there, so the version is recorded rather than
    # pinned.  It was pinned to 0.1.8 while that was thought to be universal.
    tilelang_version = importlib.metadata.version("tilelang")
    safetensors_version = importlib.metadata.version("safetensors")
    if tuple(int(part) for part in safetensors_version.split(".")[:2]) < (0, 7):
        raise RuntimeError(f"official prefix oracle requires safetensors>=0.7, got {safetensors_version}")
    __import__("fast_hadamard_transform")
    free_vram, _ = torch.cuda.mem_get_info()
    available_kib = next(
        int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if free_vram < 8 * (1 << 30):
        raise RuntimeError(f"official prefix oracle requires 8 GiB free VRAM, got {free_vram}")
    if available_kib < 8 * (1 << 20):
        raise RuntimeError(f"official prefix oracle requires 8 GiB available RAM, got {available_kib} KiB")
    return {
        "free_vram_bytes": free_vram,
        "available_ram_kib": available_kib,
        "torch": torch.__version__,
        "tilelang": tilelang_version,
        "safetensors": safetensors_version,
        "free_vram_start_threshold_bytes": 8 * (1 << 30),
        "threshold_scope": "conservative preflight, not measured allocator peak",
    }


def import_official(model_dir):
    inference = model_dir / "inference"
    sys.path.insert(0, str(inference))
    spec = importlib.util.spec_from_file_location("colibri_pinned_v4_flash_0731", inference / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def official_args(official, config):
    rope = config["rope_scaling"]
    values = {
        "max_batch_size": 1, "max_seq_len": 128, "dtype": "fp8", "scale_fmt": "ue8m0",
        "expert_dtype": "fp4", "scale_dtype": "fp8", "vocab_size": config["vocab_size"],
        "dim": config["hidden_size"], "moe_inter_dim": config["moe_intermediate_size"],
        "n_layers": config["num_hidden_layers"], "n_hash_layers": config["num_hash_layers"],
        "n_mtp_layers": config["num_nextn_predict_layers"], "n_heads": config["num_attention_heads"],
        "n_routed_experts": config["n_routed_experts"], "n_shared_experts": config["n_shared_experts"],
        "n_activated_experts": config["num_experts_per_tok"], "score_func": config["scoring_func"],
        "route_scale": config["routed_scaling_factor"], "swiglu_limit": config["swiglu_limit"],
        "q_lora_rank": config["q_lora_rank"], "head_dim": config["head_dim"],
        "rope_head_dim": config["qk_rope_head_dim"], "norm_eps": config["rms_norm_eps"],
        "o_groups": config["o_groups"], "o_lora_rank": config["o_lora_rank"],
        "window_size": config["sliding_window"], "compress_ratios": tuple(config["compress_ratios"]),
        "compress_rope_theta": config["compress_rope_theta"],
        "original_seq_len": rope["original_max_position_embeddings"], "rope_theta": config["rope_theta"],
        "rope_factor": rope["factor"], "beta_fast": rope["beta_fast"], "beta_slow": rope["beta_slow"],
        "index_n_heads": config["index_n_heads"], "index_head_dim": config["index_head_dim"],
        "index_topk": config["index_topk"], "hc_mult": config["hc_mult"],
        "hc_sinkhorn_iters": config["hc_sinkhorn_iters"], "hc_eps": config["hc_eps"],
    }
    allowed = {field.name for field in dataclasses.fields(official.ModelArgs)}
    return official.ModelArgs(**{key: value for key, value in values.items() if key in allowed})


def copy_parameters(module, shard, prefix, torch, safe_open):
    source_bytes = 0
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        for name, parameter in module.named_parameters():
            tensor_name = f"{prefix}.{name}" if prefix else name
            if tensor_name not in keys:
                raise RuntimeError(f"official parameter missing from shard: {tensor_name}")
            value = tensors.get_tensor(tensor_name)
            source_bytes += value.numel() * value.element_size()
            if parameter.dtype == torch.float4_e2m1fn_x2 and value.dtype in (torch.int8, torch.uint8):
                parameter.data = value.view(torch.float4_e2m1fn_x2).to(parameter.device)
            else:
                parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            del value
    return source_bytes


def make_streaming_moe(official, torch, safe_open, cutpoints=None, audit_state=None):
    class StreamingMoE(torch.nn.Module):
        def __init__(self, layer_id, args):
            super().__init__()
            self.layer_id, self.args = layer_id, args
            self.gate = official.Gate(layer_id, args)
            self.shared_experts = official.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
            self.shard = None
            self.last_ids = self.last_weights = None
            self.source_range_bytes = 0

        def forward(self, x, input_ids):
            shape = x.size()
            flat = x.view(-1, self.args.dim)
            weights, indices = self.gate(flat, input_ids.flatten())
            router_trace = (
                cutpoints is not None and self.layer_id == audit_state["trace_layer"]
                and audit_state["active"] and audit_state["router_trace"]
            )
            fp4_debug = (
                cutpoints is not None and self.layer_id == audit_state["trace_layer"]
                and audit_state["active"] and audit_state["hc_debug_trace"]
                and not audit_state["router_trace"]
            )
            expert_records = {}
            self.last_ids, self.last_weights = indices.detach().cpu(), weights.detach().cpu()
            if router_trace:
                softplus = audit_state.pop("router_softplus_actual", None)
                post_scores = audit_state.pop("router_post_actual", None)
                selected_scores = audit_state.pop("selected_scores_actual", None)
                selected_indices = audit_state.pop("selected_score_indices", None)
                denominator = audit_state.pop("route_denom_actual", None)
                if (softplus is None or post_scores is None or selected_scores is None
                        or selected_indices is None or denominator is None):
                    raise RuntimeError("actual router score/gather/sum hooks were not captured")
                if not torch.equal(selected_indices.detach().cpu(), indices.detach().cpu()):
                    raise RuntimeError("captured selected-score order differs from final route order")
                cutpoints.write_tensor("router_softplus_f32", softplus, torch)
                cutpoints.write_tensor("router_post_score_f32", post_scores, torch)
                cutpoints.write_tensor("selected_scores_f32", selected_scores, torch)
                cutpoints.write_tensor("route_denom_f32", denominator, torch)
                cutpoints.write_tensor(
                    "route_denom_steps_f32", replay_route_denom_steps(selected_scores, torch), torch,
                )
                cutpoints.write_tensor("route_ids_i32", indices, torch)
                cutpoints.write_tensor("route_weights_f32", weights, torch)
            elif (
                cutpoints is not None and self.layer_id == audit_state["trace_layer"]
                and audit_state["active"]
                and not audit_state["router_trace"]
            ):
                logits = audit_state.pop("router_pre", None)
                if logits is None:
                    raise RuntimeError("L0 gate projection was not captured")
                if audit_state["hc_debug_trace"]:
                    post_scores = audit_state.pop("router_post_actual", None)
                    selected_scores = audit_state.pop("selected_scores_actual", None)
                    selected_indices = audit_state.pop("selected_score_indices", None)
                    if post_scores is None or selected_scores is None or selected_indices is None:
                        raise RuntimeError("actual L0 router sqrt/gather hooks were not captured")
                    if not torch.equal(selected_indices.detach().cpu(), indices.detach().cpu()):
                        raise RuntimeError("captured L0 selected-score order differs from final route order")
                    cutpoints.write_tensor("router_post_score_f32", post_scores, torch)
                    cutpoints.write_tensor("selected_scores_f32", selected_scores, torch)
                    cutpoints.write_tensor(
                        "route_denom_f32", replay_route_denom(selected_scores, torch), torch,
                    )
                else:
                    post_scores = torch.nn.functional.softplus(logits).sqrt()
                    cutpoints.write_tensor("router_post_score_f32", post_scores, torch)
                cutpoints.write_tensor("route_ids_i32", indices, torch)
                cutpoints.write_tensor("route_weights_f32", weights, torch)
            output = torch.zeros_like(flat, dtype=torch.float32)
            for expert_id in sorted(set(indices.flatten().tolist())):
                expert = official.Expert(
                    self.args.dim, self.args.moe_inter_dim, dtype=torch.float4_e2m1fn_x2,
                    swiglu_limit=self.args.swiglu_limit,
                )
                self.source_range_bytes += copy_parameters(
                    expert, self.shard, f"layers.{self.layer_id}.ffn.experts.{expert_id}", torch, safe_open
                )
                row, slot = torch.where(indices == expert_id)
                handles = []
                record = None
                if fp4_debug:
                    if row.numel() != 1 or row.item() != 0:
                        raise RuntimeError("L0 FP4 debug trace requires one route row and unique expert IDs")
                    route_slot = int(slot.item())
                    if route_slot in expert_records:
                        raise RuntimeError("L0 FP4 debug trace route slot was captured twice")
                    record = {"expert_id": expert_id}
                    expert_records[route_slot] = record
                    audit_state["active_expert_record"] = record

                    def linear_pre(name):
                        def capture(_module, inputs):
                            if "active_expert_linear" in audit_state:
                                raise RuntimeError("nested routed expert linear call in FP4 debug trace")
                            audit_state["active_expert_linear"] = name
                            if name == "w2":
                                record["swiglu_bf16"] = inputs[0].detach().clone()
                        return capture

                    def linear_post(name):
                        def capture(_module, _inputs, result):
                            if audit_state.pop("active_expert_linear", None) != name:
                                raise RuntimeError("routed expert linear hook order changed")
                            record[f"{name}_bf16"] = result.detach().clone()
                        return capture

                    for name in ("w1", "w3", "w2"):
                        module = getattr(expert, name)
                        handles.append(module.register_forward_pre_hook(linear_pre(name)))
                        handles.append(module.register_forward_hook(linear_post(name)))
                contribution = expert(flat[row], weights[row, slot, None])
                for handle in handles:
                    handle.remove()
                if fp4_debug:
                    if audit_state.pop("active_expert_record", None) is not record:
                        raise RuntimeError("routed expert FP4 debug record changed during forward")
                    if "active_expert_linear" in audit_state:
                        raise RuntimeError("routed expert linear hook did not close")
                    required_record = {
                        "w1_quant_value", "w1_quant_scale", "w3_quant_value", "w3_quant_scale",
                        "w1_bf16", "w3_bf16", "swiglu_bf16", "w2_quant_value",
                        "w2_quant_scale", "w2_bf16",
                    }
                    missing_record = required_record - set(record)
                    if missing_record:
                        raise RuntimeError(
                            f"routed expert actual hooks missing {sorted(missing_record)}"
                        )
                    if not torch.equal(
                        record["w1_quant_value"].view(torch.uint8),
                        record["w3_quant_value"].view(torch.uint8),
                    ) or not torch.equal(
                        record["w1_quant_scale"].view(torch.uint8),
                        record["w3_quant_scale"].view(torch.uint8),
                    ):
                        raise RuntimeError("actual w1/w3 expert-input quantization returns differ")
                    if not torch.equal(record["w2_bf16"], contribution):
                        raise RuntimeError("routed Expert return differs from its actual w2 return")
                    record["expert_input_fp8"] = record.pop("w1_quant_value")
                    record["expert_input_scale"] = record.pop("w1_quant_scale")
                    del record["w3_quant_value"], record["w3_quant_scale"]
                    record["expert_out_bf16"] = contribution.detach().clone()
                output[row] += contribution
                del expert
            if fp4_debug:
                if set(expert_records) != set(range(self.args.n_activated_experts)):
                    raise RuntimeError("L0 FP4 debug trace did not capture every route slot")
                for route_slot, record in expert_records.items():
                    if record["expert_id"] != int(indices[0, route_slot].item()):
                        raise RuntimeError("L0 FP4 debug trace changed final route slot order")

                def route_stack(key, width):
                    values = []
                    for route_slot in range(self.args.n_activated_experts):
                        value = expert_records[route_slot][key]
                        if tuple(value.shape) != (1, width):
                            raise RuntimeError(
                                f"L0 FP4 debug tensor geometry changed: {key}: {tuple(value.shape)}"
                            )
                        values.append(value[0])
                    return torch.stack(values, dim=0).unsqueeze(0)

                cutpoints.write_storage_bytes(
                    "expert_input_fp8_e4m3",
                    route_stack("expert_input_fp8", self.args.dim),
                    torch.float8_e4m3fn, torch,
                )
                cutpoints.write_storage_bytes(
                    "expert_input_scale_e8m0",
                    route_stack("expert_input_scale", self.args.dim // 128),
                    torch.float8_e8m0fnu, torch,
                )
                cutpoints.write_tensor(
                    "routed_w1_bf16", route_stack("w1_bf16", self.args.moe_inter_dim), torch,
                )
                cutpoints.write_tensor(
                    "routed_w3_bf16", route_stack("w3_bf16", self.args.moe_inter_dim), torch,
                )
                cutpoints.write_tensor(
                    "swiglu_bf16", route_stack("swiglu_bf16", self.args.moe_inter_dim), torch,
                )
                cutpoints.write_storage_bytes(
                    "w2_input_fp8_e4m3",
                    route_stack("w2_quant_value", self.args.moe_inter_dim),
                    torch.float8_e4m3fn, torch,
                )
                cutpoints.write_storage_bytes(
                    "w2_input_scale_e8m0",
                    route_stack("w2_quant_scale", self.args.moe_inter_dim // 128),
                    torch.float8_e8m0fnu, torch,
                )
                cutpoints.write_tensor(
                    "routed_expert_out_bf16",
                    route_stack("expert_out_bf16", self.args.dim), torch,
                )
            shared = self.shared_experts(flat)
            if (
                cutpoints is not None and self.layer_id == audit_state["trace_layer"]
                and audit_state["active"]
                and not audit_state["router_trace"]
            ):
                cutpoints.write_tensor("routed_sum_f32", output, torch)
                cutpoints.write_tensor("shared_out_bf16", shared, torch)
            output += shared
            moe_output = output.type_as(flat)
            if fp4_debug:
                cutpoints.write_tensor("moe_out_bf16", moe_output, torch)
            return moe_output.view(shape)

    return StreamingMoE


def raw_bytes(tensor, torch):
    return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())


def fnv1a(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def tensor_record(tensor, torch):
    tensor = tensor.detach().cpu().contiguous()
    data = raw_bytes(tensor, torch)
    if tensor.dtype == torch.bfloat16:
        dtype, bits = "BF16", [value & 0xFFFF for value in tensor.view(torch.int16).reshape(-1).tolist()]
    elif tensor.dtype == torch.float32:
        dtype, bits = "F32", [value & 0xFFFFFFFF for value in tensor.view(torch.int32).reshape(-1).tolist()]
    elif tensor.dtype in (torch.int32, torch.int64):
        dtype, bits = str(tensor.dtype).removeprefix("torch.").upper(), tensor.reshape(-1).tolist()
    else:
        raise RuntimeError(f"unsupported oracle output dtype: {tensor.dtype}")
    return {
        "dtype": dtype, "shape": list(tensor.shape), "fnv1a64": fnv1a(data),
        "sha256": hashlib.sha256(data).hexdigest(), "values_bits": bits,
    }


def tensor_digest_record(tensor, torch):
    tensor = tensor.detach().cpu().contiguous()
    data = raw_bytes(tensor, torch)
    if tensor.dtype == torch.bfloat16:
        dtype = "BF16"
    elif tensor.dtype == torch.float32:
        dtype = "F32"
    elif tensor.dtype in (torch.int32, torch.int64):
        dtype = str(tensor.dtype).removeprefix("torch.").upper()
    else:
        raise RuntimeError(f"unsupported oracle digest dtype: {tensor.dtype}")
    return {
        "dtype": dtype, "shape": list(tensor.shape), "fnv1a64": fnv1a(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def replay_hc_sinkhorn_trace(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, torch):
    """Replay pinned TileLang Sinkhorn stages from its actual FP32 inputs."""
    if hc_mult != 4 or sinkhorn_iters != 20 or mixes.shape[-1] != 24:
        raise RuntimeError("L0 HC debug trace requires the pinned 4-lane, 20-iteration geometry")
    if tuple(hc_scale.shape) != (3,) or tuple(hc_base.shape) != (24,):
        raise RuntimeError("L0 HC debug trace received invalid scale/base geometry")
    mixes = mixes.detach().cpu().float().contiguous()
    scale = hc_scale.detach().cpu().float().contiguous()
    base = hc_base.detach().cpu().float().contiguous()
    eps_value = mixes.new_tensor(eps)

    def sum4(values, dimension):
        if values.size(dimension) != 4:
            raise RuntimeError("TileLang sum4 replay requires four lanes")
        lanes = values.unbind(dimension)
        return (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])

    comb = (
        mixes[..., 2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult)) * scale[2]
        + base[2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    )
    stages = [comb.clone()]
    row_max = comb.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(comb - row_max)
    comb = exponentials / sum4(exponentials, -1).unsqueeze(-1) + eps_value
    stages.append(comb.clone())
    comb = comb / (sum4(comb, -2).unsqueeze(-2) + eps_value)
    stages.append(comb.clone())
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (sum4(comb, -1).unsqueeze(-1) + eps_value)
        stages.append(comb.clone())
        comb = comb / (sum4(comb, -2).unsqueeze(-2) + eps_value)
        stages.append(comb.clone())
    trace = torch.stack(stages, dim=-3).contiguous()
    if trace.shape[-3:] != (41, 4, 4):
        raise RuntimeError(f"L0 HC replay trace geometry changed: {tuple(trace.shape)}")
    return trace


def replay_post_attn_point(branch, residual, post, comb, flat_index, torch):
    """Replay Rust/TileLang's four-lane HC post tree at one actual torch input point."""
    branch = branch.detach().cpu().contiguous()
    residual = residual.detach().cpu().contiguous()
    post = post.detach().cpu().float().contiguous()
    comb = comb.detach().cpu().float().contiguous()
    hc, hidden = residual.shape[-2:]
    rows = residual.numel() // (hc * hidden)
    if hc != 4 or branch.numel() != rows * hidden:
        raise RuntimeError("post-attention HC point replay requires four residual lanes")
    if post.numel() != rows * hc or comb.numel() != rows * hc * hc:
        raise RuntimeError("post-attention HC point replay received invalid coefficient geometry")
    row, offset = divmod(flat_index, hc * hidden)
    output_lane, column = divmod(offset, hidden)
    if row >= rows:
        raise RuntimeError("post-attention HC point replay flat index is out of bounds")

    branch_value = branch.reshape(rows, hidden)[row, column].reshape(1)
    residual_values = residual.reshape(rows, hc, hidden)[row, :, column]
    branch_coeff = post.reshape(rows, hc)[row, output_lane].reshape(1)
    residual_coeffs = comb.reshape(rows, hc, hc)[row, :, output_lane]
    branch_product = branch_coeff * branch_value.float()
    residual_products = residual_coeffs * residual_values.float()
    partial_ab = residual_products[0] + residual_products[1]
    partial_cd = residual_products[2] + residual_products[3]
    residual_sum = partial_ab + partial_cd
    total = branch_product[0] + residual_sum
    partial_sums = torch.stack((partial_ab, partial_cd, residual_sum, total))
    return {
        "coordinate": (row, output_lane, column),
        "branch_bf16": branch_value,
        "residual_bf16": residual_values,
        "branch_coeff_f32": branch_coeff,
        "residual_coeff_f32": residual_coeffs,
        "branch_product_f32": branch_product,
        "residual_products_f32": residual_products,
        "partial_sums_f32": partial_sums,
        "replay_output_bf16": total.reshape(1).to(torch.bfloat16),
    }


def replay_route_denom(selected_scores, torch):
    """Replay the pinned inline six-slot route sum from its actual gather output."""
    selected_scores = selected_scores.detach().cpu().float().contiguous()
    if selected_scores.ndim != 2 or selected_scores.shape[-1] != 6:
        raise RuntimeError("L0 route denominator replay requires six selected scores per row")
    denominator = selected_scores[:, 0:1].clone()
    for slot in range(1, 6):
        denominator = denominator + selected_scores[:, slot : slot + 1]
    return denominator


def replay_route_denom_steps(selected_scores, torch):
    """Expose serial prefix bits without claiming they are Tensor.sum internals."""
    if selected_scores.ndim != 2 or selected_scores.shape[-1] != 6:
        raise RuntimeError("route denominator trace requires six selected scores per row")
    denominator = selected_scores[:, 0:1]
    steps = [denominator]
    for slot in range(1, 6):
        denominator = denominator + selected_scores[:, slot : slot + 1]
        steps.append(denominator)
    return torch.cat(steps, dim=1)


def reference_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, torch, cutpoints=None):
    """Pinned model.py/kernel.py formula with scalar FP32 reductions."""
    q_cpu, kv_cpu = q.detach().cpu(), kv.detach().cpu()
    sink_cpu, indices_cpu = attn_sink.detach().cpu().float(), topk_idxs.detach().cpu()
    output = torch.zeros_like(q_cpu)
    scores_dump = torch.zeros((*topk_idxs.shape[:2], q.size(2), topk_idxs.size(-1)), dtype=torch.float32, device="cpu")
    probs_dump = torch.zeros_like(scores_dump)
    for batch_id in range(q.size(0)):
        for row_id in range(q.size(1)):
            keys = [
                (slot, int(key)) for slot, key in enumerate(indices_cpu[batch_id, row_id].tolist())
                if 0 <= int(key) < kv_cpu.size(1)
            ]
            for head_id in range(q.size(2)):
                scores = []
                for slot, key in keys:
                    score = torch.zeros((), dtype=torch.float32, device="cpu")
                    for column in range(q.size(3)):
                        score += q_cpu[batch_id, row_id, head_id, column].float() * kv_cpu[batch_id, key, column].float()
                    scores_dump[batch_id, row_id, head_id, slot] = score
                    scores.append(score * softmax_scale)
                if not scores:
                    continue
                score_max = torch.stack(scores).max()
                denominator = torch.zeros((), dtype=torch.float32, device="cpu")
                numerator = torch.zeros(q.size(3), dtype=torch.float32, device="cpu")
                for score, (slot, key) in zip(scores, keys):
                    probability = torch.exp(score - score_max)
                    probs_dump[batch_id, row_id, head_id, slot] = probability
                    denominator += probability
                    numerator += probability.to(torch.bfloat16).float() * kv_cpu[batch_id, key].float()
                denominator += torch.exp(sink_cpu[head_id] - score_max)
                output[batch_id, row_id, head_id] = (numerator / denominator).to(torch.bfloat16)
    if cutpoints is not None:
        cutpoints.write_tensor("attn_scores_f32", scores_dump, torch)
        cutpoints.write_tensor("attn_probs_f32", probs_dump, torch)
    return output.to(q.device)


def layer_record(block, hidden, topk, item, source_bytes, elapsed_ns, torch, compact=False):
    layer_id, ratio = item["layer"], item["compression_ratio"]
    record_tensor = tensor_digest_record if compact else tensor_record
    attention = block.attn
    empty_bf16 = torch.empty((1, 0, 512), dtype=torch.bfloat16)
    empty_f32 = torch.empty(0, dtype=torch.float32)
    main = getattr(attention, "compressor", None)
    indexer = getattr(attention, "indexer", None)
    window_columns = min(hidden.size(1), attention.window_size)
    indexer_topk = topk[..., window_columns:] if indexer is not None else torch.empty((1, 1, 0), dtype=torch.int32)
    return {
        "event": "layer", "layer": layer_id, "compression_ratio": ratio,
        "router": item["router"], "source_range_bytes": source_bytes,
        "layer_elapsed_ns": elapsed_ns, "rss_during_layer": rss(),
        "cuda_allocator": {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "hidden_bf16": record_tensor(hidden, torch),
        "route_ids": block.ffn.last_ids.reshape(-1).tolist(),
        "route_weights_f32": block.ffn.last_weights.float().reshape(-1).tolist(),
        "route_weight_bits": [value & 0xFFFFFFFF for value in block.ffn.last_weights.float().view(torch.int32).reshape(-1).tolist()],
        "attention_topk_i32": record_tensor(topk.int(), torch),
        "indexer_topk_i32": record_tensor(indexer_topk.int(), torch),
        "cache": {
            "window_active_bf16": record_tensor(attention.kv_cache[:, :1], torch),
            "main_compressed_active_bf16": record_tensor(empty_bf16, torch),
            "indexer_compressed_active_bf16": record_tensor(empty_bf16[..., :128], torch),
        },
        "main_compressor": {
            "kv_state_f32": record_tensor(main.kv_state if main is not None else empty_f32, torch),
            "score_state_f32": record_tensor(main.score_state if main is not None else empty_f32, torch),
        },
        "indexer": None if indexer is None else {
            "kv_state_f32": record_tensor(indexer.compressor.kv_state, torch),
            "score_state_f32": record_tensor(indexer.compressor.score_state, torch),
        },
    }


def attach_l0_cutpoints(block, cutpoints, audit_state, torch, hc_debug_trace=False):
    handles = []

    def output_hook(name, state_key=None):
        def capture(_module, _inputs, output):
            cutpoints.write_tensor(name, output, torch)
            if state_key is not None:
                audit_state[state_key] = output
        return capture

    handles.append(block.attn_norm.register_forward_hook(output_hook("attn_norm_bf16")))
    handles.append(block.attn.wq_a.register_forward_hook(output_hook("q_a_linear_bf16")))
    handles.append(block.attn.q_norm.register_forward_hook(output_hook("q_a_norm_bf16")))
    handles.append(block.attn.wq_b.register_forward_hook(output_hook("q_b_logical_bf16", "q_b")))
    handles.append(block.attn.wkv.register_forward_hook(output_hook("wkv_linear_bf16")))
    handles.append(block.attn.kv_norm.register_forward_hook(output_hook("kv_norm_bf16", "kv_norm")))
    handles.append(block.attn.wo_b.register_forward_hook(output_hook("wo_b_hidden_bf16")))
    if hc_debug_trace:
        handles.append(block.ffn_norm.register_forward_hook(output_hook("router_input_bf16")))

    def wo_b_input(_module, inputs):
        logical = inputs[0].unflatten(-1, (block.attn.n_groups, block.attn.o_lora_rank))
        cutpoints.write_tensor("wo_a_logical_bf16", logical, torch)

    handles.append(block.attn.wo_b.register_forward_pre_hook(wo_b_input))
    if hc_debug_trace:
        original_hc_pre = block.hc_pre
        audit_state["hc_pre_calls"] = 0

        def audited_hc_pre(x, fn, scale, base):
            output = original_hc_pre(x, fn, scale, base)
            audit_state["hc_pre_calls"] += 1
            if audit_state["hc_pre_calls"] == 2:
                collapsed, post, comb = output
                cutpoints.write_tensor("ffn_hc_collapsed_bf16", collapsed, torch)
                cutpoints.write_tensor(
                    "ffn_hc_post_f32", post.reshape(-1, post.shape[-1]), torch,
                )
                cutpoints.write_tensor(
                    "ffn_hc_comb_f32",
                    comb.reshape(-1, comb.shape[-2], comb.shape[-1]), torch,
                )
            return output

        block.hc_pre = audited_hc_pre
    original_hc_post = block.hc_post
    audit_state["hc_post_calls"] = 0

    def audited_hc_post(x, residual, post, comb):
        output = original_hc_post(x, residual, post, comb)
        audit_state["hc_post_calls"] += 1
        if audit_state["hc_post_calls"] == 1:
            cutpoints.write_tensor("post_attn_hidden_bf16", output, torch)
            if hc_debug_trace:
                point = replay_post_attn_point(
                    x, residual, post, comb, L0_HC_DEBUG_FLAT_INDEX, torch,
                )
                if point.pop("coordinate") != (0, 3, 3814):
                    raise RuntimeError("pinned post-attention flat index coordinate changed")
                names = {
                    "branch_bf16": "post_attn_flat16102_branch_bf16",
                    "residual_bf16": "post_attn_flat16102_residual_bf16",
                    "branch_coeff_f32": "post_attn_flat16102_branch_coeff_f32",
                    "residual_coeff_f32": "post_attn_flat16102_residual_coeff_f32",
                    "branch_product_f32": "post_attn_flat16102_branch_product_f32",
                    "residual_products_f32": "post_attn_flat16102_residual_products_f32",
                    "partial_sums_f32": "post_attn_flat16102_partial_sums_f32",
                    "replay_output_bf16": "post_attn_flat16102_replay_output_bf16",
                }
                for key, name in names.items():
                    cutpoints.write_tensor(name, point[key], torch)
                cutpoints.write_tensor(
                    "post_attn_flat16102_torch_output_bf16",
                    output.reshape(-1)[L0_HC_DEBUG_FLAT_INDEX].reshape(1), torch,
                )
        return output

    block.hc_post = audited_hc_post
    handles.append(attach_gate_trace(block, audit_state))
    return handles


def attach_gate_trace(block, audit_state):
    original_forward = block.ffn.gate.forward

    def traced_forward(*args, **kwargs):
        audit_state["router_gate_forward"] = True
        try:
            return original_forward(*args, **kwargs)
        finally:
            audit_state["router_gate_forward"] = False

    block.ffn.gate.forward = traced_forward

    class RestoreGateForward:
        def remove(self):
            block.ffn.gate.forward = original_forward

    return RestoreGateForward()


def attach_router_trace_cutpoint(block, cutpoints, audit_state, torch, trace_rmsnorm=False):
    """Direct router boundaries, optionally with the pinned FFN RMSNorm trace."""
    def capture(_module, _inputs, output):
        cutpoints.write_tensor("router_input_bf16", output, torch)

    handles = [block.ffn_norm.register_forward_hook(capture), attach_gate_trace(block, audit_state)]
    if not trace_rmsnorm:
        return handles

    original_hc_pre = block.hc_pre
    hc_pre_globals = original_hc_pre.__globals__
    original_hc_split_sinkhorn = hc_pre_globals["hc_split_sinkhorn"]
    hc_pre_calls = 0

    def audited_hc_split_sinkhorn(*args, **kwargs):
        output = original_hc_split_sinkhorn(*args, **kwargs)
        if hc_pre_calls == 2:
            # The TileLang return is the sole authoritative source for pre.
            # It is one four-F32 vector shared by all hidden columns; retain
            # only the coefficients consumed by the disputed hidden column.
            cutpoints.write_tensor("ffn_hc_pre_for_hidden124_f32", output[0][0, 0], torch)
        return output

    def audited_hc_pre(x, fn, scale, base):
        nonlocal hc_pre_calls
        hc_pre_calls += 1
        output = original_hc_pre(x, fn, scale, base)
        if hc_pre_calls == 2:
            cutpoints.write_tensor("ffn_hc_collapsed_bf16", output[0], torch)
        return output

    # This is the exact pinned RMSNorm.forward operation sequence, instrumented
    # only for the traced layer after the actual ffn hc_pre boundary above.
    def traced_ffn_norm(x):
        dtype = x.dtype
        x = x.float()
        squares = x.square()
        mean = squares.mean(-1, keepdim=True)
        inv = torch.rsqrt(mean + block.ffn_norm.eps)
        normalized = x * inv
        output = (block.ffn_norm.weight * normalized).to(dtype)
        cutpoints.write_tensor("router_rmsnorm_square_f32", squares, torch)
        cutpoints.write_tensor("router_rmsnorm_mean_f32", mean, torch)
        cutpoints.write_tensor("router_rmsnorm_inv_f32", inv, torch)
        return output

    hc_pre_globals["hc_split_sinkhorn"] = audited_hc_split_sinkhorn
    block.hc_pre = audited_hc_pre
    block.ffn_norm.forward = traced_ffn_norm
    class RestoreHcSplitSinkhorn:
        def remove(self):
            hc_pre_globals["hc_split_sinkhorn"] = original_hc_split_sinkhorn

    handles.append(RestoreHcSplitSinkhorn())
    return handles


def write_l2_compressor_precommit_cutpoints(block, cutpoints, torch):
    for prefix, compressor in (
        ("main", block.attn.compressor),
        ("indexer", block.attn.indexer.compressor),
    ):
        ratio = compressor.compress_ratio
        current = slice(ratio, ratio + 1)
        cutpoints.write_tensor(
            f"{prefix}_compressor_kv_precommit_f32",
            compressor.kv_state[:, current], torch,
        )
        cutpoints.write_tensor(
            f"{prefix}_compressor_score_precommit_f32",
            compressor.score_state[:, current], torch,
        )


def run_final_head(hidden, last_block, model_dir, model_args, official, torch, safe_open):
    started = time.perf_counter_ns()
    shard = model_dir / shard_name(45)
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        hc_head_fn = tensors.get_tensor("hc_head_fn").cuda()
        hc_head_scale = tensors.get_tensor("hc_head_scale").cuda()
        hc_head_base = tensors.get_tensor("hc_head_base").cuda()
    source_bytes = sum(tensor.numel() * tensor.element_size() for tensor in (
        hc_head_fn, hc_head_scale, hc_head_base,
    ))
    with torch.inference_mode():
        collapsed = last_block.hc_head(hidden, hc_head_fn, hc_head_scale, hc_head_base)
    del hc_head_fn, hc_head_scale, hc_head_base, last_block
    torch.cuda.empty_cache()

    norm = official.RMSNorm(model_args.dim, model_args.norm_eps)
    head = official.ParallelHead(model_args.vocab_size, model_args.dim, model_args.norm_eps, model_args.hc_eps)
    source_bytes += copy_parameters(norm, shard, "norm", torch, safe_open)
    source_bytes += copy_parameters(head, shard, "head", torch, safe_open)
    with torch.inference_mode():
        normalized = norm(collapsed)
        logits = head(normalized)
        argmax = int(logits.argmax(dim=-1).item())
    torch.cuda.synchronize()
    return {
        "event": "final_head", "source_range_bytes": source_bytes,
        "elapsed_ns": time.perf_counter_ns() - started, "rss_during_head": rss(),
        "hc_collapsed_bf16": tensor_digest_record(collapsed, torch),
        "norm_hidden_bf16": tensor_digest_record(normalized, torch),
        "logits_f32": tensor_digest_record(logits, torch),
        "argmax_token_id": argmax,
        "cuda_allocator": {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }


def run_head_only(args):
    model_dir, config, _ = validate_checkpoint(args.model_dir)
    import torch
    from safetensors.torch import safe_open
    validate_resources(torch)
    official = import_official(model_dir)
    official.world_size, official.rank = 1, 0
    official.default_dtype = torch.float8_e4m3fn
    official.scale_fmt, official.scale_dtype = "ue8m0", torch.float8_e8m0fnu
    model_args = official_args(official, config)
    input_manifest = json.loads(Path(args.input_manifest).read_text())
    if (input_manifest.get("schema") != "colibri-flash-0731-final-head-capture-v1"
            or input_manifest.get("model_revision") != REVISION
            or input_manifest.get("shape") != [1, model_args.hc_mult, model_args.dim]):
        raise RuntimeError("head-only input manifest mismatch")
    raw = Path(args.input_bin).read_bytes()
    hidden_meta = input_manifest.get("hidden", {})
    if (len(raw) != model_args.hc_mult * model_args.dim * 2
            or hashlib.sha256(raw).hexdigest() != hidden_meta.get("sha256")
            or fnv1a(raw) != hidden_meta.get("fnv1a64")):
        raise RuntimeError("head-only input bytes mismatch")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    hidden = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(
        1, 1, model_args.hc_mult, model_args.dim).cuda()
    shard = model_dir / shard_name(45)
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        hc_fn = tensors.get_tensor("hc_head_fn").cuda()
        hc_scale = tensors.get_tensor("hc_head_scale").cuda()
        hc_base = tensors.get_tensor("hc_head_base").cuda()
    dummy = type("HeadOnlyBlock", (), {"norm_eps": model_args.norm_eps, "hc_eps": model_args.hc_eps})()
    norm = official.RMSNorm(model_args.dim, model_args.norm_eps)
    head = official.ParallelHead(model_args.vocab_size, model_args.dim, model_args.norm_eps, model_args.hc_eps)
    copy_parameters(norm, shard, "norm", torch, safe_open)
    copy_parameters(head, shard, "head", torch, safe_open)
    with torch.inference_mode():
        collapsed = official.Block.hc_head(dummy, hidden, hc_fn, hc_scale, hc_base)
        normalized = norm(collapsed)
        logits = head(normalized)
    torch.cuda.synchronize()

    # This sidecar follows the pinned upstream expression explicitly.  The
    # gate values through `pre4_actual` are real CUDA tensor results; the four
    # collapse partials are an intentionally separate scalar replay because
    # torch.sum does not expose its internal reduction partials.
    def f32_trace(value):
        value = value.detach().float().reshape(())
        return {
            "bits": f"0x{int(value.view(torch.int32).item()) & 0xffffffff:08x}",
            "value": float(value.item()),
        }

    with torch.inference_mode():
        flat = hidden.flatten(2).float()
        square_sum_separate = flat.square().sum(-1, keepdim=True)
        mean_plus_eps = flat.square().mean(-1, keepdim=True) + dummy.norm_eps
        rsqrt = torch.rsqrt(mean_plus_eps)
        raw_dots = torch.nn.functional.linear(flat, hc_fn)
        scaled = raw_dots * rsqrt
        affine = scaled * hc_scale + hc_base
        pre = torch.sigmoid(affine) + dummy.hc_eps
        column = 3661
        inputs = hidden[0, 0, :, column].float()
        products = pre[0, 0] * inputs
        partial = torch.zeros((), device="cuda", dtype=torch.float32)
        collapse_terms = []
        for copy in range(model_args.hc_mult):
            partial = partial + products[copy]
            input_bits = int(hidden[0, 0, copy, column].view(torch.int16).item()) & 0xffff
            collapse_terms.append({
                "copy": copy,
                "input_bf16_bits": f"0x{input_bits:04x}",
                "input_f32": f32_trace(inputs[copy]),
                "pre_actual": f32_trace(pre[0, 0, copy]),
                "product_scalar_replay": f32_trace(products[copy]),
                "partial_scalar_replay": f32_trace(partial),
            })
        trace = {
            "schema": "colibri-flash-0731-final-head-official-trace-v1",
            "scope": "head_only",
            "gate": {
                "square_sum_separate_actual": f32_trace(square_sum_separate),
                "mean_plus_eps_actual": f32_trace(mean_plus_eps),
                "rsqrt_actual": f32_trace(rsqrt),
                "copies": [{
                    "copy": copy,
                    "raw_dot_actual": f32_trace(raw_dots[0, 0, copy]),
                    "scaled_actual": f32_trace(scaled[0, 0, copy]),
                    "affine_actual": f32_trace(affine[0, 0, copy]),
                    "pre_actual": f32_trace(pre[0, 0, copy]),
                } for copy in range(model_args.hc_mult)],
            },
            "collapse_index_3661": {
                "terms": collapse_terms,
                "collapsed_bf16_actual": (
                    f"0x{int(collapsed[0, 0, column].view(torch.int16).item()) & 0xffff:04x}"
                ),
                "partials": "scalar_replay; upstream torch.sum does not expose internal reduction partials",
            },
        }
    output = Path(args.out_dir)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite head-only output: {output}")
    pending = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    pending.mkdir(parents=True, exist_ok=False)
    try:
        records = {}
        for name, tensor in (("collapsed_bf16", collapsed), ("normalized_bf16", normalized), ("logits_f32", logits)):
            data = raw_bytes(tensor, torch)
            filename = f"{name}.bin"
            (pending / filename).write_bytes(data)
            records[name] = {"file": filename, **tensor_digest_record(tensor, torch)}
        (pending / "head-trace.json").write_text(json.dumps(trace, sort_keys=True, indent=2) + "\n")
        (pending / "manifest.json").write_text(json.dumps({
            "schema": "colibri-flash-0731-final-head-official-v1",
            "model_revision": REVISION,
            "input_hidden_fnv1a64": fnv1a(raw), "tensors": records,
            "argmax": int(logits.argmax(dim=-1).item()), "trace": "head-trace.json",
        }, sort_keys=True, indent=2) + "\n")
        os.replace(pending, output)
    except Exception:
        for path in pending.glob("*"):
            path.unlink()
        pending.rmdir()
        raise
    print(output)


def run_oracle(args):
    model_dir, config, checkpoint = validate_checkpoint(args.model_dir)
    import torch
    from safetensors.torch import safe_open

    resources = validate_resources(torch)
    official = import_official(model_dir)
    official.world_size, official.rank = 1, 0
    official.default_dtype = torch.float8_e4m3fn
    official.scale_fmt, official.scale_dtype = "ue8m0", torch.float8_e8m0fnu
    model_args = official_args(official, config)
    end_layer = getattr(args, "end_layer", 3)
    final_head = getattr(args, "final_head", False)
    manifest = run_manifest(end_layer, final_head)
    plan = execution_plan(end_layer)
    compact = len(plan) > len(PREFIX)
    hc_debug_trace = getattr(args, "l0_hc_debug_trace", False)
    router_trace_layer = getattr(args, "router_trace_layer", None)
    router_trace_rmsnorm = getattr(args, "router_trace_rmsnorm", False)
    cutpoint_layer = getattr(args, "cutpoint_layer", None)
    if router_trace_layer is not None and cutpoint_layer is not None:
        raise RuntimeError("router trace and full cutpoint trace are mutually exclusive")
    router_trace = router_trace_layer is not None
    trace_layer = router_trace_layer if router_trace else (
        cutpoint_layer if cutpoint_layer is not None else 0
    )
    cutpoints = None
    audit_state = {
        "active": False,
        "hc_debug_trace": hc_debug_trace,
        "router_trace": router_trace,
        "router_trace_rmsnorm": router_trace_rmsnorm,
        "trace_layer": trace_layer,
    }
    if getattr(args, "cutpoint_dir", None) is not None:
        cutpoints = CutpointWriter(
            args.cutpoint_dir, args.token_id, hc_debug_trace,
            router_trace_layer=trace_layer,
            router_trace_rmsnorm=router_trace_rmsnorm,
        )
    official.MoE = make_streaming_moe(official, torch, safe_open, cutpoints, audit_state)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    attention_topk = []

    original_linear = official.linear
    original_rotary = official.apply_rotary_emb
    original_hc_split_sinkhorn = official.hc_split_sinkhorn
    original_act_quant = official.act_quant
    original_functional_linear = official.F.linear
    original_torch_rsqrt = torch.rsqrt
    original_tensor_sqrt = torch.Tensor.sqrt
    original_tensor_gather = torch.Tensor.gather
    original_tensor_sum = torch.Tensor.sum
    original_functional_softplus = official.F.softplus

    def audited_functional_linear(x, weight, bias=None):
        output = original_functional_linear(x, weight, bias)
        if (
            cutpoints is not None and audit_state["active"] and hc_debug_trace
            and not audit_state.get("attn_hc_raw_dot_captured", False)
            and x.dtype == torch.float32 and weight.dtype == torch.float32
            and tuple(weight.shape) == (
                (2 + model_args.hc_mult) * model_args.hc_mult,
                model_args.hc_mult * model_args.dim,
            )
            and bias is None
        ):
            cutpoints.write_tensor("attn_hc_raw_dot_f32", output, torch)
            audit_state["attn_hc_raw_dot_captured"] = True
        if (
            cutpoints is not None and audit_state["active"]
            and (audit_state["router_trace"] or hc_debug_trace)
            and audit_state.get("router_gate_forward", False)
            and not audit_state.get("router_gate_logits_captured", False)
            and x.dtype == torch.float32 and weight.dtype == torch.float32
            and tuple(weight.shape) == (model_args.n_routed_experts, model_args.dim)
            and bias is None
        ):
            cutpoints.write_tensor("router_gate_logits_f32", output, torch)
            audit_state["router_gate_logits_captured"] = True
        return output

    def audited_torch_rsqrt(value, *positional, **keywords):
        output = original_torch_rsqrt(value, *positional, **keywords)
        if (
            cutpoints is not None and audit_state["active"] and hc_debug_trace
            and not audit_state.get("attn_hc_rsqrt_captured", False)
            and value.dtype == torch.float32 and tuple(value.shape) == (1, 1, 1)
        ):
            cutpoints.write_tensor("attn_hc_rsqrt_input_f32", value, torch)
            cutpoints.write_tensor("attn_hc_rsqrt_f32", output, torch)
            audit_state["attn_hc_rsqrt_captured"] = True
        return output

    def audited_tensor_sqrt(value, *positional, **keywords):
        output = original_tensor_sqrt(value, *positional, **keywords)
        if (
            cutpoints is not None and audit_state["active"]
            and (audit_state["router_trace"] or hc_debug_trace)
            and not audit_state.get("router_post_score_captured", False)
            and value.dtype == torch.float32
            and tuple(value.shape) == (1, model_args.n_routed_experts)
        ):
            audit_state["router_post_actual"] = output.detach().clone()
            audit_state["router_post_score_captured"] = True
        return output

    def audited_tensor_gather(value, dim, index, *positional, **keywords):
        output = original_tensor_gather(value, dim, index, *positional, **keywords)
        if (
            cutpoints is not None and audit_state["active"]
            and (audit_state["router_trace"] or hc_debug_trace)
            and not audit_state.get("selected_scores_captured", False)
            and value.dtype == torch.float32 and dim == 1
            and tuple(value.shape) == (1, model_args.n_routed_experts)
            and tuple(index.shape) == (1, model_args.n_activated_experts)
        ):
            audit_state["selected_scores_actual"] = output.detach().clone()
            audit_state["selected_score_indices"] = index.detach().clone()
            audit_state["selected_scores_captured"] = True
        return output

    def audited_functional_softplus(value, *positional, **keywords):
        output = original_functional_softplus(value, *positional, **keywords)
        if (
            cutpoints is not None and audit_state["active"] and audit_state["router_trace"]
            and not audit_state.get("router_softplus_captured", False)
            and value.dtype == torch.float32
            and tuple(value.shape) == (1, model_args.n_routed_experts)
        ):
            audit_state["router_softplus_actual"] = output.detach().clone()
            audit_state["router_softplus_captured"] = True
        return output

    def audited_tensor_sum(value, *positional, **keywords):
        output = original_tensor_sum(value, *positional, **keywords)
        dim = keywords.get("dim", positional[0] if positional else None)
        keepdim = keywords.get("keepdim", positional[1] if len(positional) > 1 else False)
        if (
            cutpoints is not None and audit_state["active"] and audit_state["router_trace"]
            and not audit_state.get("route_denom_captured", False)
            and value.dtype == torch.float32
            and tuple(value.shape) == (1, model_args.n_activated_experts)
            and dim in (-1, 1) and keepdim
        ):
            audit_state["route_denom_actual"] = output.detach().clone()
            audit_state["route_denom_captured"] = True
        return output

    def audited_act_quant(value, *positional, **keywords):
        output = original_act_quant(value, *positional, **keywords)
        record = audit_state.get("active_expert_record")
        stage = audit_state.get("active_expert_linear")
        if (
            record is not None and stage in ("w1", "w3", "w2")
            and cutpoints is not None and audit_state["active"] and hc_debug_trace
        ):
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("routed FP4 act_quant did not return physical value/scale tensors")
            quantized, scale = output
            if quantized.dtype != torch.float8_e4m3fn:
                raise RuntimeError("routed FP4 act_quant value dtype changed")
            if scale.dtype != torch.float8_e8m0fnu:
                raise RuntimeError("routed FP4 act_quant scale dtype changed")
            if tuple(quantized.shape) != tuple(value.shape):
                raise RuntimeError("routed FP4 act_quant value geometry changed")
            expected_scale_shape = tuple(value.shape[:-1]) + (value.shape[-1] // 128,)
            if tuple(scale.shape) != expected_scale_shape:
                raise RuntimeError("routed FP4 act_quant scale geometry changed")
            value_key, scale_key = f"{stage}_quant_value", f"{stage}_quant_scale"
            if value_key in record or scale_key in record:
                raise RuntimeError(f"routed FP4 {stage} act_quant hook fired twice")
            record[value_key] = quantized.detach().clone()
            record[scale_key] = scale.detach().clone()
        return output

    def audited_linear(x, weight, bias=None):
        output = original_linear(x, weight, bias)
        if (
            cutpoints is not None and audit_state["active"] and not audit_state["router_trace"]
            and audit_state.get("router_gate_forward", False)
            and x.dtype == torch.float32 and weight.dtype == torch.float32
            and tuple(weight.shape) == (model_args.n_routed_experts, model_args.dim)
        ):
            audit_state["router_pre"] = output
            cutpoints.write_tensor("router_pre_score_f32", output, torch)
        return output

    def audited_rotary(tensor, freqs_cis, inverse=False):
        output = original_rotary(tensor, freqs_cis, inverse)
        if cutpoints is None or not audit_state["active"] or audit_state["router_trace"]:
            return output
        if inverse:
            cutpoints.write_tensor("attn_context_bf16", audit_state.pop("sparse_output"), torch)
        elif "q_b" in audit_state:
            q = audit_state.pop("q_b").unflatten(-1, (model_args.n_heads, model_args.head_dim))
            cutpoints.write_tensor("q_rope_bf16", q, torch)
        elif "kv_norm" in audit_state:
            cutpoints.write_tensor("k_rope_bf16", audit_state.pop("kv_norm"), torch)
        return output

    def audited_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        attention_topk.append(topk_idxs.detach().cpu().contiguous())
        if cutpoints is not None and audit_state["active"] and not audit_state["router_trace"]:
            cutpoints.write_tensor("cache_kv_row_bf16", kv, torch)
        output = reference_sparse_attn(
            q, kv, attn_sink, topk_idxs, softmax_scale, torch,
            cutpoints if audit_state["active"] and not audit_state["router_trace"] else None,
        )
        if cutpoints is not None and audit_state["active"] and not audit_state["router_trace"]:
            audit_state["sparse_output"] = output
        return output

    def audited_hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6,
    ):
        output = original_hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps,
        )
        if (
            cutpoints is not None and audit_state["active"] and hc_debug_trace
            and not audit_state.get("attn_hc_sinkhorn_captured", False)
        ):
            if not all(audit_state.get(key, False) for key in (
                "attn_hc_raw_dot_captured", "attn_hc_rsqrt_captured",
            )):
                raise RuntimeError("actual L0 HC linear/rsqrt hooks were not captured")
            pre, post, comb = output
            cutpoints.write_tensor("attn_hc_scaled_mixes_f32", mixes, torch)
            cutpoints.write_tensor("attn_hc_mixes_f32", mixes, torch)
            cutpoints.write_tensor("attn_hc_scale_f32", hc_scale, torch)
            cutpoints.write_tensor("attn_hc_base_f32", hc_base, torch)
            cutpoints.write_tensor("attn_hc_pre_tilelang_f32", pre, torch)
            cutpoints.write_tensor("attn_hc_post_tilelang_f32", post, torch)
            cutpoints.write_tensor("attn_hc_comb_tilelang_f32", comb, torch)
            cutpoints.write_tensor(
                "attn_hc_sinkhorn_replay_trace_f32",
                replay_hc_sinkhorn_trace(
                    mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, torch,
                ),
                torch,
            )
            audit_state["attn_hc_sinkhorn_captured"] = True
        return output

    official.linear = audited_linear
    official.apply_rotary_emb = audited_rotary
    official.sparse_attn = audited_sparse_attn
    if hc_debug_trace or router_trace:
        official.F.linear = audited_functional_linear
        torch.Tensor.sqrt = audited_tensor_sqrt
        torch.Tensor.gather = audited_tensor_gather
        official.F.softplus = audited_functional_softplus
        torch.Tensor.sum = audited_tensor_sum
    if hc_debug_trace:
        official.act_quant = audited_act_quant
        torch.rsqrt = audited_torch_rsqrt
        official.hc_split_sinkhorn = audited_hc_split_sinkhorn
    with safe_open(model_dir / shard_name(1), framework="pt", device="cpu") as tensors:
        embedding = tensors.get_slice("embed.weight")[args.token_id : args.token_id + 1].cuda()
    hidden = embedding[:, None, None, :].repeat(1, 1, 4, 1)
    input_ids = torch.tensor([[args.token_id]], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    final_argmax = None
    with args.out.open("x") as trace:
        trace.write(json.dumps({
            "event": "run", **manifest, "implementation": "official-python-streaming",
            "token_id": args.token_id, "start_pos": 0, "executed_layers": len(plan),
            "model_dir": str(model_dir), "checkpoint": checkpoint, "resources": resources,
            "initial_embedding_bf16": tensor_record(embedding, torch), "rss_before_layers": rss(),
            "attention": "scalar FP32 reduction of pinned official sparse_attn formula",
            "ownership": "one official Block plus one selected routed expert at a time",
        }, sort_keys=True) + "\n")
        trace.flush()
        for item in plan:
            layer_id = item["layer"]
            attention_topk.clear()
            started = time.perf_counter_ns()
            shard = model_dir / shard_name(item["shard"])
            block = official.Block(layer_id, model_args)
            block.ffn.shard = shard
            source_bytes = copy_parameters(block, shard, f"layers.{layer_id}", torch, safe_open)
            handles = []
            if cutpoints is not None and layer_id == trace_layer:
                if layer_id == 2:
                    # The L0 boundary hooks are intentionally not attached to
                    # CSA: its indexer has additional RoPE calls.  This trace
                    # needs only the four compressor vectors below.
                    audit_state["active"] = False
                else:
                    audit_state["active"] = True
                    if router_trace:
                        handles = attach_router_trace_cutpoint(
                            block, cutpoints, audit_state, torch, router_trace_rmsnorm,
                        )
                    else:
                        cutpoints.write_tensor("input_hidden_bf16", hidden, torch)
                        handles = attach_l0_cutpoints(
                            block, cutpoints, audit_state, torch, hc_debug_trace,
                        )
            with torch.inference_mode():
                hidden = block(hidden, 0, input_ids)
            for handle in handles:
                handle.remove()
            if cutpoints is not None and layer_id == trace_layer:
                if router_trace:
                    required = (
                        ROUTER_TRACE_RMSNORM_CUTPOINTS
                        if router_trace_rmsnorm else ROUTER_TRACE_CUTPOINTS
                    )
                elif layer_id == 2:
                    write_l2_compressor_precommit_cutpoints(block, cutpoints, torch)
                    required = L2_COMPRESSOR_PRECOMMIT_CUTPOINTS
                else:
                    cutpoints.write_tensor("block_output_hidden_bf16", hidden, torch)
                    required = (
                        L0_CUTPOINTS
                        + (L0_HC_DEBUG_CUTPOINTS if hc_debug_trace else ())
                    )
                cutpoints.finish(required)
                audit_state["active"] = False
            torch.cuda.synchronize()
            if len(attention_topk) != 1:
                raise RuntimeError(f"official layer {layer_id} emitted {len(attention_topk)} top-k tensors")
            source_bytes += block.ffn.source_range_bytes
            record = layer_record(
                block, hidden, attention_topk[0], item, source_bytes,
                time.perf_counter_ns() - started, torch, compact,
            )
            trace.write(json.dumps(record, sort_keys=True) + "\n")
            trace.flush()
            if final_head and layer_id == end_layer:
                head_record = run_final_head(
                    hidden, block, model_dir, model_args, official, torch, safe_open,
                )
                final_argmax = head_record["argmax_token_id"]
                trace.write(json.dumps(head_record, sort_keys=True) + "\n")
                trace.flush()
                del head_record
            del block, record
            torch.cuda.empty_cache()
            trace.write(json.dumps({
                "event": "release", "layer": layer_id, "rss_after_release": rss(),
                "cuda_allocator_after_release": {
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            }, sort_keys=True) + "\n")
            trace.flush()
        final_record = {
            "event": "final", "executed_layers": len(plan),
            "full43_oracle": end_layer == 42,
            "final_hidden_bf16": (tensor_digest_record if compact else tensor_record)(hidden, torch),
            "rss_final": rss(),
            "cuda_allocator_peak_bytes": torch.cuda.max_memory_allocated(),
            "cuda_allocator_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "status": "PASS",
        }
        if final_head:
            final_record.update({"final_head": True, "final_head_argmax_token_id": final_argmax})
        trace.write(json.dumps(final_record, sort_keys=True) + "\n")
    print(args.out)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("manifest")
    header = commands.add_parser("header-only")
    header.add_argument("--model-dir", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--token-id", type=int, required=True, choices=range(129280))
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--cutpoint-dir", type=Path)
    run.add_argument("--cutpoint-layer", type=int, choices=range(43))
    run.add_argument("--l0-hc-debug-trace", action="store_true")
    run.add_argument("--router-trace-layer", type=int, choices=range(43))
    run.add_argument("--router-trace-rmsnorm", action="store_true")
    run.add_argument("--end-layer", type=int, default=3, choices=range(43))
    run.add_argument("--final-head", action="store_true")
    head_only = commands.add_parser("head-only")
    head_only.add_argument("--model-dir", type=Path, required=True)
    head_only.add_argument("--input-manifest", type=Path, required=True)
    head_only.add_argument("--input-bin", type=Path, required=True)
    head_only.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        print(json.dumps(static_manifest(), sort_keys=True))
    elif args.command == "header-only":
        model_dir, _, checkpoint = validate_checkpoint(args.model_dir)
        print(json.dumps({
            "status": "PASS", **static_manifest(), "model_dir": str(model_dir),
            "checkpoint": checkpoint, "rss": rss(),
        }, sort_keys=True))
    elif args.command == "head-only":
        run_head_only(args)
    else:
        if args.final_head and args.end_layer != 42:
            parser.error("--final-head requires --end-layer 42")
        if args.l0_hc_debug_trace and args.cutpoint_dir is None:
            parser.error("--l0-hc-debug-trace requires --cutpoint-dir")
        if args.router_trace_layer is not None and args.cutpoint_dir is None:
            parser.error("--router-trace-layer requires --cutpoint-dir")
        if args.router_trace_rmsnorm and args.router_trace_layer is None:
            parser.error("--router-trace-rmsnorm requires --router-trace-layer")
        if args.cutpoint_layer is not None and args.cutpoint_dir is None:
            parser.error("--cutpoint-layer requires --cutpoint-dir")
        if args.cutpoint_layer is not None and args.router_trace_layer is not None:
            parser.error("--cutpoint-layer and --router-trace-layer are mutually exclusive")
        if args.router_trace_layer is not None and args.end_layer < args.router_trace_layer:
            parser.error("--end-layer must include --router-trace-layer")
        if args.cutpoint_layer is not None and args.end_layer < args.cutpoint_layer:
            parser.error("--end-layer must include --cutpoint-layer")
        run_oracle(args)


if __name__ == "__main__":
    main()
