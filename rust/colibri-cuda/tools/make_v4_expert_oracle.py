#!/usr/bin/env python3
"""Generate the pinned V4-Pro layer-0 expert-0 PyTorch formula oracle.

Example:
  COLIBRI_DSV4_EXPERT_SHARD=/path/model-00002-of-00064.safetensors \
    "$COLIBRI_ORACLE_PYTHON" \
    tools/make_v4_expert_oracle.py
"""

import argparse
import hashlib
import json
import os
import struct
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import safetensors
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open


REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
HIDDEN, INTERMEDIATE, ACT_GROUP, WEIGHT_GROUP = 7168, 3072, 128, 32
INPUT_SEED, ROUTE_WEIGHT, SWIGLU_LIMIT = 97, 0.625, 10.0
MAGIC = b"CV4E0P1\n"
DEFAULT_JSON = Path(__file__).resolve().parent.parent / "fixtures/v4_expert0_pytorch_oracle.json"
DEFAULT_BIN = DEFAULT_JSON.with_suffix(".bin")
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def tensor_sha256(tensor):
    digest = hashlib.sha256()
    raw = tensor.contiguous().view(torch.uint8).reshape(-1)
    for chunk in raw.split(1 << 20):
        digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def stage(tensor):
    values = tensor.float()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "sha256": tensor_sha256(tensor),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def deterministic_input():
    index = torch.arange(HIDDEN, dtype=torch.int64)
    return (((index * 17 + INPUT_SEED) % 113).float() - 56.0).div(64.0).to(torch.bfloat16)[None]


def quantize_activation(x_bf16):
    groups = x_bf16.float().reshape(-1, ACT_GROUP)
    amax = groups.abs().amax(-1).clamp_min(1e-4)
    scale_f32 = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    scale_e8m0 = scale_f32.to(torch.float8_e8m0fnu)
    quantized = (groups / scale_e8m0.float()[:, None]).clamp(-448.0, 448.0)
    quantized = quantized.to(torch.float8_e4m3fn).reshape_as(x_bf16)
    return quantized, scale_e8m0.reshape(x_bf16.shape[0], -1)


def unpack_fp4(packed_i8):
    packed = packed_i8.view(torch.uint8)
    low = FP4_VALUES[(packed & 0x0F).long()]
    high = FP4_VALUES[(packed >> 4).long()]
    return torch.stack((low, high), dim=-1).flatten(-2)


def fp4_linear(x_bf16, packed, weight_scale):
    """PyTorch expression of official kernel.py's 32-K scale accumulation."""
    activation, activation_scale = quantize_activation(x_bf16)
    weight = unpack_fp4(packed)
    weight_scale_f32 = weight_scale.float()
    if weight.shape[1] != x_bf16.shape[1] or weight_scale.shape != (weight.shape[0], weight.shape[1] // WEIGHT_GROUP):
        raise ValueError("official FP4 value/scale layout mismatch")
    output = torch.zeros((x_bf16.shape[0], weight.shape[0]), dtype=torch.float32)
    for block in range(weight.shape[1] // WEIGHT_GROUP):
        start = block * WEIGHT_GROUP
        partial = F.linear(
            activation[:, start:start + WEIGHT_GROUP].float(),
            weight[:, start:start + WEIGHT_GROUP],
        )
        output.add_(partial * activation_scale[:, block // (ACT_GROUP // WEIGHT_GROUP), None].float()
                    * weight_scale_f32[:, block])
    return output.to(torch.bfloat16), activation, activation_scale


def load_weight(tensors, stem, source):
    weight_name = f"layers.0.ffn.experts.0.{stem}.weight"
    scale_name = f"layers.0.ffn.experts.0.{stem}.scale"
    packed = tensors.get_tensor(weight_name)
    scale = tensors.get_tensor(scale_name)
    source[stem] = {
        "weight": weight_name,
        "weight_storage_dtype": str(packed.dtype).removeprefix("torch."),
        "weight_storage_shape": list(packed.shape),
        "weight_sha256": tensor_sha256(packed),
        "scale": scale_name,
        "scale_dtype": str(scale.dtype).removeprefix("torch."),
        "scale_shape": list(scale.shape),
        "scale_sha256": tensor_sha256(scale),
    }
    return packed, scale


def build(shard):
    # ponytail: keep this a CPU formula oracle; run the pinned TileLang kernel
    # only when an existing CUDA PyTorch+TileLang environment is available.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x_pre = deterministic_input()
    stages = {"input_bf16": stage(x_pre)}
    source = {}
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        w1, s1 = load_weight(tensors, "w1", source)
        gate, input_fp8, input_scale = fp4_linear(x_pre, w1, s1)
        stages.update({"input_fp8": stage(input_fp8), "input_scale_e8m0": stage(input_scale),
                       "w1_bf16": stage(gate)})
        del w1, s1

        w3, s3 = load_weight(tensors, "w3", source)
        up, input_fp8_again, input_scale_again = fp4_linear(x_pre, w3, s3)
        if tensor_sha256(input_fp8_again) != stages["input_fp8"]["sha256"] or tensor_sha256(input_scale_again) != stages["input_scale_e8m0"]["sha256"]:
            raise RuntimeError("repeated input activation quantization changed")
        stages["w3_bf16"] = stage(up)
        del w3, s3, input_fp8_again, input_scale_again

        gate_f32 = gate.float().clamp(max=SWIGLU_LIMIT)
        up_f32 = up.float().clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        weighted_f32 = F.silu(gate_f32) * up_f32 * ROUTE_WEIGHT
        routed_bf16 = weighted_f32.to(torch.bfloat16)
        stages.update({"weighted_swiglu_f32": stage(weighted_f32), "routed_bf16": stage(routed_bf16)})

        w2, s2 = load_weight(tensors, "w2", source)
        output_bf16, routed_fp8, routed_scale = fp4_linear(routed_bf16, w2, s2)
        output_f32 = output_bf16.float()
        stages.update({"routed_fp8": stage(routed_fp8), "routed_scale_e8m0": stage(routed_scale),
                       "output_bf16": stage(output_bf16), "output_f32": stage(output_f32)})

    input_bytes = struct.pack(f"<{HIDDEN}H", *x_pre.view(torch.uint16).reshape(-1).tolist())
    output_bytes = struct.pack(f"<{HIDDEN}f", *output_f32.reshape(-1).tolist())
    binary = MAGIC + input_bytes + output_bytes
    model_source = shard.parent / "inference/model.py"
    kernel_source = shard.parent / "inference/kernel.py"
    if not model_source.is_file() or not kernel_source.is_file():
        raise FileNotFoundError("pinned inference/model.py and inference/kernel.py are required beside the shard")
    fixture = {
        "schema": "colibri-v4-expert0-pytorch-oracle-v1",
        "model_revision": REVISION,
        "shard": shard.name,
        "case": {"layer": 0, "expert": 0, "rows": 1, "hidden": HIDDEN,
                 "intermediate": INTERMEDIATE, "input_seed": INPUT_SEED,
                 "route_weight": ROUTE_WEIGHT, "swiglu_limit": SWIGLU_LIMIT},
        "semantics": {
            "input": "BF16 RNE of (((flat_index * 17 + 97) % 113) - 56) / 64",
            "activation": "BF16 groups of 128 -> E8M0 ceil-power-of-two scale -> clamped E4M3",
            "weight": "low nibble then high nibble FP4 E2M1, E8M0 scale per 32 K values",
            "gemm": "FP32 32-K partial dot, then activation_scale * weight_scale, accumulated in K order, BF16 output",
            "expert": "w1/w3 BF16 -> FP32 clamp(up,-10,10), clamp(gate,max=10) -> silu(gate)*up*0.625 -> BF16 -> w2 -> BF16",
        },
        "official_source": {
            "model.py": {"sha256": file_sha256(model_source), "lines": "108-120,587-606"},
            "kernel.py": {"sha256": file_sha256(kernel_source), "lines": "40-125,442-535"},
            "official_cuda_kernel_executed": False,
            "reason": "the isolated oracle has CPU-only PyTorch and no TileLang; this is an independent PyTorch tensor-formula reference",
        },
        "environment": {
            "device": "cpu", "torch": torch.__version__, "safetensors": safetensors.__version__,
            "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "source_tensors": source,
        "stages": stages,
        "binary": {
            "magic_ascii": MAGIC.decode("ascii").rstrip(), "sha256": hashlib.sha256(binary).hexdigest(),
            "input": {"offset": len(MAGIC), "bytes": len(input_bytes), "dtype": "bfloat16-le", "shape": [1, HIDDEN]},
            "output": {"offset": len(MAGIC) + len(input_bytes), "bytes": len(output_bytes), "dtype": "float32-le", "shape": [1, HIDDEN]},
        },
    }
    return json.dumps(fixture, indent=2, sort_keys=True) + "\n", binary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, default=os.environ.get("COLIBRI_DSV4_EXPERT_SHARD"))
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.shard is None or not args.shard.is_file():
        parser.error("set --shard or COLIBRI_DSV4_EXPERT_SHARD to the pinned model-00002 shard")
    rendered, binary = build(args.shard)
    if args.check:
        if not args.json.is_file() or args.json.read_text() != rendered or not args.bin.is_file() or args.bin.read_bytes() != binary:
            raise SystemExit("fixture differs; regenerate without --check")
        print(f"PASS: {args.json.name} and {args.bin.name} reproduce exactly")
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.bin.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered)
        args.bin.write_bytes(binary)
        print(f"wrote {args.json} and {args.bin}")


if __name__ == "__main__":
    main()
