#!/usr/bin/env python3
"""Generate the pinned L0 wq_a/wkv FP8 projection PyTorch CPU oracle."""

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import safetensors
import torch
from safetensors.torch import safe_open

REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
IDS = [0, 1, 2, 129279]
ROWS, HIDDEN, BLOCK = 4, 7168, 128
MAGIC = b"CV4P8001"
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
STATE_JSON = FIXTURES / "v4_state_pytorch_oracle.json"
STATE_BIN = FIXTURES / "v4_state_pytorch_oracle.bin"
OUT = FIXTURES / "v4_fp8_projection_pytorch_oracle.json"


def raw(tensor):
    return bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())


def sha(tensor):
    return hashlib.sha256(raw(tensor)).hexdigest()


def load_prefix():
    meta = json.loads(STATE_JSON.read_text())
    assert meta["model_revision"] == REVISION and meta["ids"] == IDS
    part = meta["binary"]["norm"]
    data = STATE_BIN.read_bytes()[part["offset"] : part["offset"] + part["bytes"]]
    tensor = torch.tensor(list(data), dtype=torch.uint8).view(torch.bfloat16)
    return tensor.reshape(ROWS, HIDDEN), meta["hashes"]["norm_bf16"]


def quantize_activation(input_bf16):
    rows, width = input_bf16.shape
    groups = input_bf16.float().reshape(rows, width // BLOCK, BLOCK)
    amax = groups.abs().amax(-1).clamp_min(1e-4)
    scale_f32 = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    scale = scale_f32.to(torch.float8_e8m0fnu)
    quantized = (
        (groups / scale.float().unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    return quantized, scale


def project(activation, activation_scale, weight, weight_scale):
    rows, blocks, block = activation.shape
    out_features = weight.shape[0]
    output = torch.zeros((rows, out_features), dtype=torch.float32)
    for index in range(blocks):
        partial = (
            activation[:, index].float()
            @ weight[:, index * block : (index + 1) * block].float().T
        )
        scale_b = weight_scale[:, index].float().repeat_interleave(block)[:out_features]
        output += partial * (
            activation_scale[:, index].float().unsqueeze(1) * scale_b
        )
    return output, output.to(torch.bfloat16)


def rmsnorm(input_bf16, weight_bf16):
    x = input_bf16.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    return (weight_bf16.float() * x).to(torch.bfloat16)


def query_finalize(wq_b_bf16):
    q = wq_b_bf16.unflatten(-1, (128, 512)).unsqueeze(0).clone()
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)

    dim, base, original_seq_len, factor = 64, 160000.0, 65536, 16.0

    def correction(num_rotations):
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction(32)), 0)
    high = min(math.ceil(correction(1)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(
        0, 1
    )
    smooth = 1 - ramp
    freqs = base ** -(torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(ROWS), freqs)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    rope = q.clone()
    tail = rope[..., -dim:]
    complex_tail = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)))
    rotated = torch.view_as_real(
        complex_tail * freqs_cis.view(1, ROWS, 1, dim // 2)
    ).flatten(-2)
    tail.copy_(rotated)
    return q, freqs_cis, rope


def kv_finalize(kv_norm, freqs_cis):
    rope = kv_norm.unsqueeze(0).clone()
    tail = rope[..., -64:]
    complex_tail = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)))
    tail.copy_(
        torch.view_as_real(complex_tail * freqs_cis.view(1, ROWS, 32)).flatten(-2)
    )

    groups = rope[..., :448].float().reshape(1, ROWS, 7, 64)
    amax = groups.abs().amax(-1).clamp_min(1e-4)
    scale_f32 = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    scale = scale_f32.to(torch.float8_e8m0fnu)
    quantized = (
        (groups / scale.float().unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    ready = rope.clone()
    ready[..., :448] = (
        quantized.float()
        * scale.float().unsqueeze(-1)
    ).to(torch.bfloat16).reshape(1, ROWS, 448)
    topk = torch.tensor(
        [[[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]]],
        dtype=torch.int32,
    )
    cache_initial = torch.zeros((1, 160, 512), dtype=torch.bfloat16)
    cache_after = cache_initial.clone()
    cache_after[:, :ROWS] = ready
    return rope, scale, ready, topk, cache_initial, cache_after


def compressor_start0(input_bf16, wkv, wgate, ape):
    x = input_bf16.unsqueeze(0).float()
    kv = torch.nn.functional.linear(x, wkv.float())
    score = torch.nn.functional.linear(x, wgate.float())
    score_plus_ape = score + ape[:ROWS].unsqueeze(0)
    kv_state_initial = torch.zeros((1, 128, 512), dtype=torch.float32)
    score_state_initial = torch.full(
        (1, 128, 512), float("-inf"), dtype=torch.float32
    )
    kv_state_after = kv_state_initial.clone()
    score_state_after = score_state_initial.clone()
    kv_state_after[:, :ROWS] = kv
    score_state_after[:, :ROWS] = score_plus_ape
    return (
        kv,
        score,
        score_plus_ape,
        kv_state_initial,
        score_state_initial,
        kv_state_after,
        score_state_after,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=OUT)
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.bin = args.bin or args.json.with_suffix(".bin")
    if args.model_dir.name != f"deepseek-v4-pro-{REVISION}":
        parser.error("model directory is not the pinned revision")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    input_bf16, state_norm_hash = load_prefix()
    activation, activation_scale = quantize_activation(input_bf16)
    shard = args.model_dir / "model-00002-of-00064.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        wq = tensors.get_tensor("layers.0.attn.wq_a.weight")
        wq_scale = tensors.get_tensor("layers.0.attn.wq_a.scale")
        wq_b = tensors.get_tensor("layers.0.attn.wq_b.weight")
        wq_b_scale = tensors.get_tensor("layers.0.attn.wq_b.scale")
        q_norm_weight = tensors.get_tensor("layers.0.attn.q_norm.weight")
        wkv = tensors.get_tensor("layers.0.attn.wkv.weight")
        wkv_scale = tensors.get_tensor("layers.0.attn.wkv.scale")
        kv_norm_weight = tensors.get_tensor("layers.0.attn.kv_norm.weight")
        compressor_wkv = tensors.get_tensor("layers.0.attn.compressor.wkv.weight")
        compressor_wgate = tensors.get_tensor(
            "layers.0.attn.compressor.wgate.weight"
        )
        compressor_ape = tensors.get_tensor("layers.0.attn.compressor.ape")
    assert (
        torch.isfinite(wq.float()).all()
        and torch.isfinite(wq_b.float()).all()
        and torch.isfinite(wkv.float()).all()
        and torch.isfinite(compressor_wkv.float()).all()
        and torch.isfinite(compressor_wgate.float()).all()
        and torch.isfinite(compressor_ape).all()
    )
    assert (
        torch.isfinite(wq_scale.float()).all()
        and torch.isfinite(wq_b_scale.float()).all()
        and torch.isfinite(wkv_scale.float()).all()
    )

    wq_f32, wq_bf16 = project(activation, activation_scale, wq, wq_scale)
    q_norm = rmsnorm(wq_bf16, q_norm_weight)
    wq_b_activation, wq_b_activation_scale = quantize_activation(q_norm)
    wq_b_f32, wq_b_bf16 = project(
        wq_b_activation, wq_b_activation_scale, wq_b, wq_b_scale
    )
    q_head_rms, freqs_cis, q_rope = query_finalize(wq_b_bf16)
    wkv_f32, wkv_bf16 = project(activation, activation_scale, wkv, wkv_scale)
    kv_norm = rmsnorm(wkv_bf16, kv_norm_weight)
    kv_rope, kv_scale, kv_ready, causal_topk, cache_initial, cache_after = (
        kv_finalize(kv_norm, freqs_cis)
    )
    (
        compressor_kv,
        compressor_score,
        compressor_score_plus_ape,
        compressor_kv_state_initial,
        compressor_score_state_initial,
        compressor_kv_state_after,
        compressor_score_state_after,
    ) = compressor_start0(
        input_bf16, compressor_wkv, compressor_wgate, compressor_ape
    )
    parts = [
        ("input_bf16", input_bf16.view(torch.uint16), "bf16-le", [ROWS, HIDDEN]),
        ("activation_fp8", activation.view(torch.uint8), "e4m3fn", [ROWS, HIDDEN]),
        ("activation_scale", activation_scale.view(torch.uint8), "e8m0fnu", [ROWS, HIDDEN // BLOCK]),
        ("wq_f32", wq_f32, "f32-le", [ROWS, 1536]),
        ("wq_bf16", wq_bf16.view(torch.uint16), "bf16-le", [ROWS, 1536]),
        ("q_norm_bf16", q_norm.view(torch.uint16), "bf16-le", [ROWS, 1536]),
        ("wkv_f32", wkv_f32, "f32-le", [ROWS, 512]),
        ("wkv_bf16", wkv_bf16.view(torch.uint16), "bf16-le", [ROWS, 512]),
        ("kv_norm_bf16", kv_norm.view(torch.uint16), "bf16-le", [ROWS, 512]),
        (
            "wq_b_activation",
            wq_b_activation.view(torch.uint8),
            "e4m3fn",
            [ROWS, 1536],
        ),
        (
            "wq_b_activation_scale",
            wq_b_activation_scale.view(torch.uint8),
            "e8m0fnu",
            [ROWS, 12],
        ),
        ("wq_b_f32", wq_b_f32, "f32-le", [ROWS, 65536]),
        (
            "wq_b_bf16",
            wq_b_bf16.view(torch.uint16),
            "bf16-le",
            [ROWS, 65536],
        ),
        (
            "q_head_rms_bf16",
            q_head_rms.view(torch.uint16),
            "bf16-le",
            [1, ROWS, 128, 512],
        ),
        (
            "freqs_cis_f32_pairs",
            torch.view_as_real(freqs_cis),
            "f32-le",
            [ROWS, 32, 2],
        ),
        (
            "q_rope_bf16",
            q_rope.view(torch.uint16),
            "bf16-le",
            [1, ROWS, 128, 512],
        ),
        (
            "kv_rope_bf16",
            kv_rope.view(torch.uint16),
            "bf16-le",
            [1, ROWS, 512],
        ),
        (
            "kv_scale_e8m0",
            kv_scale.view(torch.uint8),
            "e8m0fnu",
            [1, ROWS, 7],
        ),
        (
            "kv_ready_bf16",
            kv_ready.view(torch.uint16),
            "bf16-le",
            [1, ROWS, 512],
        ),
        ("causal_topk_i32", causal_topk, "i32-le", [1, ROWS, ROWS]),
        (
            "cache_initial_bf16",
            cache_initial.view(torch.uint16),
            "bf16-le",
            [1, 160, 512],
        ),
        (
            "cache_after_bf16",
            cache_after.view(torch.uint16),
            "bf16-le",
            [1, 160, 512],
        ),
        (
            "compressor_kv_f32",
            compressor_kv,
            "f32-le",
            [1, ROWS, 512],
        ),
        (
            "compressor_score_f32",
            compressor_score,
            "f32-le",
            [1, ROWS, 512],
        ),
        (
            "compressor_score_plus_ape_f32",
            compressor_score_plus_ape,
            "f32-le",
            [1, ROWS, 512],
        ),
        (
            "compressor_kv_state_initial_f32",
            compressor_kv_state_initial,
            "f32-le",
            [1, 128, 512],
        ),
        (
            "compressor_score_state_initial_f32",
            compressor_score_state_initial,
            "f32-le",
            [1, 128, 512],
        ),
        (
            "compressor_kv_state_after_f32",
            compressor_kv_state_after,
            "f32-le",
            [1, 128, 512],
        ),
        (
            "compressor_score_state_after_f32",
            compressor_score_state_after,
            "f32-le",
            [1, 128, 512],
        ),
    ]
    binary = bytearray(MAGIC)
    binary_meta = {}
    for name, tensor, dtype, shape in parts:
        payload = raw(tensor)
        binary_meta[name] = {
            "offset": len(binary),
            "bytes": len(payload),
            "dtype": dtype,
            "shape": shape,
        }
        binary.extend(payload)

    meta = {
        "schema": "colibri-v4-fp8-projection-pytorch-oracle-v4",
        "model_revision": REVISION,
        "ids": IDS,
        "semantics": {
            "activation": "BF16 groups of 128; max(abs) clamped to 1e-4; ceil-power-of-two E8M0 scale; E4M3FN RNE",
            "projection": "FP32 dot per K=128 block; multiply activation and 128x128 weight E8M0 scales; accumulate blocks in K order; cast BF16",
            "norm": "FP32 mean-square and rsqrt with eps=1e-6; multiply BF16 checkpoint weight in FP32; cast BF16",
            "wq_b_boundary": "q_norm BF16 is independently requantized before wq_b; wq_b output boundary is raw FP32 then BF16 with no following op in this fixture",
            "kv_boundary": "fixed kv_norm BF16; RoPE only last 64 adjacent pairs; first 448 fake-quantized in 64-wide E4M3FN/E8M0 blocks; causal topk; one contiguous capacity-1 cache write then stop",
            "compressor_boundary": "start_pos=0, bsz=1, seqlen=4, ratio=128, overlap=false; FP32 F.linear wkv/wgate; only remainder rows commit to paired FP32 states; should_compress=false; stop before norm/RoPE/quant/cache/attention",
        },
        "environment": {
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "device": "cpu",
            "threads": 1,
        },
        "source": {
            "shard": shard.name,
            "input_state_norm_sha256": state_norm_hash,
            "wq_a": {
                "weight": "layers.0.attn.wq_a.weight",
                "weight_shape": [1536, HIDDEN],
                "weight_sha256": sha(wq),
                "scale": "layers.0.attn.wq_a.scale",
                "scale_shape": [12, 56],
                "scale_sha256": sha(wq_scale),
                "norm": "layers.0.attn.q_norm.weight",
                "norm_sha256": sha(q_norm_weight),
            },
            "wq_b": {
                "weight": "layers.0.attn.wq_b.weight",
                "weight_shape": [65536, 1536],
                "weight_sha256": sha(wq_b),
                "scale": "layers.0.attn.wq_b.scale",
                "scale_shape": [512, 12],
                "scale_sha256": sha(wq_b_scale),
            },
            "wkv": {
                "weight": "layers.0.attn.wkv.weight",
                "weight_shape": [512, HIDDEN],
                "weight_sha256": sha(wkv),
                "scale": "layers.0.attn.wkv.scale",
                "scale_shape": [4, 56],
                "scale_sha256": sha(wkv_scale),
                "norm": "layers.0.attn.kv_norm.weight",
                "norm_sha256": sha(kv_norm_weight),
            },
            "compressor": {
                "wkv": "layers.0.attn.compressor.wkv.weight",
                "wkv_shape": [512, HIDDEN],
                "wkv_sha256": sha(compressor_wkv),
                "wgate": "layers.0.attn.compressor.wgate.weight",
                "wgate_shape": [512, HIDDEN],
                "wgate_sha256": sha(compressor_wgate),
                "ape": "layers.0.attn.compressor.ape",
                "ape_shape": [128, 512],
                "ape_sha256": sha(compressor_ape),
            },
        },
        "hashes": {name: sha(tensor) for name, tensor, _, _ in parts},
        "cache_contract": {
            "capacity_4_initial_sha256": "ff6335069b6e140eb47149d847aea80bf7e2b06bd80ae9708aa382efb3ae21ee",
            "capacity_4_after_sha256": "525ba206025f57ccb2459086e2abd6d571551d61c37be9ca45067f784717b45c",
            "capacity_1_untouched_tail_sha256": "30d4da31db85877f0102e0d56b9559e9d4400fde83c253a19c7c1988b4ed0c64",
            "capacity_1_compressed_view_sha256": "c35020473aed1b4642cd726cad727b63fff2824ad68cedd7ffb73c7cbd890479",
        },
        "compressor_contract": {
            "start_pos": 0,
            "batch_size": 1,
            "sequence_len": ROWS,
            "compress_ratio": 128,
            "overlap": False,
            "should_compress": False,
            "written_byte_range": [0, ROWS * 512 * 4],
            "kv_state_tail_sha256": sha(
                compressor_kv_state_after[:, ROWS:]
            ),
            "score_state_tail_sha256": sha(
                compressor_score_state_after[:, ROWS:]
            ),
        },
        "binary": {
            "magic": MAGIC.decode(),
            "sha256": hashlib.sha256(binary).hexdigest(),
        }
        | binary_meta,
    }
    rendered = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    if args.check:
        if (
            not args.json.is_file()
            or not args.bin.is_file()
            or args.json.read_text() != rendered
            or args.bin.read_bytes() != binary
        ):
            raise SystemExit("fixture differs; regenerate without --check")
        print("PASS: FP8 projection fixture reproduces exactly")
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered)
        args.bin.write_bytes(binary)
        print(f"wrote {args.json} and {args.bin}")


if __name__ == "__main__":
    main()
