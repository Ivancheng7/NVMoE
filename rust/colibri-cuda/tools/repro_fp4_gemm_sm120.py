#!/usr/bin/env python3
"""Minimal reproduction of the sm_120 FP4 GEMM defect.

The routed experts in Flash-0731 go through `kernel.fp4_gemm`, a TileLang
kernel that casts FP4 weights to FP8 and does an FP8xFP8 GEMM with an FP32
accumulator. On `sm_120` that kernel does not return the same answer twice,
and the answers it returns are far from the mathematically correct one.

This script isolates that single call. It builds one FP8 activation and one
FP4 weight, runs the official kernel several times, and compares every run
against a plain PyTorch reference that computes the same product without any
TileLang kernel:

    C[m, n] = sum_k ( A[m, k] * s_a[m, k // 128] ) * ( B[n, k] * s_b[n, k // 32] )

A disagreement of a few fp32 ULP would be reduction order. The measured
disagreement is many orders of magnitude larger, which is what makes this a
defect rather than a scheduling artifact.

Usage:
    repro_fp4_gemm_sm120.py --inference-dir DIR [--runs N] [--M 1] [--N 2048] [--K 4096]

TileLang compiles its kernels with nvcc at first use, so the glibc/CUDA header
overlay must be on the include path or the compile fails on `rsqrt`:

    CPATH=/srv/localmoe/compat-headers:/srv/localmoe/toolchains/glibc243-cuda13-math-overlay-v1/include

A populated `/root/.tilelang/cache` can hide this requirement for kernel
variants that were compiled before.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# E2M1 code points, indexed by the low nibble of the byte. The order is the
# standard FP4 table: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and the same values negated.
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
)


def unpack_fp4(packed):
    """[N, K//2] float4_e2m1fn_x2 -> [N, K] float32.

    Two FP4 values share a byte; the low nibble comes first along K.
    """
    raw = packed.view(torch.uint8)
    low = (raw & 0x0F).to(torch.int64)
    high = ((raw >> 4) & 0x0F).to(torch.int64)
    table = E2M1.to(packed.device)
    out = torch.stack((table[low], table[high]), dim=-1)
    return out.reshape(*packed.shape[:-1], -1).to(torch.float32)


def repeat_scale(scale, block, k):
    """[..., K // block] -> [..., K] by repeating each scale `block` times."""
    return scale.to(torch.float32).repeat_interleave(block, dim=-1)[..., :k]


def reference_fp4_gemm(a, a_s, b, b_s, act_block=128, weight_block=32):
    """The mathematically correct product, in fp32, without any TileLang."""
    m, k = a.shape
    a_f = a.to(torch.float32) * repeat_scale(a_s, act_block, k)
    b_f = unpack_fp4(b) * repeat_scale(b_s, weight_block, k)
    return a_f @ b_f.t()


def report(name, official_runs, ref):
    print(f"  {name}")
    for index, value in enumerate(official_runs):
        diff = (value - ref).abs()
        rel = diff / ref.abs().clamp_min(1e-30)
        print(
            f"    run {index}: max_abs={diff.max().item():.3e}"
            f"  max_rel={rel.max().item():.3e}"
            f"  n(rel>1e-3)={int((rel > 1e-3).sum())}/{ref.numel()}"
        )
    if len(official_runs) > 1:
        spread = (official_runs[0] - official_runs[1]).abs()
        print(f"    run-to-run max_abs={spread.max().item():.3e}")
    ref_spread = ref.abs().max().item()
    print(f"    reference scale max|C|={ref_spread:.3e}")


def uniform_nibble_control(kernel, m, n, k, device, runs):
    """Same comparison with a weight whose two nibbles are equal.

    Every byte is 0x22, so both halves decode to the same value and the result
    does not depend on which nibble comes first along K. If the kernel still
    disagrees with the reference here, the disagreement cannot be blamed on
    the reference's nibble order.
    """
    torch.manual_seed(1234)
    activation = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    a, a_s = kernel.act_quant(activation, 128, "ue8m0", torch.float8_e8m0fnu)

    packed = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device=device)
    b = packed.view(torch.float4_e2m1fn_x2)
    # E8M0 scale 127 is exactly 2**0 = 1.0.
    b_s = torch.full((n, k // 32), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)

    # The control only means something if these two decodings hold, so assert
    # them rather than trusting the comments above. Nibble 0b0010 is 1.0:
    # E2M1 is [sign:1][exponent:2][mantissa:1], so 0b0100 would be 2.0.
    decoded_weight = unpack_fp4(b[:1, :2])
    if not torch.all(decoded_weight == 1.0):
        raise AssertionError(f"0x22 should decode to 1.0, got {decoded_weight.tolist()}")
    if b_s[0, 0].to(torch.float32).item() != 1.0:
        raise AssertionError(f"E8M0 byte 127 should be 1.0, got {b_s[0, 0].item()}")

    ref = reference_fp4_gemm(a, a_s, b, b_s)
    torch.cuda.synchronize()
    outs = []
    for _ in range(runs):
        value = kernel.fp4_gemm(a, a_s, b, b_s, torch.float8_e8m0fnu).to(torch.float32)
        torch.cuda.synchronize()
        outs.append(value)
    return ref, outs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--N", type=int, default=2048)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.inference_dir))
    import kernel

    # fp4_gemm allocates its output with torch.get_default_dtype() and the
    # kernel only accepts bf16 there, so the default has to be bf16.
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(args.seed)
    device = "cuda"
    m, n, k = args.M, args.N, args.K

    # Inputs are built with the official quantizers so both paths see exactly
    # the same bytes. Whether those quantizers are themselves correct is a
    # separate question; it cannot affect a comparison between two GEMMs.
    activation = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    a, a_s = kernel.act_quant(activation, 128, "ue8m0", torch.float8_e8m0fnu)

    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    b, b_s = kernel.fp4_act_quant(weight, 32, False)

    print(f"shapes: M={m} N={n} K={k}")
    print(f"  a={tuple(a.shape)} {a.dtype}  a_s={tuple(a_s.shape)} {a_s.dtype}")
    print(f"  b={tuple(b.shape)} {b.dtype}  b_s={tuple(b_s.shape)} {b_s.dtype}")

    ref = reference_fp4_gemm(a, a_s, b, b_s)
    torch.cuda.synchronize()

    runs = []
    for _ in range(args.runs):
        out = kernel.fp4_gemm(a, a_s, b, b_s, torch.float8_e8m0fnu).to(torch.float32)
        torch.cuda.synchronize()
        runs.append(out)

    print()
    report("official fp4_gemm vs fp32 reference", runs, ref)

    print()
    control_ref, control_runs = uniform_nibble_control(kernel, m, n, k, device, args.runs)
    report("uniform-nibble control (nibble order cannot matter)", control_runs, control_ref)

    return 0


if __name__ == "__main__":
    sys.exit(main())
