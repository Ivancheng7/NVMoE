#!/usr/bin/env python3
"""Compare two official layer cutpoint directories tensor by tensor.

`v4_flash_0731_prefix_oracle.py run --cutpoint-dir DIR --cutpoint-layer N`
writes one raw tensor per intermediate plus a manifest. Two runs of the
official path do not agree on `sm_120`, so this script says where they start
to differ and by how much.

Magnitude matters more than the fact of a difference. A bf16 difference of a
few ULP is reduction-order noise. A large difference means the two runs
computed something different, and the tensor named here is where to look.

Usage:
    compare_oracle_cutpoints.py A_CUTPOINT_DIR B_CUTPOINT_DIR [--limit N]

Exit status is 0 when every tensor agrees, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DTYPES = {
    "BF16": np.uint16,
    "F32": np.float32,
    "I32": np.int32,
    "INT32": np.int32,
}


def load_tensor(directory, entry):
    raw = (directory / entry["file"]).read_bytes()
    dtype = entry["dtype"]
    if dtype not in DTYPES:
        raise ValueError(f"unhandled dtype {dtype}")
    return np.frombuffer(raw, dtype=DTYPES[dtype])


def ulp_distance(left, right, dtype):
    """Bit-pattern distance. Only meaningful for same-sign near values."""
    if dtype == "F32":
        a = left.view(np.uint32).astype(np.int64)
        b = right.view(np.uint32).astype(np.int64)
        return np.abs(a - b)
    if dtype == "BF16":
        a = left.astype(np.int64)
        b = right.astype(np.int64)
        return np.abs(a - b)
    return None


def describe(name, entry, left, right):
    same = np.array_equal(left, right)
    dtype = entry["dtype"]
    if same:
        return f"  {name:<28} identical"

    if dtype == "F32":
        a = left.astype(np.float64)
        b = right.astype(np.float64)
        nonzero = int(np.count_nonzero(a != b))
        diff = np.abs(a - b)
        rel = diff / np.maximum(np.abs(a), 1e-30)
        # Reduction-order noise shows up as a relative difference near fp32
        # epsilon. Anything past 1e-3 is a different computation, not a
        # different order.
        over = [int((rel > t).sum()) for t in (1e-6, 1e-3, 1e-2)]
        return (
            f"  {name:<28} DIFF {nonzero}/{a.size} elements"
            f"  max_abs={float(diff.max()):.3e}"
            f"  max_rel={float(rel.max()):.3e}"
            f"  n(rel>1e-6)={over[0]} n(rel>1e-3)={over[1]} n(rel>1e-2)={over[2]}"
        )

    if dtype == "BF16":
        ulp = ulp_distance(left, right, dtype)
        nonzero = np.count_nonzero(ulp)
        a = left.view(np.uint16).astype(np.float64)
        b = right.view(np.uint16).astype(np.float64)
        return (
            f"  {name:<28} DIFF {nonzero}/{a.size} elements"
            f"  max_ulp={int(ulp.max())}"
        )

    ulp = ulp_distance(left, right, dtype)
    nonzero = np.count_nonzero(left != right)
    return f"  {name:<28} DIFF {nonzero}/{left.size} elements  max_abs={int(np.max(ulp))}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=64,
        help="stop after this many differing tensors (0 means no limit)",
    )
    args = parser.parse_args()

    left_manifest = json.loads((args.left / "manifest.json").read_text())
    right_manifest = json.loads((args.right / "manifest.json").read_text())
    order = left_manifest["tensor_order"]

    print(f"left : {args.left}")
    print(f"right: {args.right}")
    print(f"layer: {left_manifest['layer']}  tensors: {len(order)}")
    print()

    differences = 0
    for name in order:
        entry = left_manifest["tensors"][name]
        left = load_tensor(args.left, entry)
        right = load_tensor(args.right, right_manifest["tensors"][name])
        if left.shape != right.shape:
            print(f"  {name:<28} SHAPE {left.shape} vs {right.shape}")
            differences += 1
            continue
        line = describe(name, entry, left, right)
        print(line)
        if "DIFF" in line or "SHAPE" in line:
            differences += 1
            if args.limit and differences >= args.limit:
                print(f"  ... stopping after {differences} differing tensors")
                break

    print()
    if differences:
        print(f"{differences} tensor(s) differ.")
        return 1
    print("all tensors identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
