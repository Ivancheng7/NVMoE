#!/usr/bin/env python3
"""Emit the compact CPU PyTorch parity fixture for pinned DeepSeek-V4-Pro routing.

Run with the isolated oracle environment, for example:
  "$COLIBRI_ORACLE_PYTHON" tools/make_v4_router_oracle.py \\
      --model-dir "$COLIBRI_DSV4_MODEL_DIR"
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import safetensors
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open


REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
HIDDEN, EXPERTS, TOPK, ROWS = 7168, 384, 6, 4
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "fixtures/v4_router_pytorch_oracle.json"


def sha256(tensor):
    """Hash exact tensor bytes without adding NumPy to the oracle environment."""
    raw = tensor.contiguous().view(torch.uint8).reshape(-1).tolist()
    return hashlib.sha256(bytes(raw)).hexdigest()


def deterministic_bf16(seed):
    index = torch.arange(ROWS * HIDDEN, dtype=torch.int64)
    values = ((index * 17 + seed) % 113).to(torch.float32).sub(56).div(64)
    return values.reshape(2, 2, HIDDEN).to(torch.bfloat16)


def rmsnorm(x_pre, norm):
    inv_rms = torch.rsqrt(x_pre.float().square().mean(-1, keepdim=True) + 1e-6)
    return (x_pre.float() * norm.float() * inv_rms).to(torch.bfloat16)


def selected_rows(ids, raw, logits, scores):
    weights = raw / raw.sum(-1, keepdim=True) * 2.5
    return {
        "ids": ids.tolist(),
        "weights": [[float(value) for value in row] for row in weights.tolist()],
        "raw_scores": [[float(value) for value in row] for row in raw.tolist()],
        "logits": [[float(value) for value in row] for row in logits.tolist()],
        "selection_scores": [[float(value) for value in row] for row in scores.tolist()],
    }


def load_case(model_dir, layer, seed, mode):
    shard = model_dir / ("model-00002-of-00064.safetensors" if layer == 0 else "model-00005-of-00064.safetensors")
    prefix = f"layers.{layer}.ffn"
    names = {"norm": f"{prefix}_norm.weight", "gate": f"{prefix}.gate.weight"}
    x_pre = deterministic_bf16(seed)
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        norm = tensors.get_tensor(names["norm"])
        gate = tensors.get_tensor(names["gate"])
        x_norm = rmsnorm(x_pre, norm)
        logits = F.linear(x_norm.float(), gate.float()).reshape(ROWS, EXPERTS)
        raw_all = F.softplus(logits, beta=1, threshold=20).sqrt()
        hashes = {"x_pre_bf16": sha256(x_pre), "norm_bf16": sha256(norm), "gate_bf16": sha256(gate),
                  "x_norm_bf16": sha256(x_norm), "logits_f32": sha256(logits)}
        if mode == "hash":
            ids = torch.cat((tensors.get_slice(f"{prefix}.gate.tid2eid")[0:3],
                             tensors.get_slice(f"{prefix}.gate.tid2eid")[129279:129280]))
            raw = raw_all.gather(1, ids)
            hashes["tid2eid_rows_i64"] = sha256(ids)
            result = selected_rows(ids, raw, logits.gather(1, ids), raw)
            result["table_rows"] = [0, 1, 2, 129279]
        else:
            bias = tensors.get_tensor(f"{prefix}.gate.bias")
            scores = raw_all + bias.float()
            top7_scores, top7_ids = torch.topk(scores, TOPK + 1, dim=-1, sorted=True)
            ids = top7_ids[:, :TOPK]
            raw = raw_all.gather(1, ids)
            hashes["bias_f32"] = sha256(bias)
            result = selected_rows(ids, raw, logits.gather(1, ids), top7_scores[:, :TOPK])
            result["sixth_seventh_margins"] = [
                {"sixth": float(row[5]), "seventh": float(row[6]), "margin": float(row[5] - row[6])}
                for row in top7_scores.tolist()
            ]
    return {
        "layer": layer,
        "shard": shard.name,
        "mode": mode,
        "input_seed": seed,
        "tensors": names | ({"tid2eid": f"{prefix}.gate.tid2eid"} if mode == "hash" else {"bias": f"{prefix}.gate.bias"}),
        "hashes": hashes,
        "output": result,
    }


def make_fixture(model_dir):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "schema": "colibri-v4-router-pytorch-oracle-v1",
        "model_revision": REVISION,
        "shape": {"batch": 2, "sequence": 2, "hidden": HIDDEN, "routed_experts": EXPERTS, "topk": TOPK},
        "x_pre": "BF16 RNE of (((flat_index * 17 + seed) % 113) - 56) / 64; flattened B=2,S=2 order",
        "semantics": {
            "rmsnorm": "float32 mean(x_pre^2)+1e-6, x_pre*norm*rsqrt, then BF16 RNE",
            "projection": "torch.nn.functional.linear(x_norm.float(), gate.float())",
            "score": "torch.nn.functional.softplus(logits, beta=1, threshold=20).sqrt()",
            "hash": "tid2eid table order; normalize gathered raw scores then multiply 2.5",
            "learned": "topk(raw_score+bias, sorted descending); normalize gathered raw scores then multiply 2.5",
        },
        "environment": {
            "device": "cpu",
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "cases": [load_case(model_dir, 0, 7, "hash"), load_case(model_dir, 3, 31, "learned")],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="fail if --out differs from regenerated output")
    parser.add_argument("--stdout", action="store_true", help="print JSON instead of writing --out")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error(f"missing model directory: {args.model_dir}")
    rendered = json.dumps(make_fixture(args.model_dir), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text() != rendered:
            raise SystemExit(f"fixture differs: regenerate {args.out}")
        print(f"PASS: {args.out} reproduces exactly")
    elif args.stdout:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
