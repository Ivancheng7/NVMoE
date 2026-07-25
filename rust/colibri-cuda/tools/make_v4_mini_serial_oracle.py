#!/usr/bin/env python3
"""Independent tiny V4 serial oracle; never imports or executes Rust.

Formula sources in pinned inference/model.py:
  Compressor 279-377; Indexer 380-433; Attention ordering 484-539;
  hash/learned Gate 550-608; Block attention-then-MoE 652-687;
  final norm/head/logits 769-832.
The dimensions are deliberately tiny. This fixture certifies ordering and state
contracts, not numerical parity with the 7168-wide checkpoint.
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import torch
import torch.nn.functional as F


REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
SCHEMA = "colibri-v4-mini-serial-oracle-v1"
OUT = Path(__file__).resolve().parent.parent / "fixtures/v4_mini_serial_oracle_v1.json"
DIM, EXPERTS, TOPK, ROUTE_SCALE = 8, 8, 2, 2.5


def raw(tensor):
    return bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())


def fnv(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def digest(tensor):
    data = raw(tensor)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "fnv1a64": fnv(data)}


def rms(x, weight):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * weight).to(torch.bfloat16)


def select(logits, bias, hash_ids):
    raw_score = F.softplus(logits.float()).sqrt()
    ids = hash_ids if hash_ids is not None else torch.topk(raw_score + bias, TOPK, sorted=True).indices
    selected = raw_score[ids]
    return ids, selected / selected.sum() * ROUTE_SCALE


def compressor(x, ratio, seed):
    generator = torch.Generator().manual_seed(seed)
    width = DIM * (2 if ratio == 4 else 1)
    kv = x.float() @ torch.randn(width, DIM, generator=generator).T / DIM
    score = x.float() @ torch.randn(width, DIM, generator=generator).T / DIM
    ape = torch.randn(ratio, width, generator=generator) / 16
    state = torch.zeros(ratio * (2 if ratio == 4 else 1), width)
    score_state = torch.full_like(state, float("-inf"))
    take = min(x.shape[0], ratio)
    offset = ratio if ratio == 4 else 0
    state[offset : offset + take] = kv[:take]
    score_state[offset : offset + take] = score[:take] + ape[:take]
    return state, score_state


def compressor_projection_contract():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    cases = []
    for name, outputs, ratio, start_pos in [
        ("hca_main", 512, 128, 127),
        ("csa_main", 1024, 4, 3),
        ("csa_indexer", 256, 4, 3),
    ]:
        wkv = torch.tensor([1, 2], dtype=torch.bfloat16).repeat(outputs, 1)
        wgate = torch.tensor([2, -1], dtype=torch.bfloat16).repeat(outputs, 1)
        ape = torch.arange(ratio, dtype=torch.float32).unsqueeze(1).expand(ratio, outputs)
        kv = x.float() @ wkv.float().T
        score = x.float() @ wgate.float().T
        phases = (torch.arange(x.shape[0]) + start_pos) % ratio
        score_plus_ape = score + ape[phases]
        cases.append({
            "name": name,
            "outputs": outputs,
            "ratio": ratio,
            "start_pos": start_pos,
            "kv_f32": digest(kv),
            "score_f32": digest(score),
            "score_plus_ape_f32": digest(score_plus_ape),
            "row_values": {
                "kv": kv[:, 0].tolist(),
                "score": score[:, 0].tolist(),
                "score_plus_ape": score_plus_ape[:, 0].tolist(),
            },
        })
    return {"hidden": 2, "rows": 2, "cases": cases}


def compressor_boundary_contract():
    hca_kv = torch.arange(1, 129, dtype=torch.float32).unsqueeze(1)
    hca_score = torch.zeros_like(hca_kv)
    hca_pool = (hca_kv * hca_score.softmax(0)).sum(0)

    previous = torch.arange(100, 104, dtype=torch.float32).unsqueeze(1)
    current = torch.arange(5, 9, dtype=torch.float32).unsqueeze(1)
    csa_kv = torch.cat([previous, current], dim=0)
    csa_score = torch.zeros_like(csa_kv)
    csa_pool = (csa_kv * csa_score.softmax(0)).sum(0)
    return {
        "hca_ratio128_pool_f32_bits": hca_pool.view(torch.int32).tolist(),
        "csa_overlap_pool_f32_bits": csa_pool.view(torch.int32).tolist(),
    }


def layer(hidden, layer, ratio, router, seed):
    generator = torch.Generator().manual_seed(seed)
    attn_norm = torch.randn(DIM, generator=generator).abs() + 0.5
    q = rms(hidden, attn_norm).float()
    attention = q @ (torch.randn(DIM, DIM, generator=generator) / DIM).T
    residual = (hidden.float() + attention).to(torch.bfloat16)
    kv_state, score_state = compressor(residual, ratio, seed + 1)

    indexer = None
    topk = []
    if ratio == 4:
        q_index = residual.float() @ (torch.randn(4, DIM, generator=generator) / DIM).T
        kv_index = residual.float() @ (torch.randn(4, DIM, generator=generator) / DIM).T
        head_weight = residual.float() @ (torch.randn(1, DIM, generator=generator) / DIM).T
        scores = torch.relu(q_index @ kv_index.T) * head_weight
        valid = torch.arange(hidden.shape[0]).unsqueeze(1) >= torch.arange(hidden.shape[0]).add(1).unsqueeze(0)
        scores = scores.masked_fill(valid, float("-inf"))
        topk = [torch.topk(row[: index + 1], min(2, index + 1)).indices.tolist() for index, row in enumerate(scores)]
        indexer = {"kv_state": digest(kv_index), "score": digest(scores), "topk": topk}

    ffn_norm = torch.randn(DIM, generator=generator).abs() + 0.5
    routed = rms(residual, ffn_norm).float()
    logits = routed @ (torch.randn(EXPERTS, DIM, generator=generator) / DIM).T
    if router == "hash":
        hash_ids = torch.tensor([6, 1])
        bias = torch.zeros(EXPERTS)
    else:
        hash_ids = None
        bias = torch.linspace(-0.25, 0.25, EXPERTS)
        bias[1] += 2
    ids, weights = select(logits[-1], bias, hash_ids)
    expert_scale = torch.linspace(0.25, 1.0, EXPERTS)
    routed_sum = sum((residual.float() * expert_scale[index]) * weight for index, weight in zip(ids, weights))
    shared = residual.float() * 0.125
    output = (residual.float() + routed_sum + shared).to(torch.bfloat16)
    return output, {
        "layer": layer,
        "ratio": ratio,
        "router": router,
        "hidden_bf16": digest(output),
        "kv_cache_bf16": digest(residual),
        "compressor_kv_state": digest(kv_state),
        "compressor_score_state": digest(score_state),
        "indexer": indexer,
        "route_ids": ids.tolist(),
        "route_weights_f32": [float(value) for value in weights],
    }


def build():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, DIM).sub(16).div(16).to(torch.bfloat16)
    initial = digest(hidden)
    layers = []
    hidden, record = layer(hidden, 0, 128, "hash", 202607240)
    layers.append(record)
    first_output = hidden.clone()
    hidden, record = layer(hidden, 2, 4, "learned", 202607242)
    layers.append(record)
    assert digest(first_output) == layers[0]["hidden_bf16"]
    hc_hidden = torch.stack([(hidden.float() * (copy + 1) / 4).to(torch.bfloat16) for copy in range(4)], dim=1)
    hc_fn = torch.zeros(4, 4 * DIM)
    hc_scale = torch.ones(1)
    hc_base = torch.zeros(4)
    flattened = hc_hidden.flatten(1).float()
    mixes = F.linear(flattened, hc_fn) * torch.rsqrt(flattened.square().mean(-1, keepdim=True) + 1e-6)
    hc_pre = torch.sigmoid(mixes * hc_scale + hc_base) + 1e-6
    collapsed = (hc_hidden.float() * hc_pre.unsqueeze(-1)).sum(1).to(torch.bfloat16)
    final_weight = torch.linspace(0.75, 1.25, DIM)
    normalized = rms(collapsed, final_weight)
    head_weight = torch.zeros(11, DIM)
    for row in range(11):
        head_weight[row, row % DIM] = (row + 1) / 8
    logits = normalized.float() @ head_weight.T
    assert layers[0]["ratio"] == 128 and layers[0]["indexer"] is None
    assert layers[1]["ratio"] == 4 and layers[1]["indexer"] is not None
    assert layers[0]["router"] == "hash" and layers[1]["router"] == "learned"
    return {
        "schema": SCHEMA,
        "model_revision": REVISION,
        "independence": "PyTorch CPU formulas only; no Rust import, execution, or output",
        "dimensions": {"tokens": 4, "hidden": DIM, "experts": EXPERTS, "topk": TOPK},
        "compressor_projection_contract": compressor_projection_contract(),
        "compressor_boundary_contract": compressor_boundary_contract(),
        "initial_hidden_bf16": initial,
        "layers": layers,
        "final": {
            "contract": {
                "rows": hidden.shape[0],
                "hc": 4,
                "dim": DIM,
                "vocab": 11,
                "hc_hidden_bf16_bits": hc_hidden.view(torch.int16).to(torch.int32).reshape(-1).tolist(),
                "hc_fn_f32": hc_fn.reshape(-1).tolist(),
                "hc_scale_f32": float(hc_scale[0]),
                "hc_base_f32": hc_base.tolist(),
                "hc_pre_f32": hc_pre.reshape(-1).tolist(),
                "norm_weight_f32": final_weight.tolist(),
                "head_weight_f32": head_weight.reshape(-1).tolist(),
                "collapsed_bf16_bits": collapsed.view(torch.int16).to(torch.int32).reshape(-1).tolist(),
                "norm_bf16_bits": normalized.view(torch.int16).to(torch.int32).reshape(-1).tolist(),
                "logits_f32_bits": logits.view(torch.int32).reshape(-1).tolist(),
            },
            "norm_bf16": digest(normalized),
            "logits_f32": digest(logits),
            "argmax": logits.argmax(-1).tolist(),
        },
        "environment": {"torch": torch.__version__, "threads": torch.get_num_threads()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or json.loads(args.out.read_text()) != json.loads(rendered):
            raise SystemExit(f"fixture differs: regenerate {args.out}")
        print(f"PASS: {args.out.name} reproduces exactly")
    elif args.stdout:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"WROTE: {args.out}")


if __name__ == "__main__":
    main()
