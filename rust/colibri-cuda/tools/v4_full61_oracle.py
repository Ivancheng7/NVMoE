#!/usr/bin/env python3
"""Pinned official DeepSeek-V4-Pro full61 oracle and exact trace comparator.

The runner constructs one official Block at a time and streams only the six
routed experts selected for that layer. Peak ownership is one dense layer,
one routed expert and one 256-row output-head chunk. It requires CUDA with at
least 9 GiB free and 8 GiB available host RAM; it never constructs the full
checkpoint in RAM or VRAM.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import safe_open


REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
MODEL_BASENAME = f"deepseek-v4-pro-{REVISION}"
SCHEMA = "colibri-v4-full61-trace-v1"
PINNED_SHA256 = {
    "config.json": "5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e",
    "model.safetensors.index.json": "a3a39b9ccb4e729851922fc9c770f5c5755e7b9d7e96cd02c23f0e12b5e25cb9",
    "inference/model.py": "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71",
    "inference/kernel.py": "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2",
}
REQUIRED_LAYER_FIELDS = {
    "attention_topk_digest",
    "hidden_bf16_digest",
    "kv_cache_bf16_digest",
    "main_compressed_kv_bf16_digest",
    "main_compressor_kv_state_f32_digest",
    "main_compressor_score_state_f32_digest",
    "indexer_compressed_kv_bf16_digest",
    "indexer_compressor_kv_state_f32_digest",
    "indexer_compressor_score_state_f32_digest",
    "route_ids",
    "route_weights_f32",
}
REQUIRED_FINAL_FIELDS = {
    "collapsed_bf16_digest",
    "normalized_bf16_digest",
    "logits_f32_digest",
    "argmax",
}
ADA_SPARSE_LIMITATION = (
    "pinned TileLang sparse_attn requests 344064 dynamic shared-memory bytes "
    "for h=128,d=512; RTX 4080 Ada rejects that launch"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def raw_bytes(tensor):
    flat = tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    return bytes(flat.tolist())


def fnv1a(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def tensor_digest(tensor):
    return fnv1a(raw_bytes(tensor))


def empty_digest():
    return fnv1a(b"")


def reference_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Low-shared-memory reference for pinned inference/kernel.py sparse_attn.

    This follows the pinned kernel's index order and formulas: BF16 Q/KV,
    sequential FP32 dot products, score scaling, max-subtracted FP32 exp,
    BF16-cast probabilities for the numerator, FP32 accumulation, the learned
    sink term in the denominator, and a final BF16 output cast. It is
    independent of the Rust implementation and is only used because the
    pinned h=128,d=512 TileLang launch exceeds Ada shared memory.
    """
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise RuntimeError("reference sparse attention requires BF16 Q/KV")
    batch, sequence, heads, dimension = q.shape
    if kv.ndim != 3 or topk_idxs.shape[:2] != (batch, sequence):
        raise RuntimeError("reference sparse attention geometry mismatch")
    q_cpu = q.detach().cpu()
    kv_cpu = kv.detach().cpu()
    sink_cpu = attn_sink.detach().cpu().float()
    indices_cpu = topk_idxs.detach().cpu()
    output = torch.zeros_like(q_cpu)
    for batch_id in range(batch):
        for row_id in range(sequence):
            keys = [
                int(key)
                for key in indices_cpu[batch_id, row_id].tolist()
                if 0 <= int(key) < kv_cpu.size(1)
            ]
            if not keys:
                continue
            for head_id in range(heads):
                scores = []
                q_row = q_cpu[batch_id, row_id, head_id]
                for key in keys:
                    score = torch.zeros((), dtype=torch.float32, device="cpu")
                    kv_row = kv_cpu[batch_id, key]
                    for column in range(dimension):
                        score += q_row[column].float() * kv_row[column].float()
                    scores.append(score * softmax_scale)
                score_max = torch.stack(scores).max()
                denominator = torch.zeros((), dtype=torch.float32, device="cpu")
                numerator = torch.zeros(dimension, dtype=torch.float32, device="cpu")
                for score, key in zip(scores, keys):
                    probability = torch.exp(score - score_max)
                    denominator += probability
                    numerator += probability.to(torch.bfloat16).float() * kv_cpu[
                        batch_id, key
                    ].float()
                denominator += torch.exp(sink_cpu[head_id] - score_max)
                output[batch_id, row_id, head_id] = (numerator / denominator).to(
                    torch.bfloat16
                )
    return output.to(q.device)


def validate_sparse_fallback(official_sparse_attn):
    q = torch.linspace(-1.0, 1.0, 2 * 3 * 128, device="cuda", dtype=torch.bfloat16)
    q = q.reshape(1, 2, 3, 128)
    kv = torch.linspace(-0.75, 0.75, 5 * 128, device="cuda", dtype=torch.bfloat16)
    kv = kv.reshape(1, 5, 128)
    sink = torch.tensor([-0.5, 0.25, 0.75], device="cuda", dtype=torch.float32)
    indices = torch.tensor([[[0, 2, 4], [1, 3, -1]]], device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(128)
    expected = official_sparse_attn(q, kv, sink, indices, scale)
    actual = reference_sparse_attn(q, kv, sink, indices, scale)
    torch.cuda.synchronize()
    difference = (expected.float() - actual.float()).abs()
    bit_exact = raw_bytes(expected) == raw_bytes(actual)
    max_abs = float(difference.max())
    if not torch.allclose(expected.float(), actual.float(), rtol=0.015625, atol=0.015625):
        raise RuntimeError(f"Ada sparse fallback cross-check failed: max_abs={max_abs}")
    return {
        "small_shape": {"batch": 1, "sequence": 2, "heads": 3, "dimension": 128},
        "bit_exact": bit_exact,
        "max_abs": max_abs,
        "gate": "bit-exact" if bit_exact else "canonical-compatible rtol=atol=0.015625",
    }


def validate_storage(model_dir):
    model_dir = model_dir.resolve(strict=True)
    if model_dir.name != MODEL_BASENAME or str(model_dir).startswith(("/mnt/c/", "/mnt/d/")):
        raise RuntimeError("oracle model directory is not the pinned native-ext4 snapshot")
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
    for relative, expected in PINNED_SHA256.items():
        path = model_dir / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"pinned source hash mismatch: {relative}")
    config = json.loads((model_dir / "config.json").read_text())
    expected_config = {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "vocab_size": 129280,
        "num_attention_heads": 128,
        "num_experts_per_tok": 6,
        "n_routed_experts": 384,
        "index_topk": 1024,
        "max_position_embeddings": 1048576,
        "compress_rope_theta": 160000,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(f"pinned config mismatch: {key}")
    ratios = config.get("compress_ratios")
    if (
        len(ratios or []) != 62
        or ratios[:5] != [128, 128, 4, 128, 4]
        or ratios[60:] != [4, 0]
    ):
        raise RuntimeError("pinned 61-layer plus MTP compression schedule mismatch")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map", {})
    for layer in range(61):
        name = f"layers.{layer}.attn.wq_a.weight"
        expected = f"model-{layer + 2:05}-of-00064.safetensors"
        if weight_map.get(name) != expected:
            raise RuntimeError(f"pinned layer/shard mapping mismatch: {name}")
    if any(
        weight_map.get(name) != "model-00063-of-00064.safetensors"
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale", "norm.weight", "head.weight")
    ):
        raise RuntimeError("pinned final-head shard mapping mismatch")
    for number in range(1, 65):
        shard = model_dir / f"model-{number:05}-of-00064.safetensors"
        if not shard.is_file() or shard.is_symlink():
            raise RuntimeError(f"missing regular pinned shard: {shard.name}")
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            if not tensors.keys():
                raise RuntimeError(f"empty pinned shard: {shard.name}")
    return model_dir, config, {
        "mountpoint": best[0],
        "filesystem": best[1],
        "source": best[2],
    }


def validate_resources():
    version = tuple(
        int(part)
        for part in torch.__version__.split("+", 1)[0].split(".")[:2]
        if part.isdigit()
    )
    if version < (2, 10) or not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError(f"official oracle requires PyTorch >=2.10 with FP4, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"official oracle requires CUDA PyTorch, got {torch.__version__}")
    try:
        if importlib.metadata.version("tilelang") != "0.1.8":
            raise RuntimeError("official oracle requires tilelang==0.1.8")
        import fast_hadamard_transform  # noqa: F401
        import tilelang  # noqa: F401
    except Exception as error:
        raise RuntimeError(
            "official oracle requires a working pinned inference/requirements.txt environment"
        ) from error
    free_vram, _ = torch.cuda.mem_get_info()
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if free_vram < 9 * (1 << 30):
        raise RuntimeError(f"official streaming oracle requires 9 GiB free VRAM, got {free_vram}")
    if available_kib < 8 * (1 << 20):
        raise RuntimeError(
            f"official streaming oracle requires 8 GiB available RAM, got {available_kib} KiB"
        )
    return {"free_vram_bytes": free_vram, "available_ram_kib": available_kib}


def import_official(model_dir):
    inference = model_dir / "inference"
    sys.path.insert(0, str(inference))
    spec = importlib.util.spec_from_file_location("colibri_pinned_v4_model", inference / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def official_args(official, config):
    rope = config["rope_scaling"]
    values = {
        "max_batch_size": 1,
        "max_seq_len": 128,
        "dtype": "fp8",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "scale_dtype": "fp8",
        "vocab_size": config["vocab_size"],
        "dim": config["hidden_size"],
        "moe_inter_dim": config["moe_intermediate_size"],
        "n_layers": config["num_hidden_layers"],
        "n_hash_layers": config["num_hash_layers"],
        "n_mtp_layers": config["num_nextn_predict_layers"],
        "n_heads": config["num_attention_heads"],
        "n_routed_experts": config["n_routed_experts"],
        "n_shared_experts": config["n_shared_experts"],
        "n_activated_experts": config["num_experts_per_tok"],
        "score_func": config["scoring_func"],
        "route_scale": config["routed_scaling_factor"],
        "swiglu_limit": config["swiglu_limit"],
        "q_lora_rank": config["q_lora_rank"],
        "head_dim": config["head_dim"],
        "rope_head_dim": config["qk_rope_head_dim"],
        "norm_eps": config["rms_norm_eps"],
        "o_groups": config["o_groups"],
        "o_lora_rank": config["o_lora_rank"],
        "window_size": config["sliding_window"],
        "compress_ratios": tuple(config["compress_ratios"]),
        "compress_rope_theta": config["compress_rope_theta"],
        "original_seq_len": rope["original_max_position_embeddings"],
        "rope_theta": config["rope_theta"],
        "rope_factor": rope["factor"],
        "beta_fast": rope["beta_fast"],
        "beta_slow": rope["beta_slow"],
        "index_n_heads": config["index_n_heads"],
        "index_head_dim": config["index_head_dim"],
        "index_topk": config["index_topk"],
        "hc_mult": config["hc_mult"],
        "hc_sinkhorn_iters": config["hc_sinkhorn_iters"],
        "hc_eps": config["hc_eps"],
    }
    allowed = {field.name for field in dataclasses.fields(official.ModelArgs)}
    return official.ModelArgs(**{key: value for key, value in values.items() if key in allowed})


def copy_parameters(module, shard, prefix):
    source_bytes = 0
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        for name, parameter in module.named_parameters():
            tensor_name = f"{prefix}.{name}" if prefix else name
            if tensor_name not in keys:
                raise RuntimeError(f"official parameter missing from shard: {tensor_name}")
            value = tensors.get_tensor(tensor_name)
            source_bytes += value.numel() * value.element_size()
            if (
                parameter.dtype == torch.float4_e2m1fn_x2
                and value.dtype in (torch.int8, torch.uint8)
            ):
                parameter.data = value.view(torch.float4_e2m1fn_x2).to(parameter.device)
            else:
                parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            del value
    return source_bytes


def make_streaming_moe(official):
    class StreamingMoE(torch.nn.Module):
        def __init__(self, layer_id, args):
            super().__init__()
            self.layer_id = layer_id
            self.args = args
            self.gate = official.Gate(layer_id, args)
            self.shared_experts = official.Expert(
                args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
            )
            self.shard = None
            self.last_ids = None
            self.last_weights = None
            self.source_range_bytes = 0

        def forward(self, x, input_ids):
            shape = x.size()
            flat = x.view(-1, self.args.dim)
            weights, indices = self.gate(flat, input_ids.flatten())
            self.last_ids = indices.detach().cpu()
            self.last_weights = weights.detach().cpu()
            output = torch.zeros_like(flat, dtype=torch.float32)
            expert_dtype = torch.float4_e2m1fn_x2
            for expert_id in sorted(set(indices.flatten().tolist())):
                expert = official.Expert(
                    self.args.dim,
                    self.args.moe_inter_dim,
                    dtype=expert_dtype,
                    swiglu_limit=self.args.swiglu_limit,
                )
                self.source_range_bytes += copy_parameters(
                    expert,
                    self.shard,
                    f"layers.{self.layer_id}.ffn.experts.{expert_id}",
                )
                row, slot = torch.where(indices == expert_id)
                output[row] += expert(flat[row], weights[row, slot, None])
                del expert
            output += self.shared_experts(flat)
            return output.type_as(flat).view(shape)

    return StreamingMoE


def state_fields(block, ratio, token_count):
    attention = block.attn
    compressed = token_count // ratio
    main = attention.compressor
    indexer = attention.indexer
    # ponytail: start_pos=0 owns a contiguous prefix; canonicalize ring order
    # when the official oracle accepts replayed prefixes beyond the window.
    window_rows = min(token_count, attention.window_size)
    return {
        "kv_cache_bf16_digest": tensor_digest(attention.kv_cache[:, :window_rows]),
        "main_compressed_kv_bf16_digest": (
            tensor_digest(attention.kv_cache[:, attention.window_size : attention.window_size + compressed])
            if compressed
            else empty_digest()
        ),
        "main_compressor_kv_state_f32_digest": tensor_digest(main.kv_state),
        "main_compressor_score_state_f32_digest": tensor_digest(main.score_state),
        "indexer_compressed_kv_bf16_digest": (
            tensor_digest(indexer.kv_cache[:, :compressed]) if indexer and compressed else empty_digest()
        ),
        "indexer_compressor_kv_state_f32_digest": (
            tensor_digest(indexer.compressor.kv_state) if indexer else None
        ),
        "indexer_compressor_score_state_f32_digest": (
            tensor_digest(indexer.compressor.score_state) if indexer else None
        ),
    }


def final_record(model_dir, hidden):
    shard = model_dir / "model-00063-of-00064.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        fn = tensors.get_tensor("hc_head_fn").cuda()
        base = tensors.get_tensor("hc_head_base").cuda()
        scale = tensors.get_tensor("hc_head_scale").cuda()
        norm = tensors.get_tensor("norm.weight").cuda()
        flattened = hidden.flatten(2).float()
        inverse = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + 1e-6)
        mixes = torch.nn.functional.linear(flattened, fn) * inverse
        pre = torch.sigmoid(mixes * scale + base) + 1e-6
        collapsed = (pre.unsqueeze(-1) * hidden).sum(2).to(torch.bfloat16)
        normalized = (
            collapsed.float()
            * torch.rsqrt(collapsed.float().square().mean(-1, keepdim=True) + 1e-6)
            * norm
        ).to(torch.bfloat16)
        fnv = 0xCBF29CE484222325
        sha = hashlib.sha256()
        best_token, best_value = 0, float("-inf")
        for token_start in range(0, 129280, 256):
            weight = tensors.get_slice("head.weight")[token_start : token_start + 256].cuda()
            logits = torch.nn.functional.linear(normalized[:, -1].float(), weight.float())
            data = raw_bytes(logits)
            sha.update(data)
            for byte in data:
                fnv = ((fnv ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            value, offset = logits[0].max(0)
            if float(value) > best_value:
                best_value = float(value)
                best_token = token_start + int(offset)
            del weight, logits
    return {
        "event": "final",
        "collapsed_bf16_digest": tensor_digest(collapsed),
        "normalized_bf16_digest": tensor_digest(normalized),
        "logits_f32_digest": f"{fnv:016x}",
        "logits_f32_sha256": sha.hexdigest(),
        "argmax": best_token,
        "status": "PASS",
    }


def run_oracle(args):
    model_dir, config, storage = validate_storage(args.model_dir)
    resources = validate_resources()
    official = import_official(model_dir)
    official.world_size = 1
    official.rank = 0
    official.default_dtype = torch.float8_e4m3fn
    official.scale_fmt = "ue8m0"
    official.scale_dtype = torch.float8_e8m0fnu
    model_args = official_args(official, config)
    official.MoE = make_streaming_moe(official)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sparse_crosscheck = validate_sparse_fallback(official.sparse_attn)
    attention_topk = []

    def audited_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        attention_topk.append(topk_idxs.detach().cpu().contiguous())
        return reference_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    official.sparse_attn = audited_sparse_attn
    fallback_sha256 = hashlib.sha256(
        inspect.getsource(reference_sparse_attn).encode()
    ).hexdigest()

    output = (
        args.out
        or model_dir
        / "oracles"
        / f"v4_full61_token_{args.token_id}_pos_{args.start_pos}.official.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(model_dir / "model-00001-of-00064.safetensors", framework="pt", device="cpu") as tensors:
        embedding = tensors.get_slice("embed.weight")[args.token_id : args.token_id + 1].cuda()
    hidden = embedding[:, None, None, :].repeat(1, 1, 4, 1)
    with output.open("x") as trace:
        header = {
            "event": "run",
            "schema": SCHEMA,
            "implementation": "official-python-streaming",
            "model_revision": REVISION,
            "token_id": args.token_id,
            "start_pos": args.start_pos,
            "layers": 61,
            "executed_layers": args.max_layers,
            "model_dir": str(model_dir),
            "storage": storage,
            "resource_boundary": {
                "ownership": "one dense layer + one routed expert + 256 output rows",
                **resources,
            },
            "source_sha256": PINNED_SHA256,
            "sparse_attention": {
                "official_kernel_limitation": ADA_SPARSE_LIMITATION,
                "fallback": "independent scalar PyTorch pinned-formula reference",
                "fallback_sha256": fallback_sha256,
                "crosscheck_against_official_small_shape": sparse_crosscheck,
            },
        }
        trace.write(json.dumps(header, sort_keys=True) + "\n")
        trace.flush()
        for layer_id, ratio in enumerate(config["compress_ratios"][: args.max_layers]):
            attention_topk.clear()
            started = time.perf_counter_ns()
            shard = model_dir / f"model-{layer_id + 2:05}-of-00064.safetensors"
            block = official.Block(layer_id, model_args)
            block.ffn.shard = shard
            source_bytes = copy_parameters(block, shard, f"layers.{layer_id}")
            with torch.inference_mode():
                hidden = block(
                    hidden,
                    args.start_pos,
                    torch.tensor([[args.token_id]], device="cuda"),
                )
            torch.cuda.synchronize()
            if len(attention_topk) != 1:
                raise RuntimeError(
                    f"official layer {layer_id} emitted {len(attention_topk)} attention top-k tensors"
                )
            source_bytes += block.ffn.source_range_bytes
            record = {
                "event": "layer",
                "layer": layer_id,
                "compression_ratio": ratio,
                "attention_topk_digest": tensor_digest(attention_topk[0]),
                "hidden_bf16_digest": tensor_digest(hidden),
                "route_ids": block.ffn.last_ids.reshape(-1).tolist(),
                "route_weights_f32": block.ffn.last_weights.float().reshape(-1).tolist(),
                "source_range_bytes": source_bytes,
                "layer_elapsed_ns": time.perf_counter_ns() - started,
                "kv_rows_before": 0,
                "kv_rows_after": 1,
                "indexer_topk_cpu_ns": 0,
                **state_fields(block, ratio, args.start_pos + 1),
            }
            trace.write(json.dumps(record, sort_keys=True) + "\n")
            trace.flush()
            del block
            torch.cuda.empty_cache()
        if args.max_layers == 61:
            trace.write(json.dumps(final_record(model_dir, hidden), sort_keys=True) + "\n")
    print(output)


def validate(args):
    model_dir, config, storage = validate_storage(args.model_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_dir": str(model_dir),
                "model_revision": REVISION,
                "layers": config["num_hidden_layers"],
                "regular_shards": 64,
                "storage": storage,
            },
            sort_keys=True,
        )
    )


def load_trace(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 63 or records[0].get("event") != "run":
        raise RuntimeError(f"{path}: expected run + 61 layers + final")
    layers = records[1:62]
    if [record.get("layer") for record in layers] != list(range(61)):
        raise RuntimeError(f"{path}: layer order mismatch")
    for record in layers:
        missing = REQUIRED_LAYER_FIELDS - record.keys()
        if missing:
            raise RuntimeError(f"{path}: layer {record['layer']} missing {sorted(missing)}")
    missing = REQUIRED_FINAL_FIELDS - records[-1].keys()
    if missing:
        raise RuntimeError(f"{path}: final missing {sorted(missing)}")
    return records


def compare(left_path, right_path):
    left, right = load_trace(left_path), load_trace(right_path)
    for field in ("schema", "model_revision", "token_id", "start_pos", "layers"):
        if left[0].get(field) != right[0].get(field):
            raise RuntimeError(f"run field mismatch: {field}")
    for layer in range(61):
        for field in REQUIRED_LAYER_FIELDS:
            if left[layer + 1].get(field) != right[layer + 1].get(field):
                raise RuntimeError(f"layer {layer} mismatch: {field}")
    for field in REQUIRED_FINAL_FIELDS:
        if left[-1].get(field) != right[-1].get(field):
            raise RuntimeError(f"final mismatch: {field}")
    print("PASS: full61 official/Rust trace fields match exactly")


def load_prefix(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) < 2 or records[0].get("event") != "run":
        raise RuntimeError(f"{path}: expected run plus at least one layer")
    layers = [record for record in records[1:] if record.get("event") == "layer"]
    if not layers or [record.get("layer") for record in layers] != list(range(len(layers))):
        raise RuntimeError(f"{path}: prefix layer order mismatch")
    for record in layers:
        missing = REQUIRED_LAYER_FIELDS - record.keys()
        if missing:
            raise RuntimeError(f"{path}: layer {record['layer']} missing {sorted(missing)}")
    return records[0], layers


def compare_prefix(official_path, rust_path):
    official_header, official_layers = load_prefix(official_path)
    rust_header, rust_layers = load_prefix(rust_path)
    for field in ("schema", "model_revision", "token_id", "start_pos"):
        if official_header.get(field) != rust_header.get(field):
            raise RuntimeError(f"run field mismatch: {field}")
    if len(rust_layers) < len(official_layers):
        raise RuntimeError("Rust trace is shorter than official prefix")
    for layer, official_record in enumerate(official_layers):
        for field in REQUIRED_LAYER_FIELDS:
            if official_record.get(field) != rust_layers[layer].get(field):
                raise RuntimeError(f"layer {layer} mismatch: {field}")
    print(f"PASS: {len(official_layers)} official/Rust prefix layers match exactly")


def self_test():
    if fnv1a(b"abc") != "e71fa2190541574b" or empty_digest() != "cbf29ce484222325":
        raise RuntimeError("digest self-test failed")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "trace.jsonl"
        with output.open("x"):
            pass
        try:
            output.open("x")
        except FileExistsError:
            pass
        else:
            raise RuntimeError("oracle output overwrite guard failed")
    print("PASS: v4_full61_oracle schema/digest self-test")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--token-id", type=int, required=True, choices=range(129280))
    run.add_argument("--start-pos", type=int, default=0, choices=(0,))
    run.add_argument("--max-layers", type=int, default=61, choices=range(1, 62))
    run.add_argument("--out", type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--model-dir", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("official", type=Path)
    compare_parser.add_argument("rust", type=Path)
    prefix_parser = subparsers.add_parser("compare-prefix")
    prefix_parser.add_argument("official", type=Path)
    prefix_parser.add_argument("rust", type=Path)
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "run":
        run_oracle(args)
    elif args.command == "validate":
        validate(args)
    elif args.command == "compare":
        compare(args.official, args.rust)
    elif args.command == "compare-prefix":
        compare_prefix(args.official, args.rust)
    else:
        self_test()


if __name__ == "__main__":
    main()
