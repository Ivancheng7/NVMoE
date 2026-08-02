#!/usr/bin/env python3
"""Fail-fast CPU-only comparison of the Flash 0731 full43 oracle and Rust run."""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


def _load_oracle_helper():
    path = Path(__file__).with_name("v4_flash_0731_prefix_oracle.py")
    spec = importlib.util.spec_from_file_location("v4_flash_0731_prefix_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load oracle helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORACLE = _load_oracle_helper()
RUST_SCHEMA = "colibri-deepseek-v4-flash-0731-base-v1"
ORDERED_SCHEMA = "colibri-v4-b2-ordered-layer-digest-v1"
EXPECTED_LAYERS = 43
EXPECTED_ARGMAX = 33832
HEX16 = re.compile(r"[0-9a-f]{16}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class ComparisonError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ComparisonError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ComparisonError(f"{path}:{line_number}: invalid JSON: {error}") from error
    require(records, f"{path}: empty JSONL")
    return records


def one(records, event):
    matches = [record for record in records if record.get("event") == event]
    require(len(matches) == 1, f"official: expected one {event} event, got {len(matches)}")
    return matches[0]


def fnv(record, label, dtype=None):
    require(isinstance(record, dict), f"{label}: expected tensor digest record")
    if dtype is not None:
        require(record.get("dtype") == dtype, f"{label}: dtype mismatch")
    value = record.get("fnv1a64")
    require(isinstance(value, str) and HEX16.fullmatch(value), f"{label}: invalid fnv1a64")
    return value


def hex_digest(value, label):
    require(isinstance(value, str) and HEX16.fullmatch(value), f"{label}: invalid digest")
    return value


def canonical_layer_from_official(layer):
    indexer = layer.get("indexer")
    return {
        "layer": layer["layer"],
        "route_ids": layer["route_ids"],
        "route_weight_bits": layer["route_weight_bits"],
        "attention_topk_digest": fnv(layer["attention_topk_i32"], f"layer {layer['layer']} attention_topk", "INT32"),
        "hidden_digest": fnv(layer["hidden_bf16"], f"layer {layer['layer']} hidden", "BF16"),
        "kv_digest": fnv(layer["cache"]["window_active_bf16"], f"layer {layer['layer']} kv", "BF16"),
        "main_kv_state_digest": fnv(layer["main_compressor"]["kv_state_f32"], f"layer {layer['layer']} main kv", "F32"),
        "main_score_state_digest": fnv(layer["main_compressor"]["score_state_f32"], f"layer {layer['layer']} main score", "F32"),
        "indexer_kv_state_digest": None if indexer is None else fnv(indexer["kv_state_f32"], f"layer {layer['layer']} indexer kv", "F32"),
        "indexer_score_state_digest": None if indexer is None else fnv(indexer["score_state_f32"], f"layer {layer['layer']} indexer score", "F32"),
    }


def ordered_digest(layers):
    payload = {"schema": ORDERED_SCHEMA, "layers": layers}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def compare(official_path, rust_path, release_binary, official_sha256, release_sha256):
    require(HEX64.fullmatch(official_sha256 or ""), "expected official SHA256 must be 64 lowercase hex digits")
    require(HEX64.fullmatch(release_sha256 or ""), "expected release SHA256 must be 64 lowercase hex digits")
    actual_official_sha256 = file_sha256(official_path)
    require(actual_official_sha256 == official_sha256, "official: file SHA256 mismatch")
    actual_release_sha256 = file_sha256(release_binary)
    require(actual_release_sha256 == release_sha256, "release: binary SHA256 mismatch")

    official_records = load_jsonl(official_path)
    run = one(official_records, "run")
    final_head = one(official_records, "final_head")
    final = one(official_records, "final")
    official_layers = [record for record in official_records if record.get("event") == "layer"]
    releases = [record for record in official_records if record.get("event") == "release"]
    plan = list(ORACLE.execution_plan(42))

    require(run.get("schema") == ORACLE.FULL43_SCHEMA, "official: schema mismatch")
    require(run.get("model_revision") == ORACLE.REVISION, "official: model revision mismatch")
    require(run.get("implementation") == "official-python-streaming", "official: implementation mismatch")
    require(run.get("token_id") == 0 and run.get("start_pos") == 0, "official: expected token 0 at position 0")
    require(run.get("executed_layers") == EXPECTED_LAYERS, "official: executed layer count mismatch")
    require(run.get("full43_oracle") is True and run.get("final_head") is True, "official: incomplete full43/head scope")
    require(run.get("layers") == plan, "official: execution plan mismatch")
    require(len(official_layers) == EXPECTED_LAYERS, "official: expected 43 layer events")
    require([layer.get("layer") for layer in official_layers] == list(range(EXPECTED_LAYERS)), "official: layer order mismatch")
    require(len(releases) == EXPECTED_LAYERS, "official: expected 43 release events")
    require([release.get("layer") for release in releases] == list(range(EXPECTED_LAYERS)), "official: release order mismatch")
    require(final.get("status") == "PASS", "official: final status is not PASS")
    require(final.get("executed_layers") == EXPECTED_LAYERS, "official: final layer count mismatch")
    require(final.get("full43_oracle") is True and final.get("final_head") is True, "official: final scope mismatch")
    require(final_head.get("argmax_token_id") == EXPECTED_ARGMAX, "official: final-head argmax mismatch")
    require(final.get("final_head_argmax_token_id") == EXPECTED_ARGMAX, "official: final argmax mismatch")

    rust_records = load_jsonl(rust_path)
    require(len(rust_records) == 1, f"rust: expected one result record, got {len(rust_records)}")
    rust = rust_records[0]
    require(rust.get("event") == "run", "rust: event mismatch")
    require(rust.get("schema") == RUST_SCHEMA, "rust: schema mismatch")
    require(rust.get("implementation") == "rust-cuda", "rust: implementation mismatch")
    require(rust.get("profile") == "deepseek-v4-flash-0731", "rust: profile mismatch")
    require(rust.get("model_revision") == ORACLE.REVISION, "rust: model revision mismatch")
    require(rust.get("layers") == EXPECTED_LAYERS, "rust: layer count mismatch")
    require(rust.get("greedy_steps") == 1, "rust: expected one greedy step")
    require(rust.get("execution") == "base_only_no_speculation", "rust: execution mismatch")
    require(rust.get("base_only_no_speculation") is True, "rust: speculation must be disabled")
    require(rust.get("dspark") == "disabled", "rust: dspark must be disabled")
    require(rust.get("final_head") == "canonical_cpu_streamed_bf16", "rust: final head mismatch")

    requests = rust.get("requests")
    require(isinstance(requests, list) and len(requests) == 1, "rust: expected one request")
    request = requests[0]
    require(request.get("prompt_ids") == [0], "rust: prompt IDs mismatch")
    require(request.get("stop_reason") == "max_new_tokens", "rust: stop reason mismatch")
    tokens = request.get("tokens")
    require(isinstance(tokens, list) and len(tokens) == 1, "rust: expected one token trace")
    token = tokens[0]
    require(token.get("token_position") == 0 and token.get("input_id") == 0, "rust: token identity mismatch")
    require(token.get("input_kind") == "prompt", "rust: token kind mismatch")
    rust_layers = token.get("layers")
    require(isinstance(rust_layers, list) and len(rust_layers) == EXPECTED_LAYERS, "rust: expected 43 layer digests")
    require([layer.get("layer") for layer in rust_layers] == list(range(EXPECTED_LAYERS)), "rust: layer order mismatch")

    official_canonical = []
    fields = (
        "route_ids", "route_weight_bits", "attention_topk_digest", "hidden_digest",
        "kv_digest", "main_kv_state_digest", "main_score_state_digest",
        "indexer_kv_state_digest", "indexer_score_state_digest",
    )
    for layer_id, (official_layer, rust_layer) in enumerate(zip(official_layers, rust_layers)):
        expected_plan = plan[layer_id]
        require(official_layer.get("compression_ratio") == expected_plan["compression_ratio"], f"layer {layer_id}: compression ratio mismatch")
        require(official_layer.get("router") == expected_plan["router"], f"layer {layer_id}: router family mismatch")
        expected = canonical_layer_from_official(official_layer)
        official_canonical.append(expected)
        for field in fields:
            if field.endswith("_digest") and rust_layer.get(field) is not None:
                hex_digest(rust_layer[field], f"layer {layer_id} {field}")
            require(rust_layer.get(field) == expected[field], f"layer {layer_id}: {field} mismatch")

    expected_ordered = ordered_digest(official_canonical)
    require(token.get("ordered_layer_digest_sha256") == expected_ordered, "rust: ordered layer digest mismatch")
    require(token.get("hidden_bf16_digest") == fnv(final["final_hidden_bf16"], "official final hidden", "BF16"), "rust: final hidden digest mismatch")

    finals = request.get("finals")
    require(isinstance(finals, list) and len(finals) == 1, "rust: expected one final-head result")
    rust_final = finals[0]
    require(rust_final.get("status") == "PASS", "rust: final status is not PASS")
    require(rust_final.get("token_position") == 0 and rust_final.get("input_id") == 0, "rust: final token identity mismatch")
    require(rust_final.get("input_kind") == "prompt", "rust: final token kind mismatch")
    require(rust_final.get("head") == "canonical_cpu_streamed_bf16", "rust: final head label mismatch")
    require(rust_final.get("decision") == "emit_and_stop_max_new_tokens", "rust: final decision mismatch")
    require(rust_final.get("collapsed_bf16_digest") == fnv(final_head["hc_collapsed_bf16"], "official collapsed hidden", "BF16"), "rust: collapsed hidden digest mismatch")
    require(rust_final.get("normalized_bf16_digest") == fnv(final_head["norm_hidden_bf16"], "official normalized hidden", "BF16"), "rust: normalized hidden digest mismatch")
    require(rust_final.get("logits_f32_digest") == fnv(final_head["logits_f32"], "official logits", "F32"), "rust: logits digest mismatch")
    require(rust_final.get("argmax") == EXPECTED_ARGMAX, "rust: argmax mismatch")
    require(request.get("generated_ids") == [EXPECTED_ARGMAX], "rust: generated IDs mismatch")

    return {
        "status": "PASS_EXACT",
        "model_revision": ORACLE.REVISION,
        "official_sha256": actual_official_sha256,
        "release_sha256": actual_release_sha256,
        "layers": EXPECTED_LAYERS,
        "ordered_layer_digest_sha256": expected_ordered,
        "argmax": EXPECTED_ARGMAX,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--rust-result", required=True, type=Path)
    parser.add_argument("--release-binary", required=True, type=Path)
    parser.add_argument(
        "--expected-oracle-sha256",
        required=True,
        help="SHA256 of the exact official JSONL supplied to --official",
    )
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = compare(
            args.official, args.rust_result, args.release_binary,
            args.expected_oracle_sha256, args.release_sha256,
        )
    except (ComparisonError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
