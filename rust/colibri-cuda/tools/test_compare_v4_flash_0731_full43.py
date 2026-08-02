#!/usr/bin/env python3

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("compare_v4_flash_0731_full43.py")
SPEC = importlib.util.spec_from_file_location("compare_v4_flash_0731_full43", SCRIPT)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def digest(value, dtype):
    return {"dtype": dtype, "shape": [], "fnv1a64": f"{value:016x}"}


def fixture():
    plan = list(COMPARE.ORACLE.execution_plan(42))
    official_layers = []
    rust_layers = []
    for layer_id, item in enumerate(plan):
        indexer = None if layer_id < 2 or layer_id % 2 else {
            "kv_state_f32": digest(80_000 + layer_id, "F32"),
            "score_state_f32": digest(90_000 + layer_id, "F32"),
        }
        official = {
            "event": "layer", "layer": layer_id,
            "compression_ratio": item["compression_ratio"], "router": item["router"],
            "route_ids": [layer_id, layer_id + 1],
            "route_weight_bits": [100_000 + layer_id, 200_000 + layer_id],
            "attention_topk_i32": digest(10_000 + layer_id, "INT32"),
            "hidden_bf16": digest(20_000 + layer_id, "BF16"),
            "cache": {"window_active_bf16": digest(30_000 + layer_id, "BF16")},
            "main_compressor": {
                "kv_state_f32": digest(40_000 + layer_id, "F32"),
                "score_state_f32": digest(50_000 + layer_id, "F32"),
            },
            "indexer": indexer,
        }
        official_layers.append(official)
        rust_layers.append(COMPARE.canonical_layer_from_official(official))

    final_hidden = digest(600_000, "BF16")
    collapsed = digest(700_000, "BF16")
    normalized = digest(800_000, "BF16")
    logits = digest(900_000, "F32")
    official_records = [{
        "event": "run", "schema": COMPARE.ORACLE.FULL43_SCHEMA,
        "model_revision": COMPARE.ORACLE.REVISION,
        "implementation": "official-python-streaming", "token_id": 0,
        "start_pos": 0, "executed_layers": 43, "full43_oracle": True,
        "final_head": True, "layers": plan,
    }]
    for layer in official_layers:
        official_records.extend((layer, {"event": "release", "layer": layer["layer"]}))
    official_records.extend(({
        "event": "final_head", "hc_collapsed_bf16": collapsed,
        "norm_hidden_bf16": normalized, "logits_f32": logits,
        "argmax_token_id": COMPARE.EXPECTED_ARGMAX,
    }, {
        "event": "final", "executed_layers": 43, "full43_oracle": True,
        "final_head": True, "final_head_argmax_token_id": COMPARE.EXPECTED_ARGMAX,
        "final_hidden_bf16": final_hidden, "status": "PASS",
    }))
    rust = {
        "event": "run", "schema": COMPARE.RUST_SCHEMA,
        "implementation": "rust-cuda", "profile": "deepseek-v4-flash-0731",
        "model_revision": COMPARE.ORACLE.REVISION, "layers": 43,
        "greedy_steps": 1, "execution": "base_only_no_speculation",
        "base_only_no_speculation": True, "dspark": "disabled",
        "final_head": "canonical_cpu_streamed_bf16",
        "requests": [{
            "prompt_ids": [0], "generated_ids": [COMPARE.EXPECTED_ARGMAX],
            "stop_reason": "max_new_tokens",
            "tokens": [{
                "token_position": 0, "input_id": 0, "input_kind": "prompt",
                "hidden_bf16_digest": final_hidden["fnv1a64"],
                "ordered_layer_digest_sha256": COMPARE.ordered_digest(rust_layers),
                "layers": rust_layers,
            }],
            "finals": [{
                "token_position": 0, "input_id": 0, "input_kind": "prompt",
                "head": "canonical_cpu_streamed_bf16",
                "collapsed_bf16_digest": collapsed["fnv1a64"],
                "normalized_bf16_digest": normalized["fnv1a64"],
                "logits_f32_digest": logits["fnv1a64"],
                "argmax": COMPARE.EXPECTED_ARGMAX,
                "decision": "emit_and_stop_max_new_tokens", "status": "PASS",
            }],
        }],
    }
    return official_records, rust


class Full43ComparatorTest(unittest.TestCase):
    def write_fixture(self, root, official, rust):
        official_path = root / "official.jsonl"
        rust_path = root / "rust.jsonl"
        release_path = root / "colibri-cuda"
        official_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in official))
        rust_path.write_text(json.dumps(rust, sort_keys=True) + "\n")
        release_path.write_bytes(b"fixture release")
        return official_path, rust_path, release_path

    def test_exact_fixture_passes(self):
        official, rust = fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory), official, rust)
            result = COMPARE.compare(
                *paths, hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                hashlib.sha256(paths[2].read_bytes()).hexdigest(),
            )
        self.assertEqual(result["status"], "PASS_EXACT")
        self.assertEqual(result["argmax"], 33832)

    def test_first_field_mismatch_fails_fast(self):
        official, rust = fixture()
        rust = copy.deepcopy(rust)
        rust["requests"][0]["tokens"][0]["layers"][17]["route_weight_bits"][0] += 1
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory), official, rust)
            with self.assertRaisesRegex(COMPARE.ComparisonError, "layer 17: route_weight_bits mismatch"):
                COMPARE.compare(
                    *paths, hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                    hashlib.sha256(paths[2].read_bytes()).hexdigest(),
                )

    def test_layer_difference_precedes_generated_id_difference(self):
        official, rust = fixture()
        rust = copy.deepcopy(rust)
        rust["requests"][0]["generated_ids"] = [0]
        rust["requests"][0]["tokens"][0]["layers"][4]["hidden_digest"] = "0000000000000000"
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory), official, rust)
            with self.assertRaisesRegex(COMPARE.ComparisonError, "layer 4: hidden_digest mismatch"):
                COMPARE.compare(
                    *paths, hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                    hashlib.sha256(paths[2].read_bytes()).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
