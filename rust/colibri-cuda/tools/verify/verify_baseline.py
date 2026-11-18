#!/usr/bin/env python3
"""Verify a Flash-0731 run against the reproducible golden baseline.

Golden values (token_id=0, greedy step 0):
  argmax 55222, hidden digest 5ed8bf88beae3cfe, 43 layers, block32, CPU final head.

Usage: python3 verify_baseline.py <result.jsonl>
"""
import json
import sys

if len(sys.argv) < 2:
    sys.exit("usage: python3 verify_baseline.py <result.jsonl>")

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    rec = json.loads(handle.readline())

req = rec["requests"][0]
tok = req["tokens"][0]

checks = [
    ("argmax", 55222, req["finals"][0]["argmax"]),
    ("generated_ids[0]", 55222, req["generated_ids"][0]),
    ("hidden digest", "5ed8bf88beae3cfe", tok["hidden_bf16_digest"]),
    ("executed layers", 43, rec["layers"]),
    ("fp4_accum", "block32", rec["fp4_accum"]),
    ("final_head", "canonical_cpu_streamed_bf16", rec["final_head"]),
    ("stop_reason", "max_new_tokens", req["stop_reason"]),
]

print(f"file: {path}")
print(f"model_revision: {rec['model_revision']}")
print()
print(f"{'check':<20} {'expected':<28} {'actual':<28} verdict")
print("-" * 88)
all_ok = True
for label, expected, actual in checks:
    good = actual == expected
    all_ok &= good
    print(f"{label:<20} {str(expected):<28} {str(actual):<28} {'OK' if good else 'MISMATCH'}")

print()
print("peak device bytes :", rec["peak_device_used_bytes"])
print("request e2e ns    :", rec["request_e2e_ns"])
print()
print("VERDICT:", "baseline reproduces" if all_ok else "BASELINE BROKEN")
sys.exit(0 if all_ok else 1)
