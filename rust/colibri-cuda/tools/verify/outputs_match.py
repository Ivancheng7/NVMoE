#!/usr/bin/env python3
"""Compare the model outputs of a run against recorded references.

Checks, per request: per-token hidden digests, argmax values,
ordered_layer_digest_sha256 (chain over all 43 layers), and peak device bytes.

Usage: python3 outputs_match.py REFERENCE.jsonl FILE [FILE...]
"""
import json
import sys


def last_record(path):
    record = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    pass
    return record


def outputs(path):
    record = last_record(path)
    request = (record.get("requests") or [{}])[0]
    tokens = request.get("tokens") or []
    finals = request.get("finals") or []
    return {
        "hidden digests": [token.get("hidden_bf16_digest") for token in tokens],
        "layer chain": [token.get("ordered_layer_digest_sha256") for token in tokens],
        "argmax": [final.get("argmax") for final in finals],
        "peak device bytes": record.get("peak_device_used_bytes"),
        "layers": [len(token.get("layers") or []) for token in tokens],
    }


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python3 outputs_match.py REFERENCE.jsonl FILE [FILE...]")
    reference = outputs(sys.argv[1])
    print(f"reference {sys.argv[1].split('/')[-1]}")
    for key, value in reference.items():
        shown = value if not isinstance(value, list) or len(value) < 4 else value[:2] + ["..."]
        print(f"   {key:<18} {shown}")
    print()
    for path in sys.argv[2:]:
        candidate = outputs(path)
        same = {key: reference[key] == candidate[key] for key in reference}
        verdict = "IDENTICAL" if all(same.values()) else "DIFFERS: " + ",".join(
            key for key, ok in same.items() if not ok
        )
        print(f"   {path.split('/')[-1]:<44} {verdict}")


if __name__ == "__main__":
    main()
