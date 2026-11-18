#!/usr/bin/env python3
"""Summarize per-token phase wall times from Flash-0731 result JSONL files.

Usage: python3 attention_share.py FILE [FILE...]
"""
import json
import sys

PHASES = ["fixed", "routed", "shared", "attention", "router"]


def last_record(path):
    record = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                pass
    return record


def main():
    for path in sys.argv[1:]:
        record = last_record(path)
        if record is None:
            print(f"{path}: unparsable")
            continue
        layers = (record.get("phase_profile") or {}).get("layers") or []
        chunks = [layers[i:i + 43] for i in range(0, len(layers), 43)]
        heads = (record.get("phase_profile") or {}).get("final_heads") or []
        head_read = sum(
            ((h.get("top_level") or {}).get("cpu_streamed_final_head") or {}).get(
                "streamed_weight_read_ns", 0) or 0 for h in heads)
        print(f"== {path.split('/')[-1]}   tokens={len(chunks)}")
        if heads:
            last_head = (heads[-1].get("top_level") or {}).get("cpu_streamed_final_head") or {}
            print(f"   head_read(last)={last_head.get('streamed_weight_read_ns', 0)/1e9:.3f}s "
                  f"heads_total={head_read/1e9:.2f}s")
        print(f"   {'token':>5} {'outer':>7} {'attention':>10} {'share':>7}   "
              + " ".join(f"{p[:6]:>7}" for p in PHASES))
        steady_att = []
        steady_share = []
        for index, chunk in enumerate(chunks):
            outer = sum((l.get("outer") or {}).get("wall_ns", 0) for l in chunk)
            cells = {
                p: sum(((l.get("top_level") or {}).get(p) or {}).get("wall_ns", 0) or 0 for l in chunk)
                for p in PHASES
            }
            att = cells["attention"]
            share = att / outer if outer else 0
            print(f"   {index:5d} {outer/1e9:7.2f} {att/1e9:10.3f} {share*100:6.1f}%   "
                  + " ".join(f"{cells[p]/1e9:7.2f}" for p in PHASES))
            if index >= max(0, len(chunks) - 4):
                steady_att.append(att)
                steady_share.append(share)
        if steady_att:
            n = len(steady_att)
            print(f"   last {n} tokens: attention avg={sum(steady_att)/n/1e9:.3f}s "
                  f"share avg={sum(steady_share)/n*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
