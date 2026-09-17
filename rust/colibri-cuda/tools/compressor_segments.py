"""Compressor segments per run, grouped by (outputs, ratio) layout, next to the
bench-block compressor step that defines the run's tier.

Reads COMPPROBE_HOST lines (the per-row cuBLAS path the delivery frame runs),
falling back to COMPPROBE lines (the reusable arena path) when present.

Usage: python3 compressor_segments.py FILE [FILE...]
"""
import glob
import json
import re
import statistics
import sys

PAIR = re.compile(r"([a-z0-9_]+)=(-?\d+)")
HEAD = re.compile(r"\b(COMPPROBE_HOST|COMPPROBE)\b")


def segments(path):
    """COMPPROBE lines live on stderr; the jsonl next to it carries the tier.

    Parsed generically so a renamed or added segment does not silently drop
    the whole run.
    """
    stderr = path[:-len(".jsonl")] + ".stderr" if path.endswith(".jsonl") else path
    out = {}
    order = []
    kind = None
    with open(stderr, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            head = HEAD.search(line)
            if not head:
                continue
            pairs = dict(PAIR.findall(line[head.end():]))
            if "outputs" not in pairs or "ratio" not in pairs:
                continue
            kind = head.group(1)
            key = (int(pairs["outputs"]), int(pairs["ratio"]))
            out.setdefault(key, []).append(
                {name: int(value) for name, value in pairs.items()})
            for name in pairs:
                if name not in order:
                    order.append(name)
    return out, order, kind


def bench_compressor(path):
    """Mean sync wall of the bench `compressor` step on the non-Csa4 layers."""
    record = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    pass
    walls = []
    for layer in (record.get("phase_profile") or {}).get("layers") or []:
        bench = ((layer.get("bench") or {}).get("attention") or {}).get(
            "certified_host_path") or {}
        comp = (bench.get("compressor") or {}).get("sync_wall_ns", 0) or 0
        idx = (bench.get("indexer") or {}).get("sync_wall_ns", 0) or 0
        if comp and not idx:
            walls.append(comp / 1e6)
    return statistics.mean(walls) if walls else float("nan")


def main():
    paths = []
    for arg in sys.argv[1:]:
        paths.extend(sorted(glob.glob(arg)) or [arg])
    for path in paths:
        rows, order, kind = segments(path)
        name = path.split("/")[-1].replace(".jsonl", "")
        if not rows:
            print(f"{name}: no COMPPROBE lines")
            continue
        print(f"--- {name}  [{kind}]  bench compressor (plain layers) = "
              f"{bench_compressor(path):.2f} ms")
        print(f"    {'layout':>12} {'n':>4} " +
              " ".join(f"{seg:>15}" for seg in order))
        for key in sorted(rows):
            group = rows[key]
            mean = lambda seg: statistics.mean(row.get(seg, 0) for row in group)
            print(f"    {str(key):>12} {len(group):>4} " +
                  " ".join(f"{mean(seg):>15.1f}" for seg in order))


if __name__ == "__main__":
    main()
