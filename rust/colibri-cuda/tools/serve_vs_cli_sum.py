"""Steady per-token cost on both delivery surfaces, same batch.

Serve side: the engine's `stats` line ends with `<first_token_ms> <per_char_ms>`,
and the generated ids are listed after `ids=`, so the token count comes from
there. Steady is the per-char figure as the engine reports it.

CLI side: `request_e2e_ns` covers the layer loop plus the final heads plus the
rest, and the first token's share is its layer-loop outer plus its final head,
so steady = (request - first) / (tokens - 1).

Usage: python3 serve_vs_cli_sum.py DIR
"""
import glob
import json
import re
import statistics
import sys

PER_TOKEN = 43
STATS = re.compile(r"^stats\s*:\s*(.*)$")
NUMS = re.compile(r"^-?\d+(?:\.\d+)?$")


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


def serve_steady(path):
    """(tokens, first_token_s, steady_s) from a client log, or None."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = STATS.match(line.strip())
            if not match:
                continue
            body = match.group(1)
            ids = re.search(r"ids=(\S+)", body)
            if not ids:
                continue
            tokens = len(ids.group(1).rstrip(",").split(","))
            numbers = [float(field) for field in body.split()
                       if NUMS.match(field)]
            if len(numbers) < 2 or tokens < 2:
                continue
            return tokens, numbers[-2] / 1000.0, numbers[-1] / 1000.0
    return None


def cli_steady(path):
    """(steady_s, first_token_s, per_phase_s) from a result jsonl, or None."""
    record = last_record(path)
    if record is None:
        return None
    profile = record.get("phase_profile") or {}
    request = (profile.get("outer") or {}).get("wall_ns", 0) / 1e9
    layers = profile.get("layers") or []
    chunks = [layers[i:i + PER_TOKEN] for i in range(0, len(layers), PER_TOKEN)]
    if len(chunks) < 2:
        return None
    head_walls = [(entry.get("outer") or {}).get("wall_ns", 0) / 1e9
                  for entry in (profile.get("final_heads") or [])]
    first = sum(layer["outer"]["wall_ns"] for layer in chunks[0]) / 1e9
    if head_walls:
        first += head_walls[0]
    steady = (request - first) / (len(chunks) - 1)
    body = [layer for index, layer in enumerate(layers) if index % PER_TOKEN != 0]
    phases = {}
    for name in ("fixed", "routed", "shared", "attention", "router"):
        phases[name] = statistics.mean(
            layer["top_level"][name]["wall_ns"] for layer in body) / 1e9
    phases["head"] = statistics.mean(head_walls[1:]) if len(head_walls) > 1 else 0.0
    return steady, first, phases


def main():
    directory = sys.argv[1]
    print(f"{'surface':<7} {'run':<4} {'steady':>8} {'first tok':>10} "
          f"{'request':>9}")
    serves, clis = [], []
    for path in sorted(glob.glob(f"{directory}/serve-*.log")):
        name = path.split("-")[-1].replace(".log", "")
        parsed = serve_steady(path)
        if parsed is None:
            print(f"{'serve':<7} {name:<4} (no stats line)")
            continue
        tokens, first, steady = parsed
        serves.append(steady)
        print(f"{'serve':<7} {name:<4} {steady:>8.3f} {first:>10.2f} "
              f"{first + steady * (tokens - 1):>9.2f}")
    for path in sorted(glob.glob(f"{directory}/cli-*.jsonl")):
        name = path.split("-")[-1].replace(".jsonl", "")
        parsed = cli_steady(path)
        if parsed is None:
            continue
        steady, first, phases = parsed
        clis.append((steady, phases))
        print(f"{'cli':<7} {name:<4} {steady:>8.3f} {first:>10.2f} "
              f"{first + steady * 7:>9.2f}")
    print()
    if serves:
        print(f"serve steady per char : {statistics.mean(serves):.3f} s "
              f"({min(serves):.3f}-{max(serves):.3f}, n={len(serves)})")
    if clis:
        values = [item[0] for item in clis]
        print(f"cli   steady per token: {statistics.mean(values):.3f} s "
              f"({min(values):.3f}-{max(values):.3f}, n={len(clis)})")
        names = ["fixed", "routed", "shared", "attention", "router", "head"]
        print("cli   per-token phases: " + "  ".join(
            f"{name}={statistics.mean(item[1][name] for item in clis):.2f}"
            for name in names))
    if serves and clis:
        delta = statistics.mean(item[0] for item in clis) - statistics.mean(serves)
        print(f"cli - serve = {delta:+.3f} s/token")


if __name__ == "__main__":
    main()
