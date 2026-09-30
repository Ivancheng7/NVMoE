"""Arm-wise summary for `serve_arm.sh`: id strings plus steady per char.

Reuses `serve_steady` from `serve_vs_cli_sum.py` so the steady definition stays
the one the serve-vs-cli batch already used (the engine's own `per_char`
figure); this script only adds the two things that batch did not need: grouping
by arm and comparing the generated id strings.

Usage: python3 serve_arm_sum.py DIR
"""
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_vs_cli_sum import serve_steady  # noqa: E402

IDS = re.compile(r"ids=(\S+)")
ARM_FILE = re.compile(r"^serve-(off|on)-r(\d+)\.log$")


def id_string(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip().startswith("stats"):
                continue
            match = IDS.search(line)
            if match:
                return match.group(1).rstrip(",")
    return None


def main():
    directory = sys.argv[1]
    rows = []
    for path in sorted(glob.glob(f"{directory}/serve-*.log")):
        name = os.path.basename(path)
        match = ARM_FILE.match(name)
        if not match:
            continue
        arm, round_no = match.group(1), int(match.group(2))
        parsed = serve_steady(path)
        if parsed is None:
            print(f"{name}: (no stats line)")
            continue
        tokens, first, steady = parsed
        rows.append((arm, round_no, tokens, first, steady, id_string(path)))

    print(f"{'arm':<5} {'round':>5} {'tokens':>7} {'first s':>8} {'steady s':>9}")
    for arm, round_no, tokens, first, steady, _ in rows:
        print(f"{arm:<5} {round_no:>5} {tokens:>7} {first:>8.2f} {steady:>9.3f}")

    ids = [row[5] for row in rows]
    unique = sorted({value for value in ids if value is not None})
    if len(unique) == 1:
        print(f"\nid strings: IDENTICAL across {len(rows)} runs")
        print(f"  ids={unique[0]}")
    else:
        print(f"\nid strings: {len(unique)} DISTINCT values across {len(rows)} runs")
        for value in unique:
            print(f"  ids={value}")

    print()
    for arm in ("off", "on"):
        values = [row[4] for row in rows if row[0] == arm]
        if values:
            print(f"{arm:<3} steady per char: {statistics.mean(values):.3f} s "
                  f"({min(values):.3f}-{max(values):.3f}, n={len(values)})")
    off = [row[4] for row in rows if row[0] == "off"]
    on = [row[4] for row in rows if row[0] == "on"]
    if off and on:
        print(f"on - off = {statistics.mean(on) - statistics.mean(off):+.3f} s/char")


if __name__ == "__main__":
    main()
