#!/usr/bin/env python3
"""Compare repeated runs of the pinned official Flash-0731 oracle.

The official TileLang FP4 expert GEMM is not reproducible on `sm_120`, so
identical input produces different tokens across runs. This script locates
where that shows up: it reads several official result JSONL files produced
from the same input and reports the first layer and the first field at which
they disagree.

Every field is compared, not only the ones stored as tensor digests. Routing
is stored as plain arrays (`route_ids`, `route_weight_bits`), and a comparison
that skipped them would silently miss a routing divergence. Timing, RSS and
allocator fields are excluded because they are expected to vary.

Usage:
    compare_official_run_repeatability.py RUN.jsonl [RUN.jsonl ...]

Exit status is 0 when every run agrees with the first one, 1 otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Per-run measurements. These legitimately differ between runs and are not
# part of the numerical result.
VOLATILE = {
    "cuda_allocator",
    "elapsed_ns",
    "event",
    "layer",
    "layer_elapsed_ns",
    "rss_during_head",
    "rss_during_layer",
    "rss_final",
    "source_range_bytes",
    "status",
}


def comparable(value, prefix=""):
    """Project a record onto the fields that should repeat run to run."""
    out = {}
    if isinstance(value, dict):
        if "fnv1a64" in value:
            shape = value.get("shape")
            return {prefix: (value["fnv1a64"], tuple(shape) if shape else ())}
        for key, child in value.items():
            if key in VOLATILE:
                continue
            out.update(comparable(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        out[prefix] = tuple(value)
    else:
        out[prefix] = value
    return out


def load(path):
    run = {"name": path.parent.name if path.name == "official.jsonl" else path.stem}
    run["embedding"] = None
    run["argmax"] = None
    run["layers"] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            event = record.get("event")
            if event == "run":
                run["embedding"] = comparable(record.get("initial_embedding_bf16", {}))
            elif event == "layer":
                run["layers"][record["layer"]] = comparable(record)
            elif event == "final_head":
                run["argmax"] = record.get("argmax_token_id")
    return run


def first_difference(left, right):
    """Return (layer, field) of the first disagreement, or None when equal."""
    for layer in sorted(set(left["layers"]) | set(right["layers"])):
        a = left["layers"].get(layer)
        b = right["layers"].get(layer)
        if a is None or b is None:
            return layer, "<layer missing>"
        for field in sorted(set(a) | set(b)):
            if a.get(field) != b.get(field):
                return layer, field
    return None


def differing_fields(left, right, layer):
    """Every field of one layer that disagrees, with both values."""
    a = left["layers"][layer]
    b = right["layers"][layer]
    out = []
    for field in sorted(set(a) | set(b)):
        if a.get(field) != b.get(field):
            out.append((field, a.get(field), b.get(field)))
    return out


def main(argv):
    args = [arg for arg in argv[1:] if not arg.startswith("--")]
    verbose = "--fields" in argv[1:]
    paths = [Path(arg) for arg in args]
    if len(paths) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    runs = [load(path) for path in paths]
    base = runs[0]

    print(f"{'run':<26} {'argmax':>7}  embedding_equal_to_first  fields_compared")
    for run in runs:
        same = "yes" if run["embedding"] == base["embedding"] else "NO"
        counted = sum(len(comparable_set) for comparable_set in run["layers"].values())
        print(f"{run['name']:<26} {str(run['argmax']):>7}  {same:<24} {counted}")

    print()
    disagreements = 0
    for run in runs[1:]:
        found = first_difference(base, run)
        if found is None:
            print(f"{base['name']} vs {run['name']}: identical")
            continue
        layer, field = found
        disagreements += 1
        print(f"{base['name']} vs {run['name']}: first difference at layer {layer}, {field}")
        if verbose:
            for name, left_value, right_value in differing_fields(base, run, layer):
                print(f"    {name}")
                print(f"        {base['name']}: {left_value}")
                print(f"        {run['name']}: {right_value}")

    if disagreements:
        print(f"\n{disagreements} of {len(runs) - 1} comparisons disagree.")
        return 1
    print(f"\nall {len(runs)} runs agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
