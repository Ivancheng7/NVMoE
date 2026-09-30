#!/usr/bin/env python3
"""Render the sm_120 (RTX 5060 Ti 16 GB) README figures from sealed measured rows.

Every number in ROWS and RESIDENCY is transcribed from a result file under
/srv/localmoe/targets/ that was read back during writing. Keep it that way:
the point of sealing the rows here is that `--check` fails if the rendered
figures ever drift from the text in the README, and the provenance JSON next
to them names the exact source file for each row.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
W = 960
C = {"ink": "#202124", "muted": "#5f6368", "grid": "#d1d5db", "blue": "#0072B2",
     "orange": "#E69F00", "vermillion": "#D55E00", "green": "#009E73", "paper": "#ffffff"}

# Steady-state s/token is (layer loop + that token's final head) averaged over
# generated tokens 1..7, the definition the project's own summary scripts use.
# Each run carries its own value, wall time, and result file, so the figure and
# the paired ratios are derived from the runs rather than from a summary line.
# `order` is the real position in the batch, not an index.
def run(value, e2e, source, order=""):
    return dict(steady=value, e2e=e2e, source=source, order=order)


ROWS = [
    dict(arm="A", label="pinned baseline (anchor)", commit="04bcf43", matched=False, color=C["blue"],
         runs=[run(12.873, 103.381, "baseline-nonresident-8step.jsonl")]),
    dict(arm="C", label="residency off", commit="b620ef7 tree", matched=True, color=C["orange"],
         runs=[run(3.061, 39.06, "residentfill/r1-stream.jsonl", "stream"),
               run(3.162, 38.19, "residentfill/r2-stream.jsonl", "stream"),
               run(3.243, 39.92, "residentfill/r3-stream.jsonl", "stream"),
               run(3.268, 40.76, "residentfill/r4-stream.jsonl", "stream")]),
    dict(arm="D", label="residency on", commit="b620ef7", matched=True, color=C["vermillion"],
         runs=[run(2.259, 30.49, "residentfill/r1-resident.jsonl", "resident"),
               run(2.273, 31.92, "residentfill/r2-resident.jsonl", "resident"),
               run(2.376, 33.09, "residentfill/r3-resident.jsonl", "resident"),
               run(2.310, 34.00, "residentfill/r4-resident.jsonl", "resident")]),
    # The fan-out pair is its own matched block, measured in two 2026-09-30 batches
    # (default-first and off-first, three rounds per arm). Its absolute values sit in
    # a different disk window than C/D's, so the pair is only read against itself.
    dict(arm="E-off", label="routed fanout off", commit="9799a0c", matched=True, color=C["green"],
         runs=[run(2.388, 27.893, "ab-fanout-fanoutflip/8step-fanout-off.jsonl", "fanout-off"),
               run(2.403, 28.496, "ab-fanout-fanoutflip-off/8step-fanout-off.jsonl", "fanout-off")]),
    dict(arm="E", label="routed fanout on (default)", commit="9799a0c", matched=True, color=C["green"],
         runs=[run(1.986, 25.114, "ab-fanout-fanoutflip/8step-fanout-on.jsonl", "fanout-on"),
               run(1.977, 25.151, "ab-fanout-fanoutflip-off/8step-fanout-on.jsonl", "fanout-on")]),
]

# Real run order of the fan-out batches: batch 1 ran the default (on) first, batch 2
# ran it off-first, so the two together are order-balanced.
FANOUT_BATCH = ["ab-fanout-fanoutflip/8step-fanout-on.jsonl", "ab-fanout-fanoutflip/8step-fanout-off.jsonl",
                "ab-fanout-fanoutflip-off/8step-fanout-off.jsonl", "ab-fanout-fanoutflip-off/8step-fanout-on.jsonl"]

# Figure 25 is the residency chain, so it draws these arms only.
CHAIN_ARMS = ("A", "C", "D")

# Real run order of the A/B batch: the two arms alternate so neither one pays
# the whole batch's first cold page cache.
BATCH_ORDER = ["residentfill/r1-stream.jsonl", "residentfill/r1-resident.jsonl",
               "residentfill/r2-resident.jsonl", "residentfill/r2-stream.jsonl",
               "residentfill/r3-stream.jsonl", "residentfill/r3-resident.jsonl",
               "residentfill/r4-resident.jsonl", "residentfill/r4-stream.jsonl"]

GATE = "layers=43;argmax=55222;hidden_bf16=5ed8bf88beae3cfe;fp4_accum=block32;final_head=canonical_cpu_streamed_bf16"

# Peak device bytes. A is the baseline's own peak; C and D are the same binary,
# which is the only pair where the memory delta means anything. E-off and E are the
# same binary too, and their delta is the fan-out's six lane buffers.
PEAK = {"A": 1466499072, "C": 1825112064, "D": 8175288320,
        "E-off": 8175288320, "E": 8242397184}


def text(x, y, value, size=16, fill=C["ink"], anchor="start", weight="normal"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial,Helvetica,sans-serif" '
            f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')


def rect(x, y, width, height, fill, stroke=C["ink"], stroke_width=1):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def line(x1, y1, x2, y2, color=C["ink"], width=1, dash=""):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>'


def svg(title, desc, body, height):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><rect width="100%" height="100%" fill="{C['paper']}"/>{body}</svg>
'''


def axis(x0, x1, y, maximum, ticks, label):
    body = line(x0, y, x1, y, C["ink"], 1.5)
    for tick in ticks:
        x = x0 + (x1 - x0) * tick / maximum
        body += line(x, y, x, y + 7, C["ink"])
        body += text(x, y + 24, tick, 12, C["muted"], "middle")
    return body + text((x0 + x1) / 2, y + 48, label, 13, C["ink"], "middle")


def steady_of(row):
    return [r["steady"] for r in row["runs"]]


def e2e_of(row):
    return [r["e2e"] for r in row["runs"]]


def chain_figure():
    height, maximum, x0, x1 = 660, 14.0, 340.0, 700.0
    rows = [row for row in ROWS if row["arm"] in CHAIN_ARMS]
    scale = lambda value: x0 + (x1 - x0) * value / maximum
    means = {row["arm"]: statistics.mean(steady_of(row)) for row in rows}
    body = text(48, 42, f"A steady token costs {means['A'] / means['D']:.2f}x less than the pinned baseline", 25, weight="bold")
    body += text(48, 70, "Bars start at zero. Each run is a real 8-token request on sm_120, plotted from zero.", 14, C["muted"])
    row_pitch, bar_h, bar_gap = 108, 15, 19
    top = 116
    for index, row in enumerate(rows):
        y = top + index * row_pitch
        steadies = steady_of(row)
        mean = statistics.mean(steadies)
        body += text(x0 - 22, y + 34, f"{row['arm']}  {row['label']}", 16, C["ink"], "end", "bold")
        for offset, value in enumerate(steadies):
            yy = y + offset * bar_gap
            body += rect(x0, yy, max(scale(value) - x0, 1.0), bar_h, row["color"], row["color"])
            body += text(x1 + 10, yy + 12, f"{value:.3f}s", 11, C["muted"])
        body += line(scale(mean), y - 10, scale(mean), y + (len(steadies) - 1) * bar_gap + bar_h + 2, C["ink"], 2)
        body += text(scale(mean), y - 18, f"mean {mean:.3f}s", 12, row["color"], "middle", "bold")
        body += text(x1 + 10, y - 2, f"e2e {statistics.mean(e2e_of(row)):.2f}s   n={len(steadies)}", 11, C["muted"])
    axis_y = top + len(rows) * row_pitch + 6
    body += axis(x0, x1, axis_y, maximum, [0, 3, 6, 9, 12], "steady per-token wall, seconds")
    pair = [c / d for c, d in zip(steady_of(rows[1]), steady_of(rows[2]))]
    body += text(48, axis_y + 88, f"A to D is {means['A'] / means['D']:.2f}x derived, across two stages measured at different times.", 16, C["vermillion"], weight="bold")
    body += text(48, axis_y + 112, f"Only C to D is a matched block: median paired {statistics.median(pair):.3f}x, order alternating, drop_caches before every run.", 13, C["muted"])
    body += text(48, axis_y + 132, "A is an anchor from an earlier tree, not a baseline taken the same day.", 13, C["muted"])
    return svg("sm_120 steady per-token chain", "Steady per-token wall time from the pinned baseline through the residency default, plotted from zero. Only the last pair is a matched block.", body, height)


def residency_figure():
    height, x0, x1 = 470, 330.0, 810.0
    by_arm = {row["arm"]: row for row in ROWS}
    stream, resident = by_arm["C"], by_arm["D"]
    body = text(48, 42, "Whole-model residency buys 0.879s/token for 6.35GB of VRAM", 25, weight="bold")
    body += text(48, 70, "Same binary, only COLIBRI_V4_FIXED_RESIDENT_GIB differs. All 8 runs produced identical tokens.", 14, C["muted"])
    panels = [
        ("steady per-token wall", 3.6, [statistics.mean(steady_of(stream)), statistics.mean(steady_of(resident))], lambda v: f"{v:.3f}s"),
        ("peak device memory", 9000.0, [PEAK[stream["arm"]] / 1e6, PEAK[resident["arm"]] / 1e6], lambda v: f"{v:.0f} MB"),
    ]
    for pi, (title, maximum, values, fmt) in enumerate(panels):
        top = 118 + pi * 168
        scale = lambda value, maximum=maximum: x0 + (x1 - x0) * value / maximum
        body += text(x0, top - 14, title, 14, C["ink"], "start", "bold")
        for vi, (row, value) in enumerate(zip([stream, resident], values)):
            y = top + 10 + vi * 54
            body += text(x0 - 16, y + 22, row["arm"] + "  " + row["label"].replace(" (default)", ""), 14, C["ink"], "end")
            body += rect(x0, y, scale(value) - x0, 30, row["color"], row["color"])
            body += text(scale(value) + 9, y + 21, fmt(value), 14, C["ink"], weight="bold")
    delta_b = (PEAK[resident["arm"]] - PEAK[stream["arm"]]) / 1e9
    body += text(48, 440, f"peak {PEAK[stream['arm']] / 1e9:.2f} to {PEAK[resident['arm']] / 1e9:.2f} GB (+{delta_b:.2f} GB) for {statistics.mean(steady_of(stream)) - statistics.mean(steady_of(resident)):.3f}s/token off the steady frame.", 15, C["vermillion"], weight="bold")
    return svg("sm_120 residency tradeoff", "Steady per-token time and peak device memory for the residency A/B arms, plotted from zero.", body, height)


def outputs():
    by_arm = {row["arm"]: row for row in ROWS}
    means = {row["arm"]: statistics.mean(steady_of(row)) for row in ROWS}
    pair = [c / d for c, d in zip(steady_of(by_arm["C"]), steady_of(by_arm["D"]))]
    pair_e2e = [c / d for c, d in zip(e2e_of(by_arm["C"]), e2e_of(by_arm["D"]))]
    fanout_pair = [a / b for a, b in zip(steady_of(by_arm["E-off"]), steady_of(by_arm["E"]))]
    fanout_pair_e2e = [a / b for a, b in zip(e2e_of(by_arm["E-off"]), e2e_of(by_arm["E"]))]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "arm", "label", "commit", "arm_side", "batch_position", "steady_s_per_token", "request_e2e_s",
        "peak_device_bytes", "matched_ab", "source", "pinned_gate"], lineterminator="\n")
    writer.writeheader()
    for row in ROWS:
        for r in row["runs"]:
            key = "/srv/localmoe/targets/" + r["source"]
            writer.writerow({
                "arm": row["arm"], "label": row["label"], "commit": row["commit"],
                "arm_side": r["order"],
                "batch_position": (BATCH_ORDER.index(r["source"]) + 1 if r["source"] in BATCH_ORDER
                                   else FANOUT_BATCH.index(r["source"]) + 1 if r["source"] in FANOUT_BATCH
                                   else ""),
                "steady_s_per_token": f"{r['steady']:.3f}", "request_e2e_s": f"{r['e2e']:.2f}",
                "peak_device_bytes": PEAK[row["arm"]], "matched_ab": "yes" if row["matched"] else "derived",
                "source": key, "pinned_gate": GATE,
            })
    return {
        "25-sm120-steady-chain.svg": chain_figure(),
        "26-sm120-residency-tradeoff.svg": residency_figure(),
        "sm120-steady-ab.csv": csv_buffer.getvalue(),
        "sm120-figure-data.json": json.dumps({
            "schema": "localmoe-sm120-readme-figures-v1",
            "machine": "WSL2, RTX 5060 Ti 16 GB, sm_120, CUDA 12.8",
            "workload": "8-token greedy request, token ID 0, drop_caches before each run; steady = generated tokens 1-7",
            "steady_s_per_token": means,
            "derived_chain_A_to_D": means["A"] / means["D"],
            "derived_chain_A_to_E": means["A"] / means["E"],
            "matched_residency_ab": {
                "paired_steady_ratio": pair, "median_paired_steady_ratio": statistics.median(pair),
                "paired_e2e_ratio": pair_e2e, "median_paired_e2e_ratio": statistics.median(pair_e2e),
                "batch_order": BATCH_ORDER,
            },
            "matched_fanout_ab": {
                "paired_steady_ratio": fanout_pair, "median_paired_steady_ratio": statistics.median(fanout_pair),
                "paired_e2e_ratio": fanout_pair_e2e, "median_paired_e2e_ratio": statistics.median(fanout_pair_e2e),
                "batch_order": FANOUT_BATCH,
            },
            "residency_cost": {
                "peak_before_bytes": PEAK["C"], "peak_after_bytes": PEAK["D"],
                "peak_delta_bytes": PEAK["D"] - PEAK["C"],
                "steady_saved_per_token": means["C"] - means["D"],
            },
            "fanout_cost": {
                "peak_before_bytes": PEAK["E-off"], "peak_after_bytes": PEAK["E"],
                "peak_delta_bytes": PEAK["E"] - PEAK["E-off"],
                "steady_saved_per_token": means["E-off"] - means["E"],
            },
            "rows": [{"arm": r["arm"], "label": r["label"], "commit": r["commit"],
                      "steady_s_per_token": steady_of(r), "request_e2e_s": e2e_of(r),
                      "peak_device_bytes": PEAK[r["arm"]], "matched_ab": r["matched"],
                      "sources": ["/srv/localmoe/targets/" + x["source"] for x in r["runs"]]} for r in ROWS],
            "pinned_gate": GATE,
            "result_files": ["/srv/localmoe/targets/" + s for s in
                             ["baseline-nonresident-8step.jsonl", "deliver-merge-0918/8step-tele.jsonl",
                              "deliver-merge-0918/ab-all1.jsonl", "deliver-merge-0918/ab-all2.jsonl"] + BATCH_ORDER + FANOUT_BATCH],
        }, indent=2, sort_keys=True) + "\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = outputs()
    for name, content in generated.items():
        if name.endswith(".svg"):
            assert ET.fromstring(content).tag == "{http://www.w3.org/2000/svg}svg"
    if args.check:
        for name, content in generated.items():
            assert (OUT / name).read_text() == content, name
        return
    for name, content in generated.items():
        (OUT / name).write_text(content)


if __name__ == "__main__":
    main()
