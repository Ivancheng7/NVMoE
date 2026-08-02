#!/usr/bin/env python3
"""Render auditable Flash-0731 README figures from the two sealed timing blocks."""
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
C = {"ink": "#202124", "muted": "#5f6368", "grid": "#d1d5db", "blue": "#0072B2", "orange": "#E69F00", "vermillion": "#D55E00", "paper": "#ffffff"}
GATE = "layers=43;argmax=33832;collapsed=5e9764a6e105a3d6;normalized=91495daa576afcb3;logits=e9d90c64a1441d20"

# Both blocks use fresh processes, token ID 0, one greedy step, the same pinned
# Flash-0731 revision, and the same exact gate. A2 re-baselines B before F.
ROWS = [
    ("A1", "stage 1 baseline", [93.239537780, 93.459421680], [8030632, 8030040], "680e8bf724d9aa041474f3166059755e38da51f711aa1d728224c61529c357dc", C["blue"]),
    ("B", "request session", [34.876387376, 34.345708776], [1412388, 1411612], "1cc7b299658bbefbbf7c9249c0dc8cf3a34f28793e3bbcb511ff30808c5c1f03", C["orange"]),
    ("A2", "release rebaseline", [31.487469340, 32.459039482], [1413208, 1413716], "d75ee989bca0fd52245c9c508fdc823af83570ae00c96f372839bf9a15929098", C["orange"]),
    ("F", "final release", [12.926165948, 13.326181660], [1412112, 1412524], "6a47f8fe48c87f4205f8f835a60f2f824940dea67c198463ccfb60ccf333bb77", C["vermillion"]),
]


def text(x, y, value, size=16, fill=C["ink"], anchor="start", weight="normal"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')


def rect(x, y, width, height, fill, stroke=C["ink"], stroke_width=1):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def line(x1, y1, x2, y2, color=C["ink"], width=1, dash=""):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>'


def circle(x, y, radius, fill=C["paper"], stroke=C["ink"], width=2):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


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


def latency_figure():
    height, maximum, x0, x1 = 690, 100.0, 310.0, 890.0
    scale = lambda value: x0 + (x1 - x0) * value / maximum
    a1, b, a2, d = [statistics.median(row[2]) for row in ROWS]
    chain = a1 / d
    body = text(48, 42, f"Two exact matched blocks form a {chain:.2f}x optimization chain", 25, weight="bold")
    body += text(48, 70, "Four samples per block. A2 re-baselines B before the final F path.", 14, C["muted"])
    for index, (arm, label, values, _, _, color) in enumerate(ROWS):
        y = 125 + index * 105
        median = statistics.median(values)
        body += text(x0 - 20, y + 31, f"{arm}  {label}", 16, C["ink"], "end", "bold")
        for offset, value in enumerate(values):
            yy = y + offset * 26
            body += rect(x0, yy, scale(value) - x0, 18, color, color)
            body += text(scale(value) + 9, yy + 14, f"{value:.3f}s", 12, C["ink"], weight="bold")
        body += line(scale(median), y - 11, scale(median), y + 63, C["ink"], 2)
        body += text(scale(median), y - 17, f"median {median:.3f}s", 12, color, "middle", "bold")
    first_pair = statistics.median([ROWS[0][2][0] / ROWS[1][2][0], ROWS[0][2][1] / ROWS[1][2][1]])
    second_pair = statistics.median([ROWS[2][2][0] / ROWS[3][2][0], ROWS[2][2][1] / ROWS[3][2][1]])
    body += axis(x0, x1, 555, maximum, [0, 20, 40, 60, 80, 100], "request wall time, seconds")
    body += text(48, 625, f"A1 to B: {first_pair:.3f}x. A2 to F: {second_pair:.3f}x. A1 to F: {chain:.3f}x derived across blocks.", 17, C["vermillion"], weight="bold")
    body += text(48, 655, "Every bar starts at zero. Each sample passed the same 43-layer exact gate.", 13, C["muted"])
    return svg("Flash-0731 exact optimization chain", "Eight individual exact request latencies in two matched blocks, plotted from zero.", body, height)


def rss_figure():
    height, maximum, x0, x1 = 475, 8.5, 310.0, 890.0
    scale = lambda value: x0 + (x1 - x0) * value / maximum
    first, final = ROWS[0], ROWS[-1]
    first_values = [value / 1048576 for value in first[3]]
    final_values = [value / 1048576 for value in final[3]]
    first_median, final_median = statistics.median(first_values), statistics.median(final_values)
    body = text(48, 42, "The final exact path uses 6.31 GiB less peak RSS", 25, weight="bold")
    body += text(48, 70, "First and final samples belong to separate matched blocks. The displayed difference is derived.", 14, C["muted"])
    for y, arm, label, values, color in [(145, "A1", "stage 1 baseline", first_values, first[-1]), (265, "F", "final release", final_values, final[-1])]:
        median = statistics.median(values)
        body += text(x0 - 20, y + 32, f"{arm}  {label}", 16, C["ink"], "end", "bold")
        body += rect(x0, y, scale(median) - x0, 44, color, color)
        body += text(scale(median) + 10, y + 29, f"{median:.2f} GiB", 16, C["ink"], weight="bold")
        body += text(scale(median), y + 65, f"samples {values[0]:.2f}, {values[1]:.2f} GiB", 12, C["muted"], "middle")
    body += axis(x0, x1, 370, maximum, [0, 2, 4, 6, 8], "maximum RSS, GiB")
    body += text(48, 440, f"median maxRSS {first_median:.2f} GiB to {final_median:.2f} GiB. Derived difference {first_median - final_median:.2f} GiB.", 16, C["vermillion"], weight="bold")
    return svg("Flash-0731 maximum RSS", "Maximum RSS samples for the first and final exact paths, plotted from zero.", body, height)


def outputs():
    medians = {row[0]: statistics.median(row[2]) for row in ROWS}
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=["block", "order", "arm", "request_e2e_s", "max_rss_kib", "binary_sha256", "exact_gate"], lineterminator="\n")
    writer.writeheader()
    for block, (left, right) in [("A-B-B-A", (ROWS[0], ROWS[1])), ("A-F-F-A", (ROWS[2], ROWS[3]))]:
        for order, row, sample in [(1, left, 0), (2, right, 0), (3, right, 1), (4, left, 1)]:
            writer.writerow({"block": block, "order": order, "arm": row[0], "request_e2e_s": f"{row[2][sample]:.9f}", "max_rss_kib": row[3][sample], "binary_sha256": row[4], "exact_gate": GATE})
    return {
        "23-flash-0731-exact-abba.svg": latency_figure(),
        "24-flash-0731-rss.svg": rss_figure(),
        "flash-0731-abba.csv": csv_buffer.getvalue(),
        "flash-0731-figure-data.json": json.dumps({
            "schema": "localmoe-flash-0731-readme-figures-v2",
            "workload": "one token, 43 layers, token ID 0, fresh processes; two separately matched blocks",
            "blocks": [{"order": "A-B-B-A", "rows": [ROWS[0][0], ROWS[1][0]], "paired_speedup": [ROWS[0][2][0] / ROWS[1][2][0], ROWS[0][2][1] / ROWS[1][2][1]], "median_paired_speedup": statistics.median([ROWS[0][2][0] / ROWS[1][2][0], ROWS[0][2][1] / ROWS[1][2][1]])}, {"order": "A-F-F-A", "rows": [ROWS[2][0], ROWS[3][0]], "paired_speedup": [ROWS[2][2][0] / ROWS[3][2][0], ROWS[2][2][1] / ROWS[3][2][1]], "median_paired_speedup": statistics.median([ROWS[2][2][0] / ROWS[3][2][0], ROWS[2][2][1] / ROWS[3][2][1]])}],
            "rows": [{"arm": row[0], "label": row[1], "request_e2e_s": row[2], "max_rss_kib": row[3], "binary_sha256": row[4]} for row in ROWS],
            "exact_gate": GATE,
            "derived_chain_speedup": medians["A1"] / medians["F"],
            "standalone_final_release": {"binary_sha256": ROWS[-1][4], "request_e2e_s": 14.298313983, "max_rss_kib": 1412424},
            "attention_profile_s": {"before": 19.862862, "after": 1.586645},
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
