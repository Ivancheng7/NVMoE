#!/usr/bin/env python3
"""Generate README cost/evidence SVGs from frozen Colibri artifacts (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/mnt/colibri-native/.logs")
FLASH_REPORT = REPO / "docs" / "FLASH_MIGRATION_RESEARCH.md"
W = 960

# Okabe-Ito plus neutral grays. Every categorical use also has a non-color cue.
C = {
    "ink": "#202124",
    "muted": "#6b7280",
    "line": "#d1d5db",
    "grid": "#e5e7eb",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "paper": "#ffffff",
    "gap": "#9ca3af",
}

ARTIFACTS = {
    "raw_gpu_pair1": (
        "qd4-raw-host-abba-4fedb133-20260725-raw-a.jsonl",
        "8a8690f2868e10ae0b7cb7bc0a3ad8c4055f1a7efe8b45fcaa1d13ca7178004f",
    ),
    "raw_gpu_pair2": (
        "qd4-raw-host-abba-4fedb133-20260725-raw-a-2.jsonl",
        "49dd68b2673cf0ab2126f5b0e81a16107b6179918837068053ef45294372a8b6",
    ),
    "host_bf16_pair1": (
        "qd4-raw-host-abba-4fedb133-20260725-host-b.jsonl",
        "0ee931dd095243d8cda15a32c3a12649c6cc1cb4fce0c9576b564ea27c852a4e",
    ),
    "host_bf16_pair2": (
        "qd4-raw-host-abba-4fedb133-20260725-host-b-2.jsonl",
        "6f3a60246121be268674eb671521e9c0fe37fdbae0ccfc7200bbdc37f475af65",
    ),
    "cache_off_pair1": (
        "full61-qd4-cache-abba-b43-a1.jsonl",
        "a6c7509e644366c162058822e6ba6a5e9cf3b5872156bc215e5deb749e938cf4",
    ),
    "cache_off_pair2": (
        "full61-qd4-cache-abba-b43-a2.jsonl",
        "61338061e833e85165f465b92b0ff8e00063b0a9ddffefe2187305fd67e0271c",
    ),
    "cache_on_pair1": (
        "full61-qd4-cache-abba-b43-b1.jsonl",
        "333c0ff7cd719198e5debbfd8c6b4d3b09f7ac7342b26b1b42d4eed6f549a078",
    ),
    "cache_on_pair2": (
        "full61-qd4-cache-abba-b43-b2.jsonl",
        "5af29cc0ee387f3eab7f4e383a023728f3f5f68f40e087f3e89ae4e92e7c16cb",
    ),
    "cache_strict_audit": (
        "full61-qd4-immutable-cache-strict-audit-b43-20260725.jsonl",
        "6d0972b01ee945bfb88c47f92acb5b306de2cd8e1eb3cd3c0696deeb90e920c1",
    ),
    "pos5_one_token": (
        "pos5-production-dag-0009-b15c-20260725.jsonl",
        "ae807d39e1e087a5f93da75f1d5ceca71da0b6f88ffce436fbda83cb41f0bf94",
    ),
}

RAW_KEYS = ("raw_gpu_pair1", "raw_gpu_pair2")
HOST_KEYS = ("host_bf16_pair1", "host_bf16_pair2")
CACHE_OFF_KEYS = ("cache_off_pair1", "cache_off_pair2")
CACHE_ON_KEYS = ("cache_on_pair1", "cache_on_pair2")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def text(
    x: float,
    y: float,
    value: object,
    size: int = 16,
    fill: str = C["ink"],
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" '
        'font-family="Arial,Helvetica,sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(str(value))}</text>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = C["ink"],
    width: float = 2,
    dash: str = "",
) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str = "none",
    stroke: str = C["line"],
    radius: float = 4,
    stroke_width: float = 1,
    dash: str = "",
) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{extra}/>'
    )


def circle(
    x: float,
    y: float,
    radius: float,
    fill: str = C["paper"],
    stroke: str = C["ink"],
    stroke_width: float = 2,
) -> str:
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def square(
    x: float,
    y: float,
    half: float,
    fill: str = C["paper"],
    stroke: str = C["ink"],
    stroke_width: float = 2,
) -> str:
    return rect(
        x - half,
        y - half,
        2 * half,
        2 * half,
        fill,
        stroke,
        0,
        stroke_width,
    )


def defs() -> str:
    return f"""
<defs>
  <pattern id="hatch-orange" width="8" height="8" patternUnits="userSpaceOnUse">
    <rect width="8" height="8" fill="{C['paper']}"/>
    <path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="{C['orange']}" stroke-width="2"/>
  </pattern>
  <pattern id="hatch-sky" width="8" height="8" patternUnits="userSpaceOnUse">
    <rect width="8" height="8" fill="{C['sky']}"/>
    <path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="{C['paper']}" stroke-width="1.5"/>
  </pattern>
  <pattern id="dots-green" width="8" height="8" patternUnits="userSpaceOnUse">
    <rect width="8" height="8" fill="{C['green']}"/>
    <circle cx="2" cy="2" r="1.2" fill="{C['paper']}"/>
    <circle cx="6" cy="6" r="1.2" fill="{C['paper']}"/>
  </pattern>
  <pattern id="cross-vermillion" width="8" height="8" patternUnits="userSpaceOnUse">
    <rect width="8" height="8" fill="{C['vermillion']}"/>
    <path d="M0,0 L8,8 M8,0 L0,8" stroke="{C['paper']}" stroke-width="1"/>
  </pattern>
  <pattern id="vertical-purple" width="8" height="8" patternUnits="userSpaceOnUse">
    <rect width="8" height="8" fill="{C['purple']}"/>
    <path d="M2,0 L2,8 M6,0 L6,8" stroke="{C['paper']}" stroke-width="1.2"/>
  </pattern>
</defs>"""


def svg(title_value: str, desc_value: str, body: str, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title_value)}</title>
<desc id="desc">{escape(desc_value)}</desc>
{defs()}
<rect width="100%" height="100%" fill="{C['paper']}"/>
{body}
</svg>
"""


def footer(source: str, height: int) -> str:
    return text(48, height - 14, source, 11, C["muted"])


def read_artifacts(root: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    records: dict[str, list[dict]] = {}
    metadata: dict[str, dict] = {}
    for key, (name, expected_sha) in ARTIFACTS.items():
        path = root / name
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        if digest != expected_sha:
            raise RuntimeError(f"artifact identity changed: {path}")
        if not raw.endswith(b"\n"):
            raise RuntimeError(f"artifact is not newline-terminated: {path}")
        parsed = [json.loads(item) for item in raw.decode("utf-8").splitlines()]
        if parsed[0].get("event") != "run":
            raise RuntimeError(f"missing run header: {path}")
        if parsed[-1].get("event") != "final" or parsed[-1].get("status") != "PASS":
            raise RuntimeError(f"missing PASS final: {path}")
        if parsed[0].get("schema") != "colibri-v4-layer-stationary-prefill-v1":
            raise RuntimeError(f"unexpected trace schema: {path}")
        records[key] = parsed
        metadata[key] = {
            "path": str(path),
            "sha256": digest,
            "bytes": len(raw),
            "lines": len(parsed),
        }
    return records, metadata


def layer_records(records: list[dict]) -> list[dict]:
    return [record for record in records if record.get("event") == "layer"]


def semantic_projection(records: list[dict]) -> tuple[dict, dict]:
    header = records[0]
    fields = header["semantic_projection"]
    layers = {
        (record["token_position"], record["layer"]): tuple(record.get(field) for field in fields)
        for record in layer_records(records)
    }
    final = {key: value for key, value in records[-1].items() if key != "phase_profile"}
    return layers, final


def require_semantic_exact(records: dict[str, list[dict]], keys: tuple[str, ...]) -> None:
    reference = semantic_projection(records[keys[0]])
    for key in keys[1:]:
        if semantic_projection(records[key]) != reference:
            raise RuntimeError(f"semantic comparison is not exact: {key}")


def routed_wall_ns(records: list[dict]) -> int:
    return sum(
        record["phase_profile"]["top_level"]["routed"]["wall_ns"]
        for record in layer_records(records)
    )


def median(values: list[int]) -> float:
    return statistics.median(values)


def collect(records: dict[str, list[dict]]) -> dict:
    for key in RAW_KEYS + HOST_KEYS + CACHE_OFF_KEYS + CACHE_ON_KEYS + ("cache_strict_audit",):
        if len(records[key]) != 307 or len(layer_records(records[key])) != 305:
            raise RuntimeError(f"unexpected full61 prompt record count: {key}")
    if len(records["pos5_one_token"]) != 369 or len(layer_records(records["pos5_one_token"])) != 366:
        raise RuntimeError("unexpected position-5 trace record count")

    require_semantic_exact(records, RAW_KEYS + HOST_KEYS)
    require_semantic_exact(records, CACHE_OFF_KEYS + CACHE_ON_KEYS)

    for key in RAW_KEYS:
        header = records[key][0]
        if (
            header.get("fixed_prefetch_wo_a_decode_arm") != "raw_fp8_gpu_decode"
            or header["physical"]["fixed_h2d_bytes"] != 24_967_930_296
        ):
            raise RuntimeError(f"raw GPU-decode arm changed: {key}")
    for key in HOST_KEYS:
        header = records[key][0]
        if (
            header.get("fixed_prefetch_wo_a_decode_arm") != "host_bf16_decode"
            or header["physical"]["fixed_h2d_bytes"] != 29_061_571_000
        ):
            raise RuntimeError(f"host BF16-decode arm changed: {key}")

    for key in CACHE_OFF_KEYS:
        header = records[key][0]
        physical = header["physical"]
        if (
            header.get("routed_qd4_immutable_expert_cache") is not False
            or physical["routed_source_bytes"] != 64_222_986_240
            or physical["routed_h2d_bytes"] != 64_222_986_240
        ):
            raise RuntimeError(f"cache-off arm changed: {key}")
    for key in CACHE_ON_KEYS:
        header = records[key][0]
        physical = header["physical"]
        if (
            header.get("routed_qd4_immutable_expert_cache") is not True
            or physical["routed_source_bytes"] != 29_619_781_632
            or physical["routed_h2d_bytes"] != 29_619_781_632
        ):
            raise RuntimeError(f"cache-on arm changed: {key}")

    strict = records["cache_strict_audit"][0]["physical"]["routed_qd4_immutable_expert_cache"]
    strict_expected = {
        "completed_layers": 61,
        "d2d_bytes": 64_222_986_240,
        "d2d_copies": 1830,
        "eviction": "none",
        "expert_union_loads": 844,
        "fallbacks": 0,
        "h2d_bytes": 29_619_781_632,
        "h2d_loads": 844,
        "hits": 986,
        "max_occupancy": 29,
        "route_executions": 1830,
        "slots": 32,
        "source_bytes": 29_619_781_632,
        "source_loads": 844,
        "worker_errors": 0,
    }
    if strict != strict_expected:
        raise RuntimeError("strict cache audit telemetry changed")

    raw_e2e = [records[key][0]["request_e2e_ns"] for key in RAW_KEYS]
    host_e2e = [records[key][0]["request_e2e_ns"] for key in HOST_KEYS]
    control_e2e = [records[key][0]["request_e2e_ns"] for key in CACHE_OFF_KEYS]
    cache_e2e = [records[key][0]["request_e2e_ns"] for key in CACHE_ON_KEYS]
    control_routed = [routed_wall_ns(records[key]) for key in CACHE_OFF_KEYS]
    cache_routed = [routed_wall_ns(records[key]) for key in CACHE_ON_KEYS]

    pos5_header = records["pos5_one_token"][0]
    pos5_layers = [
        record
        for record in layer_records(records["pos5_one_token"])
        if record.get("token_position") == 5
    ]
    if len(pos5_layers) != 61 or {record["layer"] for record in pos5_layers} != set(range(61)):
        raise RuntimeError("position-5 layer coverage changed")
    stage_keys = ("layer_prepare", "attention", "router", "shared", "routed")
    stages_ns = {
        key: sum(record["phase_profile"]["top_level"][key]["wall_ns"] for record in pos5_layers)
        for key in stage_keys
    }
    one_token = pos5_header["phase_profile"]["one_token_decode_outer"]
    if not one_token.get("exact"):
        raise RuntimeError("position-5 one-token closure is not exact")
    stage_sum_ns = sum(stages_ns.values())
    layer_gap_ns = one_token["decode_layer_outer_ns"] - stage_sum_ns
    if (
        layer_gap_ns < 0
        or stage_sum_ns + layer_gap_ns != one_token["decode_layer_outer_ns"]
        or one_token["decode_layer_outer_ns"] + one_token["decode_final_outer_ns"]
        != one_token["decode_outer_ns"]
    ):
        raise RuntimeError("position-5 component closure changed")

    return {
        "fixed_h2d": {
            "host_bytes": 29_061_571_000,
            "raw_bytes": 24_967_930_296,
            "saved_bytes": 4_093_640_704,
        },
        "routed_cache": {
            "off_bytes": 64_222_986_240,
            "on_bytes": 29_619_781_632,
            "saved_bytes": 34_603_204_608,
            "strict": strict_expected,
        },
        "timings": {
            "raw_e2e_ns": raw_e2e,
            "host_e2e_ns": host_e2e,
            "cache_off_e2e_ns": control_e2e,
            "cache_on_e2e_ns": cache_e2e,
            "cache_off_routed_ns": control_routed,
            "cache_on_routed_ns": cache_routed,
            "raw_e2e_median_ns": median(raw_e2e),
            "host_e2e_median_ns": median(host_e2e),
            "cache_off_e2e_median_ns": median(control_e2e),
            "cache_on_e2e_median_ns": median(cache_e2e),
            "cache_off_routed_median_ns": median(control_routed),
            "cache_on_routed_median_ns": median(cache_routed),
        },
        "pos5": {
            "stage_ns": stages_ns,
            "layer_gap_ns": layer_gap_ns,
            "decode_layer_outer_ns": one_token["decode_layer_outer_ns"],
            "decode_final_outer_ns": one_token["decode_final_outer_ns"],
            "decode_outer_ns": one_token["decode_outer_ns"],
            "rate_tok_s": 1_000_000_000 / one_token["decode_outer_ns"],
            "development_n": 1,
        },
    }


def axis(
    x0: float,
    x1: float,
    y: float,
    maximum: float,
    ticks: list[float],
    label_value: str,
) -> str:
    body = line(x0, y, x1, y, C["ink"], 1.5)
    for tick in ticks:
        x = x0 + tick / maximum * (x1 - x0)
        body += line(x, y, x, y + 7, C["ink"], 1)
        body += text(x, y + 25, f"{tick:g}", 12, C["muted"], "middle")
    body += text((x0 + x1) / 2, y + 48, label_value, 13, C["ink"], "middle")
    return body


def byte_savings_figure(
    title_value: str,
    subtitle: str,
    left_label: str,
    right_label: str,
    before_bytes: int,
    after_bytes: int,
    maximum_gb: float,
    ticks: list[float],
    note: str,
    source: str,
) -> str:
    height = 485
    x0, x1 = 250.0, 900.0
    before_gb, after_gb = before_bytes / 1e9, after_bytes / 1e9
    saved_gb = before_gb - after_gb
    scale = lambda value: x0 + value / maximum_gb * (x1 - x0)
    body = text(48, 44, title_value, 26, weight="bold")
    body += text(48, 73, subtitle, 15, C["muted"])

    for y, label_value, value, fill, stroke in (
        (130, left_label, before_gb, "url(#hatch-orange)", C["orange"]),
        (240, right_label, after_gb, C["blue"], C["blue"]),
    ):
        width = scale(value) - x0
        body += text(x0 - 18, y + 31, label_value, 15, C["ink"], "end", "bold")
        body += rect(x0, y, width, 48, fill, stroke, 2, 2)
        body += text(x0 + width - 10, y + 31, f"{value:.3f} GB", 15, C["ink"], "end", "bold")

    before_x, after_x = scale(before_gb), scale(after_gb)
    body += line(after_x, 218, before_x, 218, C["vermillion"], 2)
    body += line(after_x, 209, after_x, 227, C["vermillion"], 2)
    body += line(before_x, 209, before_x, 227, C["vermillion"], 2)
    body += text(
        (before_x + after_x) / 2,
        205,
        f"−{saved_gb:.3f} GB  (−{saved_gb / before_gb * 100:.1f}%)",
        16,
        C["vermillion"],
        "middle",
        "bold",
    )
    body += axis(x0, x1, 315, maximum_gb, ticks, "Transferred bytes (decimal GB = 10⁹ bytes)")
    body += text(48, 405, note, 14, C["ink"], weight="bold")
    body += text(
        48,
        430,
        "Bars start at zero. Fill pattern and solid fill preserve grayscale separation.",
        12,
        C["muted"],
    )
    body += footer(source, height)
    return svg(title_value, subtitle, body, height)


def paired_panel(
    x: float,
    title_value: str,
    left_label: str,
    right_label: str,
    left_values_ns: list[int],
    right_values_ns: list[int],
    maximum_s: float,
    ticks: list[float],
) -> str:
    width = 260.0
    top, bottom = 160.0, 490.0
    left_x, right_x = x + 82, x + 205
    y_of = lambda value_s: bottom - value_s / maximum_s * (bottom - top)
    body = text(x + width / 2, 132, title_value, 15, C["ink"], "middle", "bold")
    body += line(x + 45, top, x + 45, bottom, C["ink"], 1.4)
    body += line(x + 45, bottom, x + width - 10, bottom, C["ink"], 1.4)
    for tick in ticks:
        y = y_of(tick)
        body += line(x + 45, y, x + width - 10, y, C["grid"], 1)
        body += text(x + 38, y + 4, f"{tick:g}", 11, C["muted"], "end")

    colors = (C["blue"], C["orange"])
    dashes = ("", "7 5")
    for index, (left_ns, right_ns) in enumerate(zip(left_values_ns, right_values_ns)):
        left_s, right_s = left_ns / 1e9, right_ns / 1e9
        ly, ry = y_of(left_s), y_of(right_s)
        body += line(left_x, ly, right_x, ry, colors[index], 2.2, dashes[index])
        marker = circle if index == 0 else square
        body += marker(left_x, ly, 5, C["paper"], colors[index], 2)
        body += marker(right_x, ry, 5, C["paper"], colors[index], 2)
        dy = -8 if index == 0 else 15
        body += text(left_x - 8, ly + dy, fmt(left_s, 2), 11, colors[index], "end", "bold")
        body += text(right_x + 8, ry + dy, fmt(right_s, 2), 11, colors[index], "start", "bold")

    left_median = median(left_values_ns) / 1e9
    right_median = median(right_values_ns) / 1e9
    for point_x, value in ((left_x, left_median), (right_x, right_median)):
        y = y_of(value)
        body += line(point_x - 12, y, point_x + 12, y, C["ink"], 4)
    body += text(left_x, 516, left_label, 12, C["ink"], "middle", "bold")
    body += text(right_x, 516, right_label, 12, C["ink"], "middle", "bold")
    body += text(
        x + width / 2,
        551,
        f"median left/right = {left_median / right_median:.3f}×",
        12,
        C["ink"],
        "middle",
        "bold",
    )
    return body


def timings_figure(data: dict) -> str:
    height = 640
    timing = data["timings"]
    title_value = "Measured development timings: all paired n=2 runs"
    body = text(48, 42, title_value, 25, weight="bold")
    body += text(
        48,
        70,
        "Each line joins one ABBA pair. Short black ticks are medians; every recorded run remains visible.",
        14,
        C["muted"],
    )
    body += line(52, 101, 82, 101, C["blue"], 2.2)
    body += circle(67, 101, 5, C["paper"], C["blue"], 2)
    body += text(92, 106, "Pair 1", 12, C["ink"])
    body += line(168, 101, 198, 101, C["orange"], 2.2, "7 5")
    body += square(183, 101, 5, C["paper"], C["orange"], 2)
    body += text(208, 106, "Pair 2", 12, C["ink"])
    body += line(282, 101, 306, 101, C["ink"], 4)
    body += text(316, 106, "Median", 12, C["ink"])
    body += text(910, 106, "Seconds", 12, C["muted"], "end")

    body += paired_panel(
        42,
        "Fixed decode · request E2E",
        "Host BF16",
        "Raw GPU",
        timing["host_e2e_ns"],
        timing["raw_e2e_ns"],
        40,
        [0, 10, 20, 30, 40],
    )
    body += paired_panel(
        350,
        "Routed cache · request E2E",
        "Cache off",
        "32-slot cache",
        timing["cache_off_e2e_ns"],
        timing["cache_on_e2e_ns"],
        40,
        [0, 10, 20, 30, 40],
    )
    body += paired_panel(
        658,
        "Routed cache · routed wall",
        "Cache off",
        "32-slot cache",
        timing["cache_off_routed_ns"],
        timing["cache_on_routed_ns"],
        18,
        [0, 5, 10, 15, 18],
    )
    body += text(
        48,
        587,
        "MEASURED DEVELOPMENT · n=2 per arm · descriptive paired ratios · individual runs shown.",
        13,
        C["vermillion"],
        weight="bold",
    )
    body += footer("MEASURED DEVELOPMENT · paired request and routed-wall timings", height)
    return svg(title_value, "Paired raw development timings with every run and median shown.", body, height)


def pos5_figure(data: dict) -> str:
    height = 600
    pos5 = data["pos5"]
    total_ns = pos5["decode_outer_ns"]
    title_value = "Measured one-token DAG: fixed staging dominates 36.979 s"
    body = text(48, 43, title_value, 25, weight="bold")
    body += text(
        48,
        72,
        "V4-Pro position 5 · one development record · 61 layers · exact outer-time closure",
        14,
        C["muted"],
    )
    x0, x1 = 48.0, 912.0
    scale = lambda value_ns: value_ns / total_ns * (x1 - x0)
    components = [
        ("Fixed staging", pos5["stage_ns"]["layer_prepare"], C["blue"]),
        ("Attention", pos5["stage_ns"]["attention"], "url(#hatch-sky)"),
        ("Router", pos5["stage_ns"]["router"], C["orange"]),
        ("Shared", pos5["stage_ns"]["shared"], "url(#dots-green)"),
        ("Routed", pos5["stage_ns"]["routed"], "url(#cross-vermillion)"),
        ("Layer gap", pos5["layer_gap_ns"], C["gap"]),
        ("Final head", pos5["decode_final_outer_ns"], "url(#vertical-purple)"),
    ]
    cursor = x0
    for _, value_ns, fill_value in components:
        width = scale(value_ns)
        body += rect(cursor, 145, max(width, 0.8), 72, fill_value, C["ink"], 0, 0.8)
        cursor += width
    fixed_width = scale(pos5["stage_ns"]["layer_prepare"])
    body += text(x0 + fixed_width / 2, 181, "Fixed staging", 17, C["paper"], "middle", "bold")
    body += text(
        x0 + fixed_width / 2,
        202,
        f"{pos5['stage_ns']['layer_prepare'] / 1e9:.3f} s · {pos5['stage_ns']['layer_prepare'] / total_ns * 100:.1f}%",
        13,
        C["paper"],
        "middle",
        "bold",
    )
    body += line(x0, 127, x0 + scale(pos5["decode_layer_outer_ns"]), 127, C["ink"], 1.5)
    body += line(x0, 121, x0, 133, C["ink"], 1.5)
    body += line(
        x0 + scale(pos5["decode_layer_outer_ns"]),
        121,
        x0 + scale(pos5["decode_layer_outer_ns"]),
        133,
        C["ink"],
        1.5,
    )
    body += text(
        x0 + scale(pos5["decode_layer_outer_ns"]) / 2,
        116,
        f"61-layer decode outer · {pos5['decode_layer_outer_ns'] / 1e9:.3f} s",
        13,
        C["ink"],
        "middle",
        "bold",
    )
    body += text(912, 248, f"Total {total_ns / 1e9:.9f} s", 19, C["ink"], "end", "bold")
    body += text(
        912,
        274,
        f"1 / total = {pos5['rate_tok_s']:.3f} next-token/s",
        15,
        C["vermillion"],
        "end",
        "bold",
    )

    legend = [
        ("Fixed staging", pos5["stage_ns"]["layer_prepare"], C["blue"]),
        ("Attention", pos5["stage_ns"]["attention"], "url(#hatch-sky)"),
        ("Router", pos5["stage_ns"]["router"], C["orange"]),
        ("Shared expert", pos5["stage_ns"]["shared"], "url(#dots-green)"),
        ("Routed experts", pos5["stage_ns"]["routed"], "url(#cross-vermillion)"),
        ("Layer gap", pos5["layer_gap_ns"], C["gap"]),
        ("Final head", pos5["decode_final_outer_ns"], "url(#vertical-purple)"),
    ]
    for index, (label_value, value_ns, fill_value) in enumerate(legend):
        col, row = index % 3, index // 3
        x, y = 54 + col * 300, 328 + row * 68
        body += rect(x, y, 34, 22, fill_value, C["ink"], 1, 1)
        body += text(x + 46, y + 16, label_value, 13, C["ink"], weight="bold")
        body += text(x + 252, y + 16, f"{value_ns / 1e9:.3f} s", 13, C["ink"], "end")
    body += text(
        48,
        530,
        "The 0.005 s layer gap is shown so the stage components close exactly to the recorded outer time.",
        13,
        C["muted"],
    )
    body += text(
        48,
        554,
        "MEASURED DEVELOPMENT · one next-token decode · sustained rate remains unmeasured.",
        13,
        C["vermillion"],
        weight="bold",
    )
    body += footer("MEASURED DEVELOPMENT · position 5 · n=1", height)
    return svg(title_value, "Measured position-5 next-token time decomposition with exact accounting closure.", body, height)


def hardware_ladder_figure(data: dict) -> str:
    height = 650
    title_value = "Measured / Calibrated / Target hardware evidence map"
    body = text(48, 42, title_value, 25, weight="bold")
    body += text(
        48,
        70,
        "Current evidence and future gates are separated because their models, estimands, and assumptions differ.",
        14,
        C["muted"],
    )
    body += text(
        48,
        93,
        "Separate rate frames preserve the distinct models, estimands, and assumptions.",
        13,
        C["vermillion"],
        weight="bold",
    )
    body += text(48, 132, "CURRENT EVIDENCE", 13, C["ink"], weight="bold")
    body += text(480, 132, "FUTURE ACCEPTANCE GATES", 13, C["ink"], weight="bold")

    # Measured: solid border + solid point.
    body += rect(48, 150, 392, 126, C["paper"], C["blue"], 8, 2.5)
    body += circle(72, 176, 8, C["blue"], C["ink"], 1.5)
    body += text(92, 181, "MEASURED DEVELOPMENT", 12, C["blue"], weight="bold")
    body += text(70, 215, "Pro · 12 GiB", 18, C["ink"], weight="bold")
    body += text(
        418,
        215,
        f"{data['pos5']['rate_tok_s']:.3f} next-token/s",
        18,
        C["blue"],
        "end",
        "bold",
    )
    body += text(70, 245, "One position-5 decode; sustained rate remains unmeasured.", 12, C["muted"])

    # Calibrated: hatched cue and explicit run-count boundary.
    body += rect(48, 296, 392, 126, C["paper"], C["orange"], 8, 2.5)
    body += rect(64, 316, 18, 18, "url(#hatch-orange)", C["orange"], 1, 1.5)
    body += text(92, 330, "CALIBRATED", 12, C["orange"], weight="bold")
    body += text(70, 364, "Flash · 12 GiB", 18, C["ink"], weight="bold")
    body += text(418, 364, "0.08 to 0.12 token/s", 18, C["orange"], "end", "bold")
    body += text(70, 394, "Cold H1 projection from Pro service · Flash run count: 0.", 12, C["muted"])

    def target_card(
        y: float,
        model: str,
        band: str,
        assumption: str,
        current_path: str = "",
    ) -> str:
        card = rect(480, y, 432, 105, C["paper"], C["green"], 8, 2.5, "8 5")
        card += rect(496, y + 15, 20, 20, C["paper"], C["green"], 1, 2, "5 3")
        card += text(526, y + 31, "TARGET · FUTURE ACCEPTANCE GATE", 12, C["green"], weight="bold")
        card += text(502, y + 63, model, 17, C["ink"], weight="bold")
        card += text(890, y + 63, band, 17, C["green"], "end", "bold")
        card += text(502, y + 88, assumption, 12, C["muted"])
        if current_path:
            card += text(890, y + 88, current_path, 11, C["vermillion"], "end", "bold")
        return card

    body += target_card(
        150,
        "Flash · 16 GiB",
        "0.25 to 0.40 tok/s",
        "Resident-fixed H1 implementation remains pending.",
    )
    body += target_card(
        272,
        "Flash · 24 GiB",
        "0.50 to 1.00 committed tok/s",
        "H2 · q=0.8 planning assumption.",
        "current H2 calibration: 0.20 to 0.29",
    )
    body += target_card(
        394,
        "Flash · 32 to 48 GiB",
        "0.70 to 1.50 committed tok/s",
        "H5 · q=0.8 planning assumption.",
        "current H5 calibration: 0.26 to 0.37",
    )
    body += text(
        48,
        548,
        "TARGET defines a future pass/fail threshold. Hardware rate estimation remains a separate gate.",
        13,
        C["vermillion"],
        weight="bold",
    )
    body += text(
        48,
        576,
        "Target gates still require measured SSD/PCIe service, residency, context, q, route-hit, and overlap.",
        12,
        C["muted"],
    )
    body += footer("EVIDENCE TIERS · MEASURED DEVELOPMENT · CALIBRATED · TARGET", height)
    return svg(title_value, "Current evidence and future hardware acceptance gates use separate rate frames.", body, height)


def provenance(data: dict, metadata: dict[str, dict]) -> dict:
    timing = data["timings"]
    pos5 = data["pos5"]
    flash_report_sha = sha256_path(FLASH_REPORT)
    return {
        "schema": "colibri-readme-cost-figures-v1",
        "generator": "docs/assets/readme/generate_cost_figures.py",
        "units": {
            "source_bytes": "bytes",
            "display_storage": "decimal GB = bytes / 1e9",
            "time": "nanoseconds in artifacts; seconds in figures",
            "rate": "tokens per second",
            "hardware_capacity": "GiB",
        },
        "artifact_identity": {
            **metadata,
            "flash_migration_research": {
                "path": "docs/FLASH_MIGRATION_RESEARCH.md",
                "sha256": flash_report_sha,
                "role": "calibration formulas and projection boundary; no Flash run",
            },
        },
        "figures": {
            "13-fixed-h2d-savings.svg": {
                "evidence_class": "Measured",
                "scope": "V4-Pro full61 development ABBA; n=2 per arm",
                "field": "run.physical.fixed_h2d_bytes",
                "host_bf16_decode_bytes": data["fixed_h2d"]["host_bytes"],
                "raw_gpu_decode_bytes": data["fixed_h2d"]["raw_bytes"],
                "saved_bytes": data["fixed_h2d"]["saved_bytes"],
                "formula": "saved = host_bf16_decode_bytes - raw_gpu_decode_bytes",
            },
            "14-routed-cache-savings.svg": {
                "evidence_class": "Measured",
                "scope": "V4-Pro full61 development ABBA; n=2 per arm; independently strict-audited",
                "fields": ["run.physical.routed_source_bytes", "run.physical.routed_h2d_bytes"],
                "cache_off_bytes_per_field": data["routed_cache"]["off_bytes"],
                "cache_on_bytes_per_field": data["routed_cache"]["on_bytes"],
                "saved_bytes_per_field": data["routed_cache"]["saved_bytes"],
                "formula": "saved_per_field = cache_off_bytes_per_field - cache_on_bytes_per_field",
                "strict_audit": data["routed_cache"]["strict"],
            },
            "15-development-paired-timings.svg": {
                "evidence_class": "Measured development",
                "n_per_arm": 2,
                "pairs": {
                    "fixed_decode_e2e_ns": [
                        {"pair": 1, "host_bf16": timing["host_e2e_ns"][0], "raw_gpu": timing["raw_e2e_ns"][0]},
                        {"pair": 2, "host_bf16": timing["host_e2e_ns"][1], "raw_gpu": timing["raw_e2e_ns"][1]},
                    ],
                    "routed_cache_e2e_ns": [
                        {"pair": 1, "cache_off": timing["cache_off_e2e_ns"][0], "cache_on": timing["cache_on_e2e_ns"][0]},
                        {"pair": 2, "cache_off": timing["cache_off_e2e_ns"][1], "cache_on": timing["cache_on_e2e_ns"][1]},
                    ],
                    "routed_cache_wall_ns": [
                        {"pair": 1, "cache_off": timing["cache_off_routed_ns"][0], "cache_on": timing["cache_on_routed_ns"][0]},
                        {"pair": 2, "cache_off": timing["cache_off_routed_ns"][1], "cache_on": timing["cache_on_routed_ns"][1]},
                    ],
                },
                "median_ratios_left_over_right": {
                    "host_over_raw_e2e": timing["host_e2e_median_ns"] / timing["raw_e2e_median_ns"],
                    "cache_off_over_on_e2e": timing["cache_off_e2e_median_ns"] / timing["cache_on_e2e_median_ns"],
                    "cache_off_over_on_routed_wall": timing["cache_off_routed_median_ns"] / timing["cache_on_routed_median_ns"],
                },
                "uncertainty_interval": None,
                "note": "All points and pair lines are shown; n=2 is not used to draw a CI.",
            },
            "16-pos5-one-token-dag.svg": {
                "evidence_class": "Measured development",
                "n": 1,
                "token_position": 5,
                "layer_records": 61,
                "stage_ns": pos5["stage_ns"],
                "layer_unattributed_gap_ns": pos5["layer_gap_ns"],
                "decode_layer_outer_ns": pos5["decode_layer_outer_ns"],
                "decode_final_outer_ns": pos5["decode_final_outer_ns"],
                "decode_outer_ns": pos5["decode_outer_ns"],
                "next_token_rate_tok_s": pos5["rate_tok_s"],
                "formulas": [
                    "sum(stage_ns) + layer_unattributed_gap_ns = decode_layer_outer_ns",
                    "decode_layer_outer_ns + decode_final_outer_ns = decode_outer_ns",
                    "next_token_rate_tok_s = 1e9 / decode_outer_ns",
                ],
                "note": "One development next-token decode; not sustained throughput.",
            },
            "17-hardware-evidence-ladder.svg": {
                "display_structure": "Current evidence and future gates are separated; there is no shared rate axis.",
                "mixed_workload_warning": "Rows are not like-for-like benchmark speedups.",
                "rows": [
                    {
                        "model": "V4-Pro",
                        "vram_gib": 12,
                        "evidence_class": "Measured",
                        "rate_tok_s": pos5["rate_tok_s"],
                        "formula": "1e9 / measured position-5 decode_outer_ns",
                        "scope": "one H1 next-token development decode; not sustained throughput",
                    },
                    {
                        "model": "V4-Flash",
                        "vram_gib": 12,
                        "evidence_class": "Calibrated",
                        "hardcoded_display_band_tok_s": [0.08, 0.12],
                        "source": "docs/FLASH_MIGRATION_RESEARCH.md section 6.3",
                        "formula": "cold H1 projection 8-12 s gives reciprocal 0.083-0.125 tok/s; display band is conservatively rounded",
                    },
                    {
                        "model": "V4-Flash",
                        "vram_gib": 16,
                        "evidence_class": "Target",
                        "hardcoded_band_tok_s": [0.25, 0.40],
                        "source": "docs/FLASH_MIGRATION_RESEARCH.md section 6.3 resident-fixed H1 hypothesis",
                        "formula": "1 / [4.0, 2.5] s; requires unimplemented short-context fixed residency",
                    },
                    {
                        "model": "V4-Flash",
                        "vram_gib": 24,
                        "evidence_class": "Target",
                        "hardcoded_band_tok_s": [0.5, 1.0],
                        "assumptions": {"H": 2, "q": 0.8},
                        "current_path_calibrated_band_tok_s": [0.20, 0.29],
                        "formula": "No predictive formula: this is a hardcoded acceptance gate. At q=0.8, meeting it would require T_verify=2.44-4.88 s; that consequence is not evidence.",
                        "source": "README planning brief; current report calibrates only 0.20-0.29 committed tok/s for H2",
                    },
                    {
                        "model": "V4-Flash",
                        "vram_gib": [32, 48],
                        "evidence_class": "Target",
                        "hardcoded_band_tok_s": [0.7, 1.5],
                        "assumptions": {"H": 5, "q": 0.8},
                        "current_path_calibrated_band_tok_s": [0.26, 0.37],
                        "formula": "No predictive formula: this is a hardcoded acceptance gate. At q=0.8, meeting it would require T_verify=2.46-5.27 s; that consequence is not evidence.",
                        "source": "README planning brief; current report calibrates only 0.26-0.37 committed tok/s for H5",
                    },
                ],
            },
        },
    }


def render_all(data: dict, metadata: dict[str, dict]) -> dict[str, bytes]:
    fixed = data["fixed_h2d"]
    routed = data["routed_cache"]
    outputs = {
        "13-fixed-h2d-savings.svg": byte_savings_figure(
            "Measured fixed H2D: raw GPU decode avoids 4.094 GB",
            "V4-Pro full61 · development ABBA · n=2 per arm · identical bytes within each arm",
            "Host BF16 decode",
            "Raw GPU decode",
            fixed["host_bytes"],
            fixed["raw_bytes"],
            35,
            [0, 5, 10, 15, 20, 25, 30, 35],
            "Measured field: physical.fixed_h2d_bytes. E2E timing is shown separately in Figure 15.",
            "MEASURED DEVELOPMENT · paired fixed-weight transfer evidence",
        ),
        "14-routed-cache-savings.svg": byte_savings_figure(
            "Measured routed cache: 34.603 GB less source and H2D each",
            "V4-Pro full61 · development ABBA · n=2 per arm · 32-slot immutable cache",
            "Cache off",
            "32-slot cache",
            routed["off_bytes"],
            routed["on_bytes"],
            70,
            [0, 10, 20, 30, 40, 50, 60, 70],
            "Strict audit: 1,830 routes · 844 loads · 986 hits · 61/61 layers · 0 fallback/errors.",
            "MEASURED DEVELOPMENT · paired cache evidence + strict accounting audit",
        ),
        "15-development-paired-timings.svg": timings_figure(data),
        "16-pos5-one-token-dag.svg": pos5_figure(data),
        "17-hardware-evidence-ladder.svg": hardware_ladder_figure(data),
    }
    outputs["cost-figures-provenance.json"] = (
        json.dumps(provenance(data, metadata), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return {
        name: value.encode("utf-8") if isinstance(value, str) else value
        for name, value in outputs.items()
    }


def validate_outputs(outputs: dict[str, bytes]) -> None:
    for name, raw in outputs.items():
        if name.endswith(".svg"):
            root = ET.fromstring(raw)
            if root.tag != "{http://www.w3.org/2000/svg}svg":
                raise RuntimeError(f"invalid SVG root: {name}")
            if root.attrib.get("viewBox") is None:
                raise RuntimeError(f"missing SVG viewBox: {name}")
        elif name.endswith(".json"):
            json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=f"directory containing frozen JSONL artifacts (default: {DEFAULT_ARTIFACT_ROOT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="render twice and require byte-stable SVG/JSON output",
    )
    args = parser.parse_args()
    records, metadata = read_artifacts(args.artifact_root)
    data = collect(records)
    first = render_all(data, metadata)
    validate_outputs(first)
    if args.check:
        second = render_all(data, metadata)
        if first != second:
            raise RuntimeError("cost figure output is not deterministic")
    for name, raw in first.items():
        (OUT / name).write_bytes(raw)
    hashes = {name: sha256_bytes(raw) for name, raw in sorted(first.items())}
    print(json.dumps({"status": "PASS", "files": sorted(first), "sha256": hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
