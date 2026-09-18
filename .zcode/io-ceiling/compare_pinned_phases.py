import json
import statistics
import sys


def routed_stats(path):
    d = json.load(open(path))
    rows = []
    for layer in d["phase_profile"]["layers"]:
        tl = layer["top_level"]
        r = tl["routed"]["diagnostics"]
        rows.append(
            {
                "outer": layer["outer"]["wall_ns"] / 1e6,
                "shared": tl["shared"]["wall_ns"] / 1e6,
                "routed": tl["routed"]["wall_ns"] / 1e6,
                "read": r["read_ns"] / 1e6,
                "alloc": r["alloc_ns"] / 1e6,
                "setup": r["setup_ns"] / 1e6,
                "h2d": r["h2d_ns"] / 1e6,
                "kernel": r["kernel_ns"] / 1e6,
                "fixed": tl["fixed"]["wall_ns"] / 1e6,
                "attn": tl["attention"]["wall_ns"] / 1e6,
            }
        )
    return d["request_e2e_ns"] / 1e9, rows


def summarize(name, path):
    e2e, rows = routed_stats(path)
    out = {"e2e_s": e2e}
    for k in rows[0]:
        out[k] = statistics.mean(r[k] for r in rows)
    print("%-26s e2e=%6.2fs " % (name, out["e2e_s"]) +
          " ".join("%s=%.2f" % (k, out[k]) for k in
                   ("outer", "shared", "routed", "read", "alloc", "setup", "h2d", "kernel", "fixed", "attn")))
    return out


a = summarize("baseline", sys.argv[1])
b = summarize("pinned-lanes", sys.argv[2])
print()
print("%-10s %10s %10s %10s" % ("field", "baseline", "pinned", "delta"))
for k in ("outer", "shared", "routed", "read", "alloc", "setup", "h2d", "kernel", "fixed", "attn"):
    print("%-10s %10.2f %10.2f %+10.2f" % (k, a[k], b[k], b[k] - a[k]))
print("%-10s %10.3f %10.3f %+10.3f" % ("e2e_s", a["e2e_s"], b["e2e_s"], b["e2e_s"] - a["e2e_s"]))
