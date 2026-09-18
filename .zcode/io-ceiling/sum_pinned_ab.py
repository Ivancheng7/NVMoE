import glob
import json
import statistics
import sys

OUT = "/srv/localmoe/targets/io-ceiling"
REF = ["5ed8bf88beae3cfe", "316e5495ca424b39", "a521547b2586e458", "1963dfeaaf30264c",
       "b970da7913a57a1c", "12c6b45cecf9d53b", "e0f441542fa53991", "69ed859929ee9042"]


def row(path):
    d = json.load(open(path))
    layers = d["phase_profile"]["layers"]
    digests = [t["hidden_bf16_digest"] for t in d["requests"][0]["tokens"]]

    def mean(get):
        return statistics.mean(get(l) for l in layers)

    return {
        "e2e": d["request_e2e_ns"] / 1e9,
        "outer": mean(lambda l: l["outer"]["wall_ns"] / 1e6),
        "shared": mean(lambda l: l["top_level"]["shared"]["wall_ns"] / 1e6),
        "routed": mean(lambda l: l["top_level"]["routed"]["wall_ns"] / 1e6),
        "read": mean(lambda l: l["top_level"]["routed"]["diagnostics"]["read_ns"] / 1e6),
        "alloc": mean(lambda l: l["top_level"]["routed"]["diagnostics"]["alloc_ns"] / 1e6),
        "setup": mean(lambda l: l["top_level"]["routed"]["diagnostics"]["setup_ns"] / 1e6),
        "h2d": mean(lambda l: l["top_level"]["routed"]["diagnostics"]["h2d_ns"] / 1e6),
        "kernel": mean(lambda l: l["top_level"]["routed"]["diagnostics"]["kernel_ns"] / 1e6),
        "fixed": mean(lambda l: l["top_level"]["fixed"]["wall_ns"] / 1e6),
        "attn": mean(lambda l: l["top_level"]["attention"]["wall_ns"] / 1e6),
        "digests_ok": digests == REF,
    }


for arm in ("baseline", "pinned"):
    print("==", arm)
    for path in sorted(glob.glob("%s/pinned-ab-%s-r*.jsonl" % (OUT, arm))):
        r = row(path)
        print("  %-40s e2e=%.2fs routed=%.2f setup=%.2f alloc=%.2f read=%.2f digests_ok=%s" %
              (path.split("/")[-1], r["e2e"], r["routed"], r["setup"], r["alloc"], r["read"], r["digests_ok"]))

print()
print("field        baseline  pinned   delta   (mean of 3)")
fields = ("e2e", "outer", "shared", "routed", "read", "alloc", "setup", "h2d", "kernel", "fixed", "attn")
acc = {}
for arm in ("baseline", "pinned"):
    rows = [row(p) for p in sorted(glob.glob("%s/pinned-ab-%s-r*.jsonl" % (OUT, arm)))]
    acc[arm] = {k: statistics.mean(r[k] for r in rows) for k in fields}
for k in fields:
    print("%-10s %8.3f %8.3f %+8.3f" % (k, acc["baseline"][k], acc["pinned"][k], acc["pinned"][k] - acc["baseline"][k]))
