import glob
import json
import statistics as s
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "t1"
OUT = "/srv/localmoe/targets/io-ceiling"
REF = ["5ed8bf88beae3cfe", "316e5495ca424b39", "a521547b2586e458", "1963dfeaaf30264c",
       "b970da7913a57a1c", "12c6b45cecf9d53b", "e0f441542fa53991", "69ed859929ee9042"]


def load(p):
    d = json.load(open(p))
    L = d["phase_profile"]["layers"]
    def m(f): return s.mean(f(l) for l in L)
    tl = lambda l: l["top_level"]
    def g(l, k):
        v = tl(l).get(k)
        return v["wall_ns"] / 1e6 if isinstance(v, dict) else 0.0
    return dict(
        e2e=d["request_e2e_ns"] / 1e9,
        outer=m(lambda l: l["outer"]["wall_ns"] / 1e6),
        shared=m(lambda l: g(l, "shared")),
        routed=m(lambda l: g(l, "routed")),
        attention=m(lambda l: g(l, "attention")),
        fixed=m(lambda l: g(l, "fixed")),
        digs=[t["hidden_bf16_digest"] for t in d["requests"][0]["tokens"]],
    )


arms = {}
for arm in ("baseline", "routed", "both"):
    rows = []
    for p in sorted(glob.glob("%s/three-%s-%s-r*.jsonl" % (OUT, TAG, arm))):
        rows.append(load(p))
    arms[arm] = rows
    for p, r in zip(sorted(glob.glob("%s/three-%s-%s-r*.jsonl" % (OUT, TAG, arm))), rows):
        print("%-9s %s e2e=%6.2f outer=%7.2f attn=%6.2f routed=%6.2f digs_ok=%s" %
              (arm, p[-11:-6], r["e2e"], r["outer"], r["attention"], r["routed"], r["digs"] == REF))

print()
print("%-10s %9s %9s %9s" % ("field", "baseline", "routed", "both"))
mean = lambda arm, k: s.mean(r[k] for r in arms[arm])
for k in ("e2e", "outer", "shared", "routed", "attention", "fixed"):
    print("%-10s %9.3f %9.3f %9.3f" % (k, mean("baseline", k), mean("routed", k), mean("both", k)))
print()
print("per-run e2e (paired by round):")
for i in range(len(arms["baseline"])):
    b, r, t = arms["baseline"][i]["e2e"], arms["routed"][i]["e2e"], arms["both"][i]["e2e"]
    print("  round %d: base=%.2f routed=%.2f both=%.2f   (routed %+.2f, both %+.2f, sum-of-parts %+.2f)" %
          (i + 1, b, r, t, r - b, t - b, (r - b) + (t - r)))
print("digests all identical:", all(r["digs"] == REF for arm in arms for r in arms[arm]))
