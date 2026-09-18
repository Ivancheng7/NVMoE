import json
import sys

new_path, ref_path = sys.argv[1], sys.argv[2]
new = json.load(open(new_path))
ref = json.load(open(ref_path))


def sig(d):
    t = d["requests"][0]["tokens"][0]
    return {
        "hidden_bf16_digest": t["hidden_bf16_digest"],
        "ordered_layer_digest_sha256": t["ordered_layer_digest_sha256"],
        "L0_route_ids": t["layers"][0]["route_ids"],
        "layers": len(t["layers"]),
        "peak_device_used_bytes": d.get("peak_device_used_bytes"),
    }


a, b = sig(new), sig(ref)
for k in a:
    verdict = "SAME" if a[k] == b[k] else "DIFF"
    print("%-28s %s  new=%s ref=%s" % (k, verdict, a[k], b[k]))

nd = [t["hidden_bf16_digest"] for t in new["requests"][0]["tokens"]]
rd = [t["hidden_bf16_digest"] for t in ref["requests"][0]["tokens"]]
print("all-token digests match:", nd == rd)
if nd != rd:
    for i, (x, y) in enumerate(zip(nd, rd)):
        if x != y:
            print("  token", i, "new", x, "ref", y)
print("new e2e_ns:", new["request_e2e_ns"], "ref:", ref["request_e2e_ns"])
