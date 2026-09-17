"""Summarise SHAREDPROBE lines: what actually fills the `shared` phase wall.

Usage: python3 shared_window_sum.py STDERR [STDERR...]
"""
import re
import statistics
import sys

PER_TOKEN = 43
LINE = re.compile(r"SHAREDPROBE layer=(\d+) rows=(\d+) prefetch=(\S+) window_us=(\d+) "
                  r"mlp_us=(\d+) workers=(\d+) worker_sum_us=(\d+) worker_max_us=(\d+) "
                  r"join_wait_us=(\d+)")


def parse(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LINE.search(line)
            if match:
                rows.append(
                    {
                        "layer": int(match.group(1)),
                        "prefetch": match.group(3) == "true",
                        "window": int(match.group(4)),
                        "mlp": int(match.group(5)),
                        "workers": int(match.group(6)),
                        "worker_sum": int(match.group(7)),
                        "worker_max": int(match.group(8)),
                        "join_wait": int(match.group(9)),
                    }
                )
    return rows


def main():
    print(f"{'run':<34} {'lay':>4} {'window':>7} {'mlp':>7} {'wmax':>7} {'wsum':>7} "
          f"{'join':>6} {'bound by':>9}")
    for path in sys.argv[1:]:
        rows = parse(path)
        if not rows:
            print(f"{path}: no SHAREDPROBE lines")
            continue
        # Drop the first layer of each token: it pays the cold page cache.
        body = [row for index, row in enumerate(rows) if index % PER_TOKEN != 0]
        if not body:
            body = rows
        mean = lambda key: statistics.mean(row[key] for row in body)
        read_bound = sum(1 for row in body if row["worker_max"] > row["mlp"])
        name = path.split("/")[-1].replace(".stderr", "")[:33]
        print(
            f"{name:<34} {len(body):>4} {mean('window'):>7.1f} {mean('mlp'):>7.1f} "
            f"{mean('worker_max'):>7.1f} {mean('worker_sum'):>7.1f} {mean('join_wait'):>6.1f} "
            f"{read_bound * 100 // len(body):>7}% read"
        )
        workers = statistics.mean(row["workers"] for row in body)
        if mean("worker_sum") > 0 and workers:
            print(
                f"{'':<34} workers={workers:.0f}  mlp range "
                f"{min(r['mlp'] for r in body)}-{max(r['mlp'] for r in body)} us, "
                f"wmax range {min(r['worker_max'] for r in body)}-"
                f"{max(r['worker_max'] for r in body)} us"
            )
        order = sorted(body, key=lambda row: row["window"])
        for label, row in (("p10", order[len(order) // 10]),
                           ("p50", order[len(order) // 2]),
                           ("p90", order[len(order) * 9 // 10])):
            print(
                f"{'':<34} {label}: window={row['window']:>6} mlp={row['mlp']:>6} "
                f"wmax={row['worker_max']:>6} join={row['join_wait']:>6}"
            )


if __name__ == "__main__":
    main()
