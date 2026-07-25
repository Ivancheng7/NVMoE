#!/usr/bin/env python3
"""Run the fixed V4-Pro direct/cached read matrix on one verified ext4 snapshot."""

import argparse
import csv
import json
import os
import stat
import subprocess
from pathlib import Path


REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
SHARD_BYTES = 864_721_029_744
SIZES_MIB = (35, 210, 421)
THREADS = (1, 2, 4, 8)
FIELDS = (
    "cell", "repeat", "requested_mode", "actual_mode", "threads",
    "concurrency", "reads", "block_mib", "bytes", "pread_return_wall_elapsed_s",
    "pread_return_wall_definition", "intermediate_digest_in_pread_return_wall",
    "final_aggregate_digest_in_pread_return_wall", "pread_return_wall_gbps",
    "pread_return_wall_gbps_definition", "read_elapsed_s", "read_elapsed_definition",
    "read_timing", "max_thread_pread_elapsed_s", "verified_elapsed_s",
    "verified_elapsed_definition", "verified_pipeline_gbps",
    "verified_pipeline_gbps_definition", "read_gbps", "read_gbps_definition", "digest",
    "digest_algorithm",
    "digest_in_read_timing", "revision", "device",
    "fstype", "model_total_bytes",
)


def fail(message):
    raise SystemExit(message)


def parse_result(stdout):
    prefix = "IOBENCH_JSON "
    rows = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        fail("iobench emitted no unique IOBENCH_JSON record")
    return json.loads(rows[0])


def mount_identity(path):
    result = subprocess.run(
        ["findmnt", "-T", str(path), "-n", "-o", "SOURCE,FSTYPE"],
        text=True, capture_output=True, check=True,
    ).stdout.split()
    if len(result) != 2:
        fail(f"cannot identify mount for {path}")
    return tuple(result)


def shard_header(path):
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            fail(f"{path.name}: truncated safetensors prefix")
        header_len = int.from_bytes(prefix, "little")
        if header_len > 512 * 1024 * 1024:
            fail(f"{path.name}: unreasonable safetensors header")
        header = source.read(header_len)
    if len(header) != header_len:
        fail(f"{path.name}: truncated safetensors header")
    try:
        return json.loads(header)
    except json.JSONDecodeError as error:
        fail(f"{path.name}: invalid safetensors header: {error}")


def preflight(model_dir, device):
    model_dir = model_dir.resolve()
    if str(model_dir).startswith(("/mnt/c/", "/mnt/d/")):
        fail("DrvFS model paths are forbidden")
    if model_dir.name != f"deepseek-v4-pro-{REVISION}":
        fail("model directory name does not pin the required revision")
    source, fstype = mount_identity(model_dir)
    if source != device or fstype != "ext4":
        fail(f"model must be on {device} ext4, got {source} {fstype}")

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file() or index_path.is_symlink():
        fail("missing regular model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        fail("index weight_map is missing or empty")
    if int(index.get("metadata", {}).get("total_size", -1)) <= 0:
        fail("index metadata.total_size is missing or invalid")

    expected = {f"model-{number:05d}-of-00064.safetensors" for number in range(1, 65)}
    if set(weight_map.values()) != expected:
        fail("index does not reference exactly the pinned 64 shards")
    total = 0
    tensors_by_shard = {name: set() for name in expected}
    for tensor, shard in weight_map.items():
        tensors_by_shard[shard].add(tensor)
    for name in sorted(expected):
        path = model_dir / name
        mode = os.lstat(path).st_mode if path.exists() or path.is_symlink() else 0
        if path.is_symlink() or not stat.S_ISREG(mode):
            fail(f"{name}: missing regular shard")
        total += path.stat().st_size
        header = shard_header(path)
        tensors = set(header) - {"__metadata__"}
        if tensors != tensors_by_shard[name]:
            fail(f"{name}: header tensors disagree with the pinned index")
    if total != SHARD_BYTES:
        fail(f"shard bytes {total} != pinned {SHARD_BYTES}")
    return model_dir, source, fstype, total


def run_iobench(binary, shard, block_mib, reads, threads, mode):
    direct = "1" if mode == "direct_uncached" else "0"
    completed = subprocess.run(
        [str(binary), str(shard), str(block_mib), str(reads), str(threads), direct],
        text=True, capture_output=True,
    )
    if completed.returncode:
        fail(completed.stderr.strip() or f"iobench failed with {completed.returncode}")
    result = parse_result(completed.stdout)
    if result["requested_mode"] != mode or result["actual_mode"] != mode:
        fail(f"requested {mode}, iobench used {result['actual_mode']}")
    expected_bytes = block_mib * 1024 * 1024 * reads
    if result["bytes"] != expected_bytes:
        fail(f"iobench byte count {result['bytes']} != {expected_bytes}")
    if result.get("pread_return_wall_definition") != (
        "before_parallel_read_loop_to_last_pread_return"
    ) or result.get("intermediate_digest_in_pread_return_wall") is not True or result.get(
        "final_aggregate_digest_in_pread_return_wall"
    ) is not False:
        fail("iobench must distinguish pread-return wall from final digest aggregation")
    if result.get("verified_elapsed_definition") != (
        "before_parallel_read_loop_to_final_aggregate_digest"
    ) or result.get("verified_pipeline_gbps_definition") != "bytes / verified_elapsed_s":
        fail("iobench must report the full verified-pipeline throughput")
    if not isinstance(result.get("verified_pipeline_gbps"), (int, float)) or result[
        "verified_pipeline_gbps"
    ] <= 0:
        fail("iobench reported no usable verified-pipeline throughput")
    return result


def selftest():
    line = ('noise\nIOBENCH_JSON {"requested_mode":"cached","actual_mode":"cached",'
            '"bytes":8,"pread_return_wall_definition":"before_parallel_read_loop_to_last_pread_return",'
            '"intermediate_digest_in_pread_return_wall":true,'
            '"final_aggregate_digest_in_pread_return_wall":false,'
            '"verified_elapsed_definition":"before_parallel_read_loop_to_final_aggregate_digest",'
            '"verified_pipeline_gbps_definition":"bytes / verified_elapsed_s",'
            '"digest_in_read_timing":true,"digest":"01"}\n')
    parsed = parse_result(line)
    assert parsed == {
        "requested_mode": "cached", "actual_mode": "cached", "bytes": 8,
        "pread_return_wall_definition": "before_parallel_read_loop_to_last_pread_return",
        "intermediate_digest_in_pread_return_wall": True,
        "final_aggregate_digest_in_pread_return_wall": False,
        "verified_elapsed_definition": "before_parallel_read_loop_to_final_aggregate_digest",
        "verified_pipeline_gbps_definition": "bytes / verified_elapsed_s",
        "digest_in_read_timing": True, "digest": "01"
    }
    original_mount_identity = mount_identity
    try:
        globals()["mount_identity"] = lambda _: ("/dev/other", "ext4")
        preflight(Path(f"/tmp/deepseek-v4-pro-{REVISION}"), "/dev/expected")
    except SystemExit as error:
        assert str(error) == "model must be on /dev/expected ext4, got /dev/other ext4"
    else:
        raise AssertionError("preflight accepted a mismatched device")
    finally:
        globals()["mount_identity"] = original_mount_identity
    print("PASS: V4 I/O matrix result parser")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parent.parent / "iobench")
    parser.add_argument("--shard", default="model-00002-of-00064.safetensors")
    parser.add_argument("--reads", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.model_dir or not args.device or not args.jsonl or not args.csv:
        parser.error("--model-dir, --device, --jsonl, and --csv are required")
    if args.reads <= 0 or args.repeats <= 0:
        parser.error("--reads and --repeats must be positive")
    if not args.binary.is_file():
        fail(f"missing iobench binary: {args.binary}")
    for output in (args.jsonl, args.csv):
        if output.exists():
            fail(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    model_dir, source, fstype, total = preflight(args.model_dir, args.device)
    shard = model_dir / args.shard
    if shard.parent != model_dir or not shard.is_file() or shard.is_symlink():
        fail("--shard must name one regular file inside --model-dir")

    expected_digests = {}
    with args.jsonl.open("x") as jsonl, args.csv.open("x", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        for block_mib in SIZES_MIB:
            for mode in ("direct_uncached", "cached"):
                for threads in THREADS:
                    if mode == "cached":
                        warmup = run_iobench(
                            args.binary, shard, block_mib, args.reads, threads, mode
                        )
                        expected_digests.setdefault(block_mib, warmup["digest"])
                        if warmup["digest"] != expected_digests[block_mib]:
                            fail(f"{block_mib} MiB cached warmup digest changed")
                    for repeat in range(1, args.repeats + 1):
                        result = run_iobench(
                            args.binary, shard, block_mib, args.reads, threads, mode
                        )
                        expected_digests.setdefault(block_mib, result["digest"])
                        if result["digest"] != expected_digests[block_mib]:
                            fail(f"{block_mib} MiB buffer digest changed")
                        record = {
                            **result,
                            "cell": f"{block_mib}MiB-t{threads}-{mode}",
                            "repeat": repeat,
                            "concurrency": "threads_approx_qd_not_true_io_uring_qd",
                            "revision": REVISION,
                            "device": source,
                            "fstype": fstype,
                            "model_total_bytes": total,
                        }
                        record = {field: record[field] for field in FIELDS}
                        jsonl.write(json.dumps(record, sort_keys=True) + "\n")
                        jsonl.flush()
                        writer.writerow(record)
                        csv_file.flush()
                        print(
                            f"{record['cell']} repeat={repeat} "
                            f"verified_pipeline={record['verified_pipeline_gbps']:.3f}GB/s "
                            f"digest={record['digest']}"
                        )


if __name__ == "__main__":
    main()
