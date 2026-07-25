#!/usr/bin/env python3
"""Record one C/CUDA-or-Rust benchmark invocation and laptop GPU telemetry."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


SAMPLES = ("timestamp", "index", "name", "pstate", "power_w", "temperature_c",
           "sm_clock_mhz", "mem_clock_mhz", "vram_used_mib", "gpu_util_pct",
           "mem_util_pct", "sw_power_cap", "hw_thermal_slowdown", "hw_power_brake")
NUMERIC_SAMPLES = {"index", "power_w", "temperature_c", "sm_clock_mhz",
                   "mem_clock_mhz", "vram_used_mib", "gpu_util_pct", "mem_util_pct"}
SMI_FIELDS = ("timestamp", "index", "name", "pstate", "power.draw", "temperature.gpu",
              "clocks.sm", "clocks.mem", "memory.used", "utilization.gpu",
              "utilization.memory", "clocks_throttle_reasons.sw_power_cap",
              "clocks_throttle_reasons.hw_thermal_slowdown",
              "clocks_throttle_reasons.hw_power_brake_slowdown")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *command], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def number(value: str):
    value = value.strip()
    if value in ("", "N/A"):
        return None
    try:
        return int(value)
    except ValueError:
        return float(value)


def smi_sample(line: str) -> dict | None:
    values = [value.strip() for value in line.split(",")]
    if len(values) != len(SAMPLES):
        return None
    sample = dict(zip(SAMPLES, values))
    for key in NUMERIC_SAMPLES:
        sample[key] = number(sample[key])
    return sample


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def relay(stream, name: str, raw, events, lock: threading.Lock) -> None:
    for line in stream:
        raw.write(line)
        raw.flush()
        with lock:
            events.write(json.dumps({"at": now(), "stream": name, "line": line.rstrip("\n")}) + "\n")
            events.flush()
        target = sys.stdout if name == "stdout" else sys.stderr
        target.write(line)
        target.flush()


def telemetry(command: list[str], output, lock: threading.Lock, holder: dict, ready: threading.Event) -> None:
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError as err:
        with lock:
            output.write(json.dumps({"at": now(), "error": str(err)}) + "\n")
        ready.set()
        return
    holder["process"] = proc
    ready.set()
    for line in proc.stdout:
        sample = smi_sample(line)
        with lock:
            output.write(json.dumps({"at": now(), "raw": line.rstrip("\n"), "sample": sample}) + "\n")
            output.flush()


def proc_sample(pid: int) -> dict:
    base = Path("/proc") / str(pid)
    status = {}
    for line in (base / "status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "VmHWM", "VmSwap", "Threads"}:
            status[key] = value.strip()
    tasks = {}
    for task in (base / "task").iterdir():
        try:
            tasks[task.name] = (task / "stat").read_text().strip()
        except OSError:
            pass
    return {
        "at": now(),
        "pid": pid,
        "pid_stat": (base / "stat").read_text().strip(),
        "task_stat": tasks,
        "pid_io": (base / "io").read_text().splitlines(),
        "pid_status": status,
        "cpu_stat": Path("/proc/stat").read_text().splitlines()[0],
        "diskstats": Path("/proc/diskstats").read_text().splitlines(),
    }


def proc_telemetry(pid: int, output, lock: threading.Lock, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            sample = proc_sample(pid)
        except OSError:
            break
        with lock:
            output.write(json.dumps(sample) + "\n")
            output.flush()
        stop.wait(1)


def selftest() -> None:
    line = ("2026/01/01 00:00:00, 0, test, P0, 100.5, 60, 2000, 8000, 1000, "
            "99, 25, Not Active, Not Active, Not Active")
    assert smi_sample(line)["sm_clock_mhz"] == 2000
    assert proc_sample(os.getpid())["pid_status"]["Threads"]
    print("PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--implementation", required=True, help="for example c-cuda or rust")
    parser.add_argument("--model", required=True, help="checkpoint or fixture path/name")
    model_id = parser.add_mutually_exclusive_group(required=True)
    model_id.add_argument("--model-hash", help="pinned checkpoint hash/revision")
    model_id.add_argument("--model-manifest", type=Path, help="local manifest to hash")
    parser.add_argument("--quant", required=True)
    parser.add_argument("--prompt-ids", required=True, help="prompt ID list or file")
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--kv-layout", help="for example mla, mha, or gqa")
    parser.add_argument("--active-experts", type=int, help="routed experts selected per token")
    parser.add_argument("--total-experts", type=int, help="routed experts available in the checkpoint")
    parser.add_argument("--cache-state", required=True, choices=("cold", "warm"))
    parser.add_argument("--phase", required=True, choices=("prefill", "decode", "both"))
    parser.add_argument("--power-state", required=True, help="for example ac-performance")
    parser.add_argument("--engine", type=Path, help="binary to hash")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--nsys", action="store_true", help="capture an Nsight Systems report")
    parser.add_argument("--ncu", action="store_true", help="capture Nsight Compute basic counters")
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if not args.command:
        parser.error("benchmark command is required after --")
    if args.nsys and args.ncu:
        parser.error("choose --nsys or --ncu; profile them in separate runs")
    if (args.active_experts is None) != (args.total_experts is None):
        parser.error("--active-experts and --total-experts must be supplied together")
    if args.active_experts is not None and not 0 < args.active_experts <= args.total_experts:
        parser.error("expert counts must satisfy 0 < active <= total")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.model_manifest and not args.model_manifest.is_file():
        parser.error(f"model manifest is not a file: {args.model_manifest}")
    if args.engine and not args.engine.is_file():
        parser.error(f"engine is not a file: {args.engine}")

    args.out.mkdir(parents=True)
    command = args.command
    profiler = None
    if args.nsys:
        profiler = "nsys"
        command = ["nsys", "profile", "--force-overwrite", "true", "--trace=cuda",
                   "--sample=none", "--output", str(args.out / "nsys"), *command]
    elif args.ncu:
        profiler = "ncu"
        command = ["ncu", "--force-overwrite", "--set", "basic", "--export",
                   str(args.out / "ncu"), *command]

    model = {"path": args.model}
    if args.model_hash:
        model["hash"] = args.model_hash
    else:
        model["manifest"] = str(args.model_manifest.resolve())
        model["manifest_sha256"] = digest(args.model_manifest)
    metadata = {
        "schema": 1,
        "started_at": now(),
        "repository": {"commit": git(["rev-parse", "HEAD"]), "dirty": bool(git(["status", "--porcelain"]))},
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "implementation": args.implementation,
        "model": model,
        "engine": None if not args.engine else {"path": str(args.engine.resolve()), "sha256": digest(args.engine)},
        "workload": {"quant": args.quant, "prompt_ids": args.prompt_ids, "batch": args.batch,
                     "context": args.context, "kv_layout": args.kv_layout,
                     "active_experts": args.active_experts, "total_experts": args.total_experts,
                     "cache_state": args.cache_state, "phase": args.phase, "power_state": args.power_state},
        "command": command,
        "profiler": profiler,
        "telemetry": {"enabled": not args.no_telemetry, "gpu": args.gpu, "interval_s": 1,
                      "process_and_system": True},
    }
    write_json(args.out / "run.json", metadata)

    # ponytail: NVIDIA-only because the current laptop target is CUDA; add another collector only for a real target.
    smi = ["nvidia-smi", "--id", args.gpu,
           "--query-gpu=" + ",".join(SMI_FIELDS),
           "--format=csv,noheader,nounits", "--loop=1"]
    lock = threading.Lock()
    with (args.out / "stdout.log").open("w") as stdout, (args.out / "stderr.log").open("w") as stderr, \
         (args.out / "samples.jsonl").open("w") as events, \
         (args.out / "telemetry.jsonl").open("w") as samples, \
         (args.out / "process.jsonl").open("w") as process_samples:
        holder = {}
        ready = threading.Event()
        watcher = None if args.no_telemetry else threading.Thread(target=telemetry, args=(smi, samples, lock, holder, ready), daemon=True)
        if watcher:
            watcher.start()
            ready.wait()
        started = time.monotonic()
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as err:
            watcher_proc = holder.get("process")
            if watcher_proc:
                watcher_proc.terminate()
            if watcher:
                watcher.join(timeout=2)
            metadata.update(finished_at=now(), duration_s=time.monotonic() - started, returncode=127, error=str(err))
            write_json(args.out / "run.json", metadata)
            return 127
        stop = threading.Event()
        process_watcher = threading.Thread(
            target=proc_telemetry, args=(proc.pid, process_samples, lock, stop), daemon=True)
        process_watcher.start()
        threads = [threading.Thread(target=relay, args=(proc.stdout, "stdout", stdout, events, lock)),
                   threading.Thread(target=relay, args=(proc.stderr, "stderr", stderr, events, lock))]
        for thread in threads:
            thread.start()
        code = proc.wait()
        stop.set()
        process_watcher.join(timeout=2)
        for thread in threads:
            thread.join()
        watcher_proc = holder.get("process")
        if watcher_proc:
            watcher_proc.terminate()
        if watcher:
            watcher.join(timeout=2)
        metadata.update(finished_at=now(), duration_s=time.monotonic() - started, returncode=code)
    write_json(args.out / "run.json", metadata)
    return code


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        raise SystemExit(main())
