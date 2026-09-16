#!/usr/bin/env python3
"""Minimal client for the Flash-0731 stdio service protocol.

Talks to `colibri-cuda` started with `COLIBRI_V4_FLASH_SERVE=1`: the engine
reads SUBMIT commands on stdin and streams DATA/DONE frames on stdout.  Sends
one chat request holding real prompt text and prints the decoded text plus the
engine's DONE statistics.

  flash_serve_client.py --binary <path/to/colibri-cuda> --model-dir <dir> \
      --prompt "你好" --max-tokens 8
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time


def read_line(stream):
    line = stream.readline()
    if not line:
        raise RuntimeError("engine closed stdout")
    return line.decode("utf-8", "replace").rstrip("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument(
        "--prompt-json",
        default=None,
        help="prompt as a JSON string (use \\uXXXX escapes when the shell cannot carry"
             " non-ASCII bytes); overrides --prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1, help="number of requests")
    parser.add_argument("--cublas-lib", default=None)
    parser.add_argument("--ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--answer-timeout-s", type=float, default=300.0)
    args = parser.parse_args(argv)

    env = dict(os.environ)
    env.update({
        "COLIBRI_V4_FLASH_SERVE": "1",
        "COLIBRI_DSV4_MODEL_DIR": args.model_dir,
        "COLIBRI_V4_STREAM_READ_MODE": "buffered",
        "COLIBRI_V4_FLASH_PREFETCH": "1",
        "COLIBRI_V4_FLASH_MHC_LINEAR_ARM": "cublas",
        "COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM": "cublas",
        "COLIBRI_V4_FLASH_ROUTER_SCORE_ARM": "libdevice",
    })
    if args.cublas_lib:
        env["COLIBRI_CUBLAS_EXACT_LIB"] = args.cublas_lib

    started = time.monotonic()
    process = subprocess.Popen([args.binary], env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, bufsize=0)
    watchdog = threading.Timer(args.ready_timeout_s + args.answer_timeout_s,
                               process.kill)
    watchdog.daemon = True
    watchdog.start()
    stdout = process.stdout
    prompt = args.prompt if args.prompt_json is None else json.loads(args.prompt_json)
    try:
        while True:
            line = read_line(stdout)
            print(f"< {line}", flush=True)
            if "READY" in line:
                break
            if time.monotonic() - started > args.ready_timeout_s:
                raise RuntimeError(f"no READY within {args.ready_timeout_s:.0f}s")

        for turn in range(args.repeat):
            request_id = str(turn + 1)
            payload = json.dumps([{"role": "user", "content": prompt}],
                                 ensure_ascii=False).encode("utf-8")
            header = f"SUBMIT {request_id} 0 {len(payload)} {args.max_tokens} 0.0 1.0\n".encode()
            process.stdin.write(header + payload + b"\n")
            process.stdin.flush()
            print(f"> SUBMIT {request_id} 0 {len(payload)} {args.max_tokens} 0.0 1.0", flush=True)

            decoded = []
            stats = None
            answer_started = time.monotonic()
            while True:
                line = read_line(stdout)
                if line.startswith("DATA "):
                    _, framed_id, length = line.split()
                    chunk = stdout.read(int(length))
                    terminator = stdout.read(1)
                    if terminator != b"\n":
                        raise RuntimeError("DATA frame lacks newline terminator")
                    piece = chunk.decode("utf-8", "replace")
                    decoded.append(piece)
                    print(piece, end="", flush=True)
                elif line.startswith("DONE "):
                    stats = line
                    break
                elif line.startswith("ERROR "):
                    raise RuntimeError(f"engine error: {line}")
                else:
                    print(f"< {line}", flush=True)
                if time.monotonic() - answer_started > args.answer_timeout_s:
                    raise RuntimeError(f"no DONE within {args.answer_timeout_s:.0f}s")

            text = "".join(decoded)
            print()
            print(f"prompt   : {json.dumps(prompt, ensure_ascii=True)}")
            print(f"text     : {json.dumps(text, ensure_ascii=True)}")
            print("stats    :", stats)
            print(f"request  : {time.monotonic() - answer_started:.1f} s")
            print(f"wall     : {time.monotonic() - started:.1f} s")
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
