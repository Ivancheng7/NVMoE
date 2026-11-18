#!/usr/bin/env python3
"""从魔搭下载并校验 DeepSeek-V4-Flash-0731 权重。

下载在 WSL 原生 ext4 目录中完成，使用逐文件临时名和同文件系统改名。
脚本不会删除旧模型；调用者可在新模型校验完成后再清理旧目录。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.parse
import urllib.request


MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
UPSTREAM_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
MODELSCOPE_REVISION = "master"
API = (
    "https://www.modelscope.cn/api/v1/models/"
    f"{MODEL}/repo/files?Revision={MODELSCOPE_REVISION}&Recursive=true"
)
RESOLVE = f"https://www.modelscope.cn/api/v1/models/{MODEL}/resolve/{MODELSCOPE_REVISION}/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_one(file_record: dict[str, object], partial: Path) -> int:
    relative = str(file_record["Path"])
    expected_size = int(file_record.get("Size") or 0)
    expected_sha256 = str(file_record.get("Sha256") or "")
    output = partial / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")

    host_free = shutil.disk_usage("/mnt/c").free
    if host_free < 80 * 1024**3:
        raise RuntimeError(f"宿主 C 盘余量低于 80 GiB: {host_free} bytes")

    if output.exists() and output.stat().st_size == expected_size:
        pass
    else:
        url = RESOLVE + urllib.parse.quote(relative, safe="/")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--retry",
                "8",
                "--retry-delay",
                "5",
                "--continue-at",
                "-",
                "--output",
                str(temporary),
                url,
            ],
            check=True,
        )
        actual_size = temporary.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"{relative}: 大小 {actual_size} != {expected_size}")
        os.replace(temporary, output)

    if output.stat().st_size != expected_size:
        raise RuntimeError(f"{relative}: 完成后大小不匹配")
    if expected_sha256:
        actual_sha256 = sha256_file(output)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"{relative}: SHA256 {actual_sha256} != {expected_sha256}")
    print(f"校验通过 {relative} {expected_size} B", flush=True)
    return expected_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/srv/localmoe/models"))
    parser.add_argument(
        "--directory",
        default=f"deepseek-v4-flash-0731-{UPSTREAM_REVISION}",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.root.is_symlink():
        raise SystemExit(f"拒绝符号链接下载根目录: {args.root}")
    root = args.root.resolve()
    if str(root) == "/mnt" or str(root).startswith("/mnt/"):
        raise SystemExit(f"拒绝 DrvFS/9p 下载路径: {root}")
    filesystem = subprocess.check_output(
        ["stat", "-f", "-c", "%T", str(root)], text=True
    ).strip()
    if filesystem not in {"ext2/ext3", "ext4"}:
        raise SystemExit(f"下载根目录必须位于 WSL 原生 ext4 块设备: {root} ({filesystem})")
    partial = root / f".{args.directory}.partial"
    final = root / args.directory
    if final.exists():
        raise SystemExit(f"目标目录已存在，不覆盖: {final}")
    partial.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(API, headers={"User-Agent": "localmoe-model-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        records = json.load(response)["Data"]["Files"]
    records = sorted((record for record in records if record.get("Type") == "blob"), key=lambda x: x["Path"])
    weight_bytes = sum(int(record.get("Size") or 0) for record in records if str(record["Path"]).endswith(".safetensors"))
    print(f"文件数={len(records)} 权重字节数={weight_bytes}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch_one, record, partial) for record in records]
        total = sum(future.result() for future in concurrent.futures.as_completed(futures))
    print(f"全部文件校验通过，总字节数={total}", flush=True)

    provenance = {
        "source": "modelscope.cn",
        "model": MODEL,
        "modelscope_revision": MODELSCOPE_REVISION,
        "modelscope_commit": "f981a343464c25f82b901e5882716b3b2fa514de",
        "upstream_revision": UPSTREAM_REVISION,
        "files": len(records),
        "weight_bytes": weight_bytes,
    }
    (partial / ".modelscope-source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    os.replace(partial, final)
    print(f"已发布 {final}", flush=True)


if __name__ == "__main__":
    main()
