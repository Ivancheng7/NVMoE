"""Decisive H2D rate test: push the same byte volume the steady state pushes
per token (fixed 6.71 GB + routed 3.44 GB) through pinned->device copies, to
check whether the engine's reported fixed rate (156 MB in 5.44 ms = 28.7 GB/s,
above the Gen4 x8 link) is physically reachable.

usage: python3 h2d_rate6g.py [gigabytes]
"""
import ctypes
import sys
import time

CUDART = "/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12"
H2D = 1

lib = ctypes.CDLL(CUDART)
lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]


def check(rc, what):
    if rc != 0:
        raise SystemExit(f"{what}: cuda error {rc}")


total_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 6.71
chunk = 156 * 1024 * 1024          # the fixed per-layer chunk
reps = int(total_gb * 1024 * 1024 * 1024 / chunk)

stream = ctypes.c_void_p()
check(lib.cudaStreamCreate(ctypes.byref(stream)), "stream")

pinned = ctypes.c_void_p()
check(lib.cudaHostAlloc(ctypes.byref(pinned), chunk, 0), "cudaHostAlloc")
ctypes.memset(pinned, 7, chunk)
dev = ctypes.c_void_p()
check(lib.cudaMalloc(ctypes.byref(dev), chunk), "cudaMalloc")

# warmup
check(lib.cudaMemcpyAsync(dev, pinned, chunk, H2D, stream), "warmup")
check(lib.cudaStreamSynchronize(stream), "warmup sync")

t0 = time.perf_counter()
for _ in range(reps):
    check(lib.cudaMemcpyAsync(dev, pinned, chunk, H2D, stream), "copy")
check(lib.cudaStreamSynchronize(stream), "sync")
elapsed = time.perf_counter() - t0
moved = reps * chunk
print(f"{reps} x {chunk/1048576:.0f} MB pinned->device  {elapsed*1e3:8.1f} ms  "
      f"{moved/1e9/elapsed:6.2f} GB/s   ({elapsed/reps*1e3:.2f} ms per {chunk/1048576:.0f} MB)")
