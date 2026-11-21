"""Timing and memory measurement helpers.

Fair-timing rules applied throughout:
  * warm-up iterations are discarded (JIT/allocator/cache effects),
  * we report the median of several repeats, not the mean,
  * on CUDA we call torch.cuda.synchronize() before reading the clock,
    because GPU kernels launch asynchronously.
"""
from __future__ import annotations

import resource
import statistics
import sys
import time
import tracemalloc
from typing import Callable


def _sync(device: str | None) -> None:
    if device and device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def time_fn(fn: Callable[[], object], repeats: int = 10, warmup: int = 2, device: str | None = None) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(times), "min_s": min(times),
            "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0}


def tracemalloc_peak(fn: Callable[[], object]) -> float:
    """Peak Python-heap bytes allocated during fn (in MB).

    NumPy reports its buffers to tracemalloc, so this captures the from-scratch
    implementation well. PyTorch's C++ allocator is NOT visible here -> use
    `torch_peak_memory` for the PyTorch side.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


def torch_peak_memory(fn: Callable[[], object], device: str) -> float:
    """Peak memory (MB) for a PyTorch step: CUDA allocator stats on GPU,
    CPU allocator profiler on CPU."""
    import torch

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    # Walk events in time order and track the running allocated total.
    events = sorted((e for e in prof.events() if e.cpu_memory_usage), key=lambda e: e.time_range.start)
    cur = peak = 0
    for e in events:
        cur += e.cpu_memory_usage
        peak = max(peak, cur)
    return peak / 2**20


def peak_rss_mb() -> float:
    """Process-lifetime peak resident set size (coarse, but framework-agnostic)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10
