#!/usr/bin/env python
"""Micro-benchmarks: params, FLOPs, train-step / inference time, peak memory.

For every (arch, dataset) it measures:
  * NumPy (CPU)          - our from-scratch implementation
  * PyTorch (CPU)        - same architecture, same thread count
  * PyTorch (CUDA)       - if a GPU is available
and writes results/benchmark.csv plus a Markdown table.

Usage
  python scripts/benchmark.py                      # all archs x datasets
  python scripts/benchmark.py --threads 1          # single-threaded BLAS for a fair op-level comparison
"""
from __future__ import annotations

import argparse
import csv
import os


def set_threads(n: int | None) -> None:
    """Must run before NumPy/PyTorch are imported to affect BLAS pools."""
    if n:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archs", nargs="+", default=["mlp", "cnn"])
    p.add_argument("--datasets", nargs="+", default=["mnist", "cifar10"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--infer-batch-size", type=int, default=256)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    set_threads(a.threads)

    import numpy as np
    import torch
    import torch.nn.functional as F

    from nnbench.architectures import build_numpy, build_torch, get_spec
    from nnbench.flops import summarize, torch_tool_flops
    from nnbench.numpy_nn import SoftmaxCrossEntropy, make_optimizer
    from nnbench.profiling import time_fn, torch_peak_memory, tracemalloc_peak
    from nnbench.torch_nn import make_optimizer as torch_opt

    if a.threads:
        torch.set_num_threads(a.threads)
    shapes = {"mnist": (1, 28, 28), "cifar10": (3, 32, 32)}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() and not a.no_cuda else [])
    rng = np.random.default_rng(0)
    rows = []

    for ds in a.datasets:
        shape = shapes[ds]
        x = rng.standard_normal((a.batch_size, *shape)).astype(np.float32)
        y = rng.integers(0, 10, a.batch_size)
        xi = rng.standard_normal((a.infer_batch_size, *shape)).astype(np.float32)
        for arch in a.archs:
            spec = get_spec(arch, shape)
            fl = summarize(spec, shape, batch_size=a.batch_size)
            base = {"arch": arch, "dataset": ds, "params": fl["params"],
                    "fwd_mflops_per_sample": fl["fwd_flops_per_sample"] / 1e6,
                    "train_mflops_per_sample": fl["train_flops_per_sample"] / 1e6,
                    "analytic_mem_mb": (3 * fl["param_bytes"] + fl["activation_bytes_per_batch"]) / 2**20}

            # ---- NumPy --------------------------------------------------- #
            m = build_numpy(spec)
            opt = make_optimizer("adam", m.parameters(), lr=1e-3)
            loss_fn = SoftmaxCrossEntropy()

            def np_step():
                loss_fn(m.forward(x), y)
                m.backward(loss_fn.backward())
                opt.step()

            step = time_fn(np_step, repeats=a.repeats)
            inf = time_fn(lambda: m.forward(xi, training=False), repeats=a.repeats)
            rows.append({**base, "impl": "numpy", "device": "cpu",
                         "train_step_ms": step["median_s"] * 1e3,
                         "train_samples_per_s": a.batch_size / step["median_s"],
                         "infer_ms_per_batch": inf["median_s"] * 1e3,
                         "infer_us_per_sample": inf["median_s"] / a.infer_batch_size * 1e6,
                         "achieved_gflops": fl["train_flops_per_sample"] * a.batch_size / step["median_s"] / 1e9,
                         "peak_mem_mb": tracemalloc_peak(np_step)})
            print(rows[-1])

            # ---- PyTorch ------------------------------------------------- #
            for dev in devices:
                tm = build_torch(spec).to(dev)
                topt = torch_opt("adam", tm.parameters(), lr=1e-3)
                xt, yt = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
                xit = torch.from_numpy(xi).to(dev)

                def t_step():
                    topt.zero_grad(set_to_none=True)
                    F.cross_entropy(tm(xt), yt).backward()
                    topt.step()

                def t_inf():
                    with torch.no_grad():
                        tm(xit)

                step = time_fn(t_step, repeats=a.repeats, device=dev)
                inf = time_fn(t_inf, repeats=a.repeats, device=dev)
                row = {**base, "impl": "torch", "device": dev,
                       "train_step_ms": step["median_s"] * 1e3,
                       "train_samples_per_s": a.batch_size / step["median_s"],
                       "infer_ms_per_batch": inf["median_s"] * 1e3,
                       "infer_us_per_sample": inf["median_s"] / a.infer_batch_size * 1e6,
                       "achieved_gflops": fl["train_flops_per_sample"] * a.batch_size / step["median_s"] / 1e9,
                       "peak_mem_mb": torch_peak_memory(t_step, dev)}
                if dev == "cpu":
                    row.update(torch_tool_flops(tm, shape))
                rows.append(row)
                print(rows[-1])

    os.makedirs(a.out, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.out, "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    cols = ["arch", "dataset", "impl", "device", "params", "fwd_mflops_per_sample", "train_step_ms",
            "infer_us_per_sample", "achieved_gflops", "peak_mem_mb"]
    with open(os.path.join(a.out, "benchmark.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(f"{r[c]:.3g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n")
    print(f"wrote {a.out}/benchmark.csv and benchmark.md")


if __name__ == "__main__":
    main()
