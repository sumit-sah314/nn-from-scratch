#!/usr/bin/env python
"""cProfile breakdown of one NumPy training step (forward + backward + optimizer).

Answers "where does the time go?" for the from-scratch CNN: how much is im2col /
col2im memory traffic, how much is BLAS GEMM, and how much is Python overhead.

  python scripts/profile_numpy.py                         # CNN on CIFAR-10 shapes
  python scripts/profile_numpy.py --arch mlp --dataset mnist
Writes results/profile_<arch>_<dataset>.txt (the full pstats report).
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats

import numpy as np

from nnbench.architectures import build_numpy, get_spec
from nnbench.numpy_nn import SoftmaxCrossEntropy, make_optimizer

SHAPES = {"mnist": (1, 28, 28), "cifar10": (3, 32, 32)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["mlp", "cnn"], default="cnn")
    p.add_argument("--dataset", choices=list(SHAPES), default="cifar10")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    shape = SHAPES[a.dataset]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((a.batch_size, *shape)).astype(np.float32)
    y = rng.integers(0, 10, a.batch_size)
    model = build_numpy(get_spec(a.arch, shape))
    opt = make_optimizer("adam", model.parameters(), lr=1e-3)
    loss_fn = SoftmaxCrossEntropy()

    def step():
        loss_fn(model.forward(x), y)
        model.backward(loss_fn.backward())
        opt.step()

    step()  # warm-up outside the profile
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(a.steps):
        step()
    prof.disable()

    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).sort_stats("tottime")
    stats.print_stats(25)
    stats.sort_stats("cumulative").print_stats(25)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"profile_{a.arch}_{a.dataset}.txt")
    with open(path, "w") as f:
        f.write(f"# {a.steps} training steps, batch {a.batch_size}, {a.arch} on {a.dataset} shapes\n")
        f.write(buf.getvalue())
    print(buf.getvalue()[:4000])
    print("wrote", path)


if __name__ == "__main__":
    main()
