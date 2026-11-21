#!/usr/bin/env python
"""Plot convergence curves and benchmark bars from results/.

  python scripts/plot_results.py            # reads results/runs/*.json and results/benchmark.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_runs(path: str) -> list[dict]:
    runs = []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def label(cfg: dict) -> str:
    dev = cfg["device"].split(":")[0]
    return f"{cfg['framework']}-{dev} / {cfg['optimizer']}"


def plot_convergence(runs: list[dict], out: str) -> None:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["config"]["arch"], r["config"]["dataset"])].append(r)
    for (arch, ds), rs in groups.items():
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        for r in rs:
            ep = [e["epoch"] for e in r["epochs"]]
            lbl = label(r["config"])
            ls = "-" if r["config"]["framework"] == "numpy" else "--"
            steps, losses = zip(*r["step_loss"]) if r["step_loss"] else ([], [])
            ax[0].plot(steps, losses, ls, label=lbl, alpha=0.8)
            ax[1].plot(ep, [e["test_loss"] for e in r["epochs"]], ls + "o", label=lbl)
            ax[2].plot(ep, [100 * e["test_acc"] for e in r["epochs"]], ls + "o", label=lbl)
        ax[0].set(title="training loss (per step)", xlabel="step", ylabel="cross-entropy", yscale="log")
        ax[1].set(title="test loss", xlabel="epoch")
        ax[2].set(title="test accuracy (%)", xlabel="epoch")
        for a_ in ax:
            a_.grid(alpha=0.3)
        ax[2].legend(fontsize=8)
        fig.suptitle(f"{arch.upper()} on {ds}  (solid = NumPy, dashed = PyTorch)")
        fig.tight_layout()
        path = os.path.join(out, f"convergence_{arch}_{ds}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print("wrote", path)


def plot_benchmark(csv_path: str, out: str) -> None:
    if not os.path.exists(csv_path):
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    metrics = [("train_step_ms", "train step (ms, batch)", True), ("infer_us_per_sample", "inference (µs / sample)", True),
               ("peak_mem_mb", "peak memory (MB)", False), ("achieved_gflops", "achieved GFLOP/s", True)]
    configs = list(dict.fromkeys((r["arch"], r["dataset"]) for r in rows))
    impls = list(dict.fromkeys(f"{r['impl']}-{r['device']}" for r in rows))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    w = 0.8 / len(impls)
    for ax, (key, title, logy) in zip(axes, metrics):
        for k, impl in enumerate(impls):
            vals = []
            for c in configs:
                match = [r for r in rows if (r["arch"], r["dataset"]) == c and f"{r['impl']}-{r['device']}" == impl]
                vals.append(float(match[0][key]) if match else 0)
            ax.bar([i + k * w for i in range(len(configs))], vals, w, label=impl)
        ax.set_xticks([i + w * (len(impls) - 1) / 2 for i in range(len(configs))])
        ax.set_xticklabels([f"{a}\n{d}" for a, d in configs])
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.tight_layout()
    path = os.path.join(out, "benchmark.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--benchmark", default="results/benchmark.csv")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plot_convergence(load_runs(a.runs), a.out)
    plot_benchmark(a.benchmark, a.out)


if __name__ == "__main__":
    main()
