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
import numpy as np

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


def group_by_config(runs: list[dict]) -> dict:
    """(arch, dataset, framework-device, optimizer) -> list of runs, one per seed."""
    groups = defaultdict(list)
    for r in runs:
        c = r["config"]
        groups[(c["arch"], c["dataset"], label(c))].append(r)
    return groups


def mean_std(vals) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def plot_convergence(runs: list[dict], out: str) -> None:
    """One figure per (arch, dataset). Lines are the mean over seeds; bands are +/- 1 std."""
    by_arch_ds = defaultdict(dict)
    for (arch, ds, lbl), rs in group_by_config(runs).items():
        by_arch_ds[(arch, ds)][lbl] = rs
    for (arch, ds), series in by_arch_ds.items():
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        for lbl, rs in series.items():
            ls = "-" if rs[0]["config"]["framework"] == "numpy" else "--"
            lbl_n = f"{lbl} (n={len(rs)})"
            # per-step loss: same logging schedule for every seed, so align by index
            n_steps = min(len(r["step_loss"]) for r in rs)
            steps = np.array([s for s, _ in rs[0]["step_loss"][:n_steps]])
            losses = np.array([[l for _, l in r["step_loss"][:n_steps]] for r in rs])
            ax[0].plot(steps, losses.mean(0), ls, label=lbl_n, alpha=0.8)
            ep = [e["epoch"] for e in rs[0]["epochs"]]
            tl = np.array([[e["test_loss"] for e in r["epochs"]] for r in rs])
            ta = 100 * np.array([[e["test_acc"] for e in r["epochs"]] for r in rs])
            for axis, y in ((ax[1], tl), (ax[2], ta)):
                axis.plot(ep, y.mean(0), ls + "o", label=lbl_n, ms=4)
                if len(rs) > 1:
                    axis.fill_between(ep, y.mean(0) - y.std(0, ddof=1), y.mean(0) + y.std(0, ddof=1), alpha=0.15)
        ax[0].set(title="training loss (per step, mean over seeds)", xlabel="step", ylabel="cross-entropy",
                  yscale="log")
        ax[1].set(title="test loss (mean ± std)", xlabel="epoch")
        ax[2].set(title="test accuracy % (mean ± std)", xlabel="epoch")
        for a_ in ax:
            a_.grid(alpha=0.3)
        ax[2].legend(fontsize=8)
        fig.suptitle(f"{arch.upper()} on {ds}  (solid = NumPy, dashed = PyTorch)")
        fig.tight_layout()
        path = os.path.join(out, f"convergence_{arch}_{ds}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print("wrote", path)


def write_summary(runs: list[dict], out_md: str) -> None:
    """Markdown table: final-epoch test accuracy/loss and train time, mean ± std over seeds."""
    lines = ["| dataset | arch | impl / optimizer | seeds | final test acc (%) | final test loss | train time (s) |",
             "|---|---|---|---|---|---|---|"]
    for (arch, ds, lbl), rs in sorted(group_by_config(runs).items()):
        acc_m, acc_s = mean_std([100 * r["epochs"][-1]["test_acc"] for r in rs])
        loss_m, loss_s = mean_std([r["epochs"][-1]["test_loss"] for r in rs])
        t_m, t_s = mean_std([r["train_seconds"] for r in rs])
        lines.append(f"| {ds} | {arch} | {lbl} | {len(rs)} | {acc_m:.2f} ± {acc_s:.2f} | "
                     f"{loss_m:.4f} ± {loss_s:.4f} | {t_m:.1f} ± {t_s:.1f} |")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_md)


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
    runs = load_runs(a.runs)
    if runs:
        plot_convergence(runs, a.out)
        write_summary(runs, os.path.join(os.path.dirname(a.out.rstrip("/")) or ".", "summary.md"))
    plot_benchmark(a.benchmark, a.out)


if __name__ == "__main__":
    main()
