#!/usr/bin/env python
"""Train one (framework, arch, dataset, optimizer) configuration and save its history.

Examples
  python scripts/train.py --framework numpy --arch mlp --dataset mnist --optimizer adam --epochs 5
  python scripts/train.py --framework torch --arch cnn --dataset cifar10 --optimizer momentum --lr 0.01
  python scripts/train.py --framework numpy --arch cnn --dataset mnist --synthetic --epochs 1   # offline smoke test
"""
from __future__ import annotations

import argparse
import json
import os

from nnbench.data import load
from nnbench.training import train_numpy, train_torch

DEFAULT_LR = {"sgd": 0.05, "momentum": 0.01, "rmsprop": 1e-3, "adam": 1e-3}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework", choices=["numpy", "torch"], required=True)
    p.add_argument("--arch", choices=["mlp", "cnn"], default="mlp")
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--optimizer", choices=list(DEFAULT_LR), default="adam")
    p.add_argument("--lr", type=float, default=None, help="default depends on optimizer")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="torch only: cpu / cuda (default: auto)")
    p.add_argument("--subset", type=int, default=None, help="use only the first N training samples")
    p.add_argument("--synthetic", action="store_true", help="random learnable data, no download")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results/runs")
    a = p.parse_args()

    lr = a.lr if a.lr is not None else DEFAULT_LR[a.optimizer]
    ds = load(a.dataset, a.data_root, subset=a.subset, synthetic_data=a.synthetic)
    common = dict(arch=a.arch, optimizer=a.optimizer, lr=lr, epochs=a.epochs, batch_size=a.batch_size,
                  seed=a.seed, weight_decay=a.weight_decay)
    hist = train_numpy(ds, **common) if a.framework == "numpy" else train_torch(ds, device=a.device, **common)
    model = hist.pop("model")

    os.makedirs(a.out, exist_ok=True)
    dev = hist["config"]["device"].split(":")[0]
    stem = os.path.join(a.out, f"{a.framework}-{dev}_{a.arch}_{ds.name}_{a.optimizer}_s{a.seed}")
    path = stem + ".json"
    with open(path, "w") as f:
        json.dump(hist, f, indent=1)
    print(f"saved {path}  (train time {hist['train_seconds']:.1f}s)")

    if a.framework == "numpy":
        weights = stem + ".npz"
        model.save(weights)
    else:
        import torch
        weights = stem + ".pt"
        torch.save(model.state_dict(), weights)
    print(f"saved weights {weights}")


if __name__ == "__main__":
    main()
