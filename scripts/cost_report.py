#!/usr/bin/env python
"""Turn measured training times into dollar estimates.

  python scripts/cost_report.py                       # all runs in results/runs
  python scripts/cost_report.py --scale-epochs 50     # extrapolate to a 50-epoch run

Cost = (measured seconds / 3600) x hourly rate from configs/pricing.json.
CPU runs are priced on CPU instances, CUDA runs on GPU instances.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

from nnbench.cost import cost_table, load_pricing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--pricing", default="configs/pricing.json")
    p.add_argument("--scale-epochs", type=int, default=None,
                   help="extrapolate using the mean measured epoch time")
    p.add_argument("--out", default="results/cost.csv")
    a = p.parse_args()

    pricing = load_pricing(a.pricing)
    runs = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*.json"))):
        with open(f) as fh:
            h = json.load(fh)
        secs = h["train_seconds"]
        if a.scale_epochs:
            secs = secs / len(h["epochs"]) * a.scale_epochs
        runs.append({**h["config"], "train_seconds": secs})

    rows = cost_table(runs, pricing)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)

    print(f"prices checked {pricing.get('checked')}  (edit {a.pricing} to update)\n")
    print(f"{'run':45s} {'instance':26s} {'hours':>8s} {'USD':>10s}")
    for r in rows:
        name = f"{r['framework']}-{r['device']} {r['arch']} {r['dataset']} {r['optimizer']}"
        print(f"{name:45s} {r['instance']:26s} {r['train_seconds'] / 3600:8.4f} {r['usd']:10.5f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
