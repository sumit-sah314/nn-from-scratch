"""Cloud cost estimation: dollars = measured wall-clock hours x hourly rate.

Rates live in configs/pricing.json so they can be updated without touching code.
Cloud prices change often and vary by region; always record the date you
checked them alongside your results.
"""
from __future__ import annotations

import json
import os

DEFAULT_PRICING = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "pricing.json")


def load_pricing(path: str | None = None) -> dict:
    with open(path or DEFAULT_PRICING) as f:
        return json.load(f)


def estimate(train_seconds: float, usd_per_hour: float) -> float:
    return train_seconds / 3600.0 * usd_per_hour


def cost_table(runs: list[dict], pricing: dict) -> list[dict]:
    """runs: dicts with keys framework, arch, dataset, device ('cpu'/'cuda'), train_seconds.

    Each run is priced against every instance of the matching device class, so a
    CPU run is priced on CPU instances and a GPU run on GPU instances.
    """
    rows = []
    for r in runs:
        kind = "gpu" if str(r["device"]).startswith("cuda") else "cpu"
        for name, inst in pricing["instances"].items():
            if inst["kind"] != kind:
                continue
            rows.append({**{k: r[k] for k in ("framework", "arch", "dataset", "optimizer", "device")},
                         "instance": name, "usd_per_hour": inst["usd_per_hour"],
                         "train_seconds": round(r["train_seconds"], 2),
                         "usd": estimate(r["train_seconds"], inst["usd_per_hour"])})
    return rows
