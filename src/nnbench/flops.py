"""Analytic FLOP / parameter / memory accounting.

Convention: 1 multiply-accumulate (MAC) = 2 FLOPs. Tools like ptflops report
MACs, so divide our numbers by 2 when cross-checking.

Backward pass: for Dense/Conv layers, computing dL/dW and dL/dx are each a GEMM
the same size as the forward GEMM, so backward ~= 2 x forward. ReLU / pooling
backward ~= 1 x forward. We use these standard estimates (see e.g. Kaplan et al.
2020, "training FLOPs ~= 3 x forward FLOPs").
"""
from __future__ import annotations

import numpy as np

from .architectures import Spec, build_numpy


def flop_table(spec: Spec, in_shape: tuple[int, int, int]) -> list[dict]:
    model = build_numpy(spec)
    rows = model.flops(in_shape)
    for r in rows:
        heavy = r["layer"].startswith(("Dense", "Conv2D"))
        r["bwd_flops"] = 2 * r["fwd_flops"] if heavy else r["fwd_flops"]
    return rows


def summarize(spec: Spec, in_shape: tuple[int, int, int], batch_size: int = 1, bytes_per_el: int = 4) -> dict:
    rows = flop_table(spec, in_shape)
    params = sum(r["params"] for r in rows)
    fwd = sum(r["fwd_flops"] for r in rows)
    bwd = sum(r["bwd_flops"] for r in rows)
    act = int(np.prod(in_shape)) + sum(int(np.prod(r["out_shape"])) for r in rows)
    return {
        "params": params,
        "fwd_flops_per_sample": fwd,
        "bwd_flops_per_sample": bwd,
        "train_flops_per_sample": fwd + bwd,
        "fwd_macs_per_sample": fwd // 2,
        # memory a training step must hold: weights + grads + activations for the batch
        "param_bytes": params * bytes_per_el,
        "activation_bytes_per_batch": act * batch_size * bytes_per_el,
    }


def torch_tool_flops(model, in_shape: tuple[int, int, int]) -> dict:
    """Cross-check with third-party counters if installed (both report MACs)."""
    import torch

    out: dict = {}
    try:
        from ptflops import get_model_complexity_info

        macs, _ = get_model_complexity_info(model, in_shape, as_strings=False, print_per_layer_stat=False,
                                            verbose=False)
        out["ptflops_macs"] = int(macs)
    except Exception as e:  # noqa: BLE001
        out["ptflops_macs"] = f"unavailable ({type(e).__name__})"
    try:
        from fvcore.nn import FlopCountAnalysis

        fca = FlopCountAnalysis(model, torch.zeros(1, *in_shape))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        out["fvcore_macs"] = int(fca.total())  # fvcore counts one fused multiply-add as 1 "flop"
    except Exception as e:  # noqa: BLE001
        out["fvcore_macs"] = f"unavailable ({type(e).__name__})"
    return out
