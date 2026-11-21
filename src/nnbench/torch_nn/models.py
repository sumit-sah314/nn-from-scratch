"""PyTorch side of the comparison."""
from __future__ import annotations

import torch

from ..architectures import build_torch, get_spec


def build_model(arch: str, in_shape: tuple[int, int, int], num_classes: int = 10, seed: int = 0) -> torch.nn.Module:
    return build_torch(get_spec(arch, in_shape, num_classes), seed=seed)


def make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0, momentum: float = 0.9):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-8, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")
