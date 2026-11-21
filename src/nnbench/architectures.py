"""Framework-agnostic architecture specs.

An architecture is a list of tuples. The same spec is turned into a NumPy
`Sequential` and a PyTorch `nn.Sequential`, which guarantees that both sides
of every comparison have exactly the same layers and parameter counts.

    ("flatten",)
    ("dense", in_features, out_features)
    ("relu",)
    ("conv", in_ch, out_ch, kernel, stride, padding)
    ("maxpool", kernel, stride)
"""
from __future__ import annotations

Spec = list[tuple]


def mlp_spec(in_shape: tuple[int, int, int], num_classes: int = 10, hidden: tuple[int, ...] = (256, 128)) -> Spec:
    c, h, w = in_shape
    dims = [c * h * w, *hidden]
    spec: Spec = [("flatten",)]
    for a, b in zip(dims[:-1], dims[1:]):
        spec += [("dense", a, b), ("relu",)]
    spec.append(("dense", dims[-1], num_classes))
    return spec


def cnn_spec(in_shape: tuple[int, int, int], num_classes: int = 10, channels: tuple[int, int] = (16, 32),
             fc: int = 128) -> Spec:
    """LeNet-style: [conv3x3 -> ReLU -> maxpool2] x2 -> FC -> ReLU -> FC."""
    c, h, w = in_shape
    c1, c2 = channels
    flat = c2 * (h // 4) * (w // 4)
    return [
        ("conv", c, c1, 3, 1, 1), ("relu",), ("maxpool", 2, 2),
        ("conv", c1, c2, 3, 1, 1), ("relu",), ("maxpool", 2, 2),
        ("flatten",),
        ("dense", flat, fc), ("relu",),
        ("dense", fc, num_classes),
    ]


ARCHITECTURES = {"mlp": mlp_spec, "cnn": cnn_spec}


def get_spec(arch: str, in_shape: tuple[int, int, int], num_classes: int = 10) -> Spec:
    return ARCHITECTURES[arch](in_shape, num_classes)


def build_numpy(spec: Spec, seed: int = 0, dtype=None):
    import numpy as np

    from .numpy_nn import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential

    dtype = dtype or np.float32
    rng = np.random.default_rng(seed)
    layers = []
    for kind, *a in spec:
        if kind == "flatten":
            layers.append(Flatten())
        elif kind == "relu":
            layers.append(ReLU())
        elif kind == "dense":
            layers.append(Dense(a[0], a[1], rng=rng, dtype=dtype))
        elif kind == "conv":
            layers.append(Conv2D(a[0], a[1], a[2], stride=a[3], padding=a[4], rng=rng, dtype=dtype))
        elif kind == "maxpool":
            layers.append(MaxPool2D(a[0], a[1]))
        else:
            raise ValueError(kind)
    return Sequential(*layers)


def build_torch(spec: Spec, seed: int = 0):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mods = []
    for kind, *a in spec:
        if kind == "flatten":
            mods.append(nn.Flatten())
        elif kind == "relu":
            mods.append(nn.ReLU())
        elif kind == "dense":
            mods.append(nn.Linear(a[0], a[1]))
        elif kind == "conv":
            mods.append(nn.Conv2d(a[0], a[1], a[2], stride=a[3], padding=a[4]))
        elif kind == "maxpool":
            mods.append(nn.MaxPool2d(a[0], a[1]))
        else:
            raise ValueError(kind)
    model = nn.Sequential(*mods)
    # Same He-normal init as the NumPy side (zero bias) so training starts comparably.
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return model


def copy_numpy_to_torch(np_model, torch_model) -> None:
    """Copy weights so both models compute the exact same function (for parity tests)."""
    import torch

    t_layers = [m for m in torch_model if any(True for _ in m.parameters(recurse=False))]
    n_layers = [l for l in np_model.layers if l.params]
    assert len(t_layers) == len(n_layers), "architectures differ"
    with torch.no_grad():
        for t, n in zip(t_layers, n_layers):
            W = n.params["W"]
            W = W.T if W.ndim == 2 else W  # Dense: (in,out) -> torch (out,in)
            t.weight.copy_(torch.from_numpy(W.copy()))
            t.bias.copy_(torch.from_numpy(n.params["b"].copy()))
