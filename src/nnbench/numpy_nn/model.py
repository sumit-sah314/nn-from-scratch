"""A minimal Sequential container."""
from __future__ import annotations

import numpy as np

from .layers import Layer


class Sequential:
    def __init__(self, *layers: Layer) -> None:
        self.layers = list(layers)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    __call__ = forward

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(param, grad) pairs; grads are updated in place by backward()."""
        return [(l.params[k], l.grads[k]) for l in self.layers for k in l.params]

    def num_params(self) -> int:
        return sum(p.size for p, _ in self.parameters())

    def flops(self, in_shape: tuple[int, ...]) -> list[dict]:
        """Per-layer forward FLOPs for one sample plus output shapes."""
        rows, shape = [], tuple(in_shape)
        for layer in self.layers:
            f, shape = layer.flops(shape)
            rows.append({"layer": repr(layer), "fwd_flops": int(f), "out_shape": shape,
                         "params": sum(p.size for p in layer.params.values())})
        return rows

    def state_dict(self) -> dict[str, np.ndarray]:
        return {f"{i}.{k}": v.copy() for i, l in enumerate(self.layers) for k, v in l.params.items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for i, l in enumerate(self.layers):
            for k in l.params:
                l.params[k][...] = state[f"{i}.{k}"]

    def save(self, path: str) -> None:
        np.savez(path, **self.state_dict())

    def load(self, path: str) -> None:
        with np.load(path) as f:
            self.load_state_dict(dict(f))

    def __repr__(self) -> str:
        inner = "\n".join(f"  ({i}) {l!r}" for i, l in enumerate(self.layers))
        return f"Sequential(\n{inner}\n)"
