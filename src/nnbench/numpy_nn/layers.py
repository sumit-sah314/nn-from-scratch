"""Neural-network layers implemented from scratch with NumPy.

Conventions (chosen to match PyTorch so weights can be copied 1:1):
  * Images are NCHW: (batch, channels, height, width).
  * Dense weights are stored as (in_features, out_features); PyTorch stores the
    transpose, so `copy_numpy_to_torch` transposes them.
  * Conv weights are (out_channels, in_channels, kh, kw), identical to PyTorch.

Every layer exposes:
  forward(x, training=True) -> y      caches whatever backward needs
  backward(dout) -> dx                fills self.grads in place, returns dL/dx
  params / grads                      dicts of np.ndarray (same keys)
  flops(in_shape) -> (fwd_flops, out_shape)   per-sample, analytic
"""
from __future__ import annotations

import numpy as np


class Layer:
    """Base class. Parameter-free layers simply leave params/grads empty."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def flops(self, in_shape: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        """Forward FLOPs for ONE sample (multiply and add counted separately)."""
        return 0, in_shape

    def _init_grads(self) -> None:
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# --------------------------------------------------------------------------- #
# Fully connected
# --------------------------------------------------------------------------- #
class Dense(Layer):
    """y = x @ W + b. He (Kaiming) normal init, which suits ReLU networks."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None,
                 dtype=np.float32) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        std = np.sqrt(2.0 / in_features)
        self.params["W"] = (rng.standard_normal((in_features, out_features)) * std).astype(dtype)
        self.params["b"] = np.zeros(out_features, dtype=dtype)
        self._init_grads()
        self.in_features, self.out_features = in_features, out_features

    def forward(self, x, training=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"][...] = self.x.T @ dout
        self.grads["b"][...] = dout.sum(axis=0)
        return dout @ self.params["W"].T

    def flops(self, in_shape):
        # in*out multiplies + in*out adds (incl. bias add)
        return 2 * self.in_features * self.out_features, (self.out_features,)

    def __repr__(self):
        return f"Dense({self.in_features}, {self.out_features})"


class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

    def flops(self, in_shape):
        return int(np.prod(in_shape)), in_shape


class Flatten(Layer):
    def forward(self, x, training=True):
        self.in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.in_shape)

    def flops(self, in_shape):
        return 0, (int(np.prod(in_shape)),)


# --------------------------------------------------------------------------- #
# im2col / col2im helpers
# --------------------------------------------------------------------------- #
def _out_size(size: int, k: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - k) // stride + 1


def im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """Unfold image patches.

    x: (N, C, H, W) -> cols: (N, C, kh, kw, out_h, out_w)

    Instead of looping over output pixels (out_h*out_w iterations) we loop over
    kernel offsets (kh*kw iterations, e.g. 9) and copy a strided slice each time.
    """
    N, C, H, W = x.shape
    oh, ow = _out_size(H, kh, stride, pad), _out_size(W, kw, stride, pad)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    cols = np.empty((N, C, kh, kw, oh, ow), dtype=x.dtype)
    for i in range(kh):
        i_end = i + stride * oh
        for j in range(kw):
            j_end = j + stride * ow
            cols[:, :, i, j] = xp[:, :, i:i_end:stride, j:j_end:stride]
    return cols


def col2im(cols: np.ndarray, x_shape: tuple[int, ...], kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """Inverse of im2col: scatter-ADD patches back (overlaps accumulate)."""
    N, C, H, W = x_shape
    oh, ow = cols.shape[-2:]
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        i_end = i + stride * oh
        for j in range(kw):
            j_end = j + stride * ow
            xp[:, :, i:i_end:stride, j:j_end:stride] += cols[:, :, i, j]
    return xp[:, :, pad:pad + H, pad:pad + W] if pad else xp


# --------------------------------------------------------------------------- #
# Convolution and pooling
# --------------------------------------------------------------------------- #
class Conv2D(Layer):
    """2-D convolution (cross-correlation, like PyTorch) via im2col + one GEMM."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, rng: np.random.Generator | None = None, dtype=np.float32) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        k = kernel_size
        fan_in = in_channels * k * k
        self.params["W"] = (rng.standard_normal((out_channels, in_channels, k, k))
                            * np.sqrt(2.0 / fan_in)).astype(dtype)
        self.params["b"] = np.zeros(out_channels, dtype=dtype)
        self._init_grads()
        self.cin, self.cout, self.k, self.stride, self.pad = in_channels, out_channels, k, stride, padding

    def forward(self, x, training=True):
        N, C, H, W = x.shape
        k, s, p = self.k, self.stride, self.pad
        oh, ow = _out_size(H, k, s, p), _out_size(W, k, s, p)
        cols = im2col(x, k, k, s, p)                                          # (N,C,k,k,oh,ow)
        self.col = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * oh * ow, -1)  # (N*oh*ow, C*k*k)
        self.x_shape = x.shape
        W_r = self.params["W"].reshape(self.cout, -1)                         # (F, C*k*k)
        out = self.col @ W_r.T + self.params["b"]                             # (N*oh*ow, F)
        return out.reshape(N, oh, ow, self.cout).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, F, oh, ow = dout.shape
        C, k = self.cin, self.k
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, F)                    # (N*oh*ow, F)
        self.grads["W"][...] = (dout_r.T @ self.col).reshape(self.params["W"].shape)
        self.grads["b"][...] = dout_r.sum(axis=0)
        dcol = dout_r @ self.params["W"].reshape(F, -1)                       # (N*oh*ow, C*k*k)
        dcol = dcol.reshape(N, oh, ow, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        return col2im(dcol, self.x_shape, k, k, self.stride, self.pad)

    def flops(self, in_shape):
        C, H, W = in_shape
        oh, ow = _out_size(H, self.k, self.stride, self.pad), _out_size(W, self.k, self.stride, self.pad)
        macs = self.cout * oh * ow * C * self.k * self.k
        return 2 * macs, (self.cout, oh, ow)

    def __repr__(self):
        return f"Conv2D({self.cin}, {self.cout}, k={self.k}, s={self.stride}, p={self.pad})"


class MaxPool2D(Layer):
    """Max pooling. Reuses im2col so any kernel/stride combination works."""

    def __init__(self, kernel_size: int = 2, stride: int | None = None) -> None:
        super().__init__()
        self.k = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x, training=True):
        N, C, H, W = x.shape
        k, s = self.k, self.stride
        oh, ow = _out_size(H, k, s, 0), _out_size(W, k, s, 0)
        cols = im2col(x.reshape(N * C, 1, H, W), k, k, s, 0).reshape(N * C, k * k, oh, ow)
        self.argmax = cols.argmax(axis=1)                                     # (N*C, oh, ow)
        self.x_shape = x.shape
        out = np.take_along_axis(cols, self.argmax[:, None], axis=1)[:, 0]
        return out.reshape(N, C, oh, ow)

    def backward(self, dout):
        N, C, H, W = self.x_shape
        k, s = self.k, self.stride
        oh, ow = dout.shape[-2:]
        dcols = np.zeros((N * C, k * k, oh, ow), dtype=dout.dtype)
        np.put_along_axis(dcols, self.argmax[:, None], dout.reshape(N * C, 1, oh, ow), axis=1)
        dx = col2im(dcols.reshape(N * C, 1, k, k, oh, ow), (N * C, 1, H, W), k, k, s, 0)
        return dx.reshape(self.x_shape)

    def flops(self, in_shape):
        C, H, W = in_shape
        oh, ow = _out_size(H, self.k, self.stride, 0), _out_size(W, self.k, self.stride, 0)
        return C * oh * ow * (self.k * self.k - 1), (C, oh, ow)  # comparisons

    def __repr__(self):
        return f"MaxPool2D(k={self.k}, s={self.stride})"
