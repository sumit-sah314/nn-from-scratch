"""Numerical gradient checks (centered finite differences, float64).

These prove that each hand-written backward pass is correct independently of PyTorch.
"""
import numpy as np
import pytest

from nnbench.numpy_nn import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential, SoftmaxCrossEntropy

RNG = np.random.default_rng(0)
EPS = 1e-6


def rel_err(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def numeric_grad(f, x):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + EPS
        fp = f()
        x[i] = old - EPS
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * EPS)
    return g


def check_layer(layer, x):
    """Use L = sum(y * R) with random R so dL/dy = R."""
    y = layer.forward(x)
    R = RNG.standard_normal(y.shape)
    dx = layer.backward(R)
    f = lambda: float((layer.forward(x) * R).sum())  # noqa: E731
    assert rel_err(dx, numeric_grad(f, x)) < 1e-6, "dx mismatch"
    for k, p in layer.params.items():
        layer.forward(x)
        layer.backward(R)
        analytic = layer.grads[k].copy()
        assert rel_err(analytic, numeric_grad(f, p)) < 1e-6, f"d{k} mismatch"


def test_dense():
    check_layer(Dense(5, 4, rng=RNG, dtype=np.float64), RNG.standard_normal((3, 5)))


def test_relu():
    x = RNG.standard_normal((4, 6))
    x[np.abs(x) < 1e-3] = 0.5  # avoid the kink at 0
    check_layer(ReLU(), x)


@pytest.mark.parametrize("stride,pad", [(1, 0), (1, 1), (2, 1)])
def test_conv(stride, pad):
    check_layer(Conv2D(2, 3, 3, stride=stride, padding=pad, rng=RNG, dtype=np.float64),
                RNG.standard_normal((2, 2, 6, 6)))


def test_maxpool():
    # distinct values so argmax is unique and the function is locally linear
    x = RNG.permutation(2 * 3 * 4 * 4).reshape(2, 3, 4, 4).astype(np.float64)
    check_layer(MaxPool2D(2), x)


def test_softmax_ce():
    logits = RNG.standard_normal((5, 4))
    y = RNG.integers(0, 4, 5)
    lf = SoftmaxCrossEntropy()
    lf(logits, y)
    analytic = lf.backward()
    assert rel_err(analytic, numeric_grad(lambda: lf(logits, y), logits)) < 1e-6


def test_full_cnn_end_to_end():
    model = Sequential(Conv2D(1, 2, 3, padding=1, rng=RNG, dtype=np.float64), ReLU(), MaxPool2D(2), Flatten(),
                       Dense(2 * 3 * 3, 3, rng=RNG, dtype=np.float64))
    x = RNG.standard_normal((2, 1, 6, 6))
    y = np.array([0, 2])
    lf = SoftmaxCrossEntropy()
    lf(model.forward(x), y)
    model.backward(lf.backward())
    analytic = [g.copy() for _, g in model.parameters()]
    for (p, _), a in zip(model.parameters(), analytic):
        assert rel_err(a, numeric_grad(lambda: lf(model.forward(x), y), p)) < 1e-5
