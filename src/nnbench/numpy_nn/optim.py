"""Optimizers implemented from scratch.

Update rules deliberately follow torch.optim's exact formulas (including where
weight decay and epsilon enter) so the two implementations can be compared
step-for-step; see tests/test_optim_parity.py.

All optimizers take a list of (param, grad) array pairs and update params IN PLACE.
"""
from __future__ import annotations

import numpy as np

ParamList = list[tuple[np.ndarray, np.ndarray]]


class Optimizer:
    def __init__(self, params: ParamList, lr: float, weight_decay: float = 0.0) -> None:
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.t = 0

    def _grad(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        """L2 regularisation as in torch: g <- g + wd * p."""
        return g + self.weight_decay * p if self.weight_decay else g

    def step(self) -> None:
        self.t += 1
        for i, (p, g) in enumerate(self.params):
            self._update(i, p, self._grad(p, g))

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError


class SGD(Optimizer):
    """theta <- theta - lr * g"""

    def _update(self, i, p, g):
        p -= self.lr * g


class Momentum(Optimizer):
    """Heavy-ball momentum (PyTorch convention):
        v <- mu * v + g
        theta <- theta - lr * v            (nesterov: theta -= lr * (g + mu * v))
    """

    def __init__(self, params, lr=0.01, momentum=0.9, nesterov=False, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.mu, self.nesterov = momentum, nesterov
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def _update(self, i, p, g):
        v = self.v[i]
        v *= self.mu
        v += g
        p -= self.lr * (g + self.mu * v if self.nesterov else v)


class RMSProp(Optimizer):
    """s <- alpha * s + (1 - alpha) * g^2
       theta <- theta - lr * g / (sqrt(s) + eps)
    """

    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.alpha, self.eps = alpha, eps
        self.s = [np.zeros_like(p) for p, _ in self.params]

    def _update(self, i, p, g):
        s = self.s[i]
        s *= self.alpha
        s += (1 - self.alpha) * g * g
        p -= self.lr * g / (np.sqrt(s) + self.eps)


class Adam(Optimizer):
    """m <- b1 m + (1-b1) g ;  v <- b2 v + (1-b2) g^2
       m_hat = m / (1 - b1^t) ;  v_hat = v / (1 - b2^t)
       theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p, _ in self.params]
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def _update(self, i, p, g):
        m, v = self.m[i], self.v[i]
        m *= self.b1
        m += (1 - self.b1) * g
        v *= self.b2
        v += (1 - self.b2) * g * g
        m_hat = m / (1 - self.b1 ** self.t)
        v_hat = v / (1 - self.b2 ** self.t)
        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "rmsprop": RMSProp, "adam": Adam}


def make_optimizer(name: str, params: ParamList, lr: float, **kw) -> Optimizer:
    try:
        return OPTIMIZERS[name.lower()](params, lr=lr, **kw)
    except KeyError:
        raise ValueError(f"unknown optimizer {name!r}; choose from {list(OPTIMIZERS)}") from None
