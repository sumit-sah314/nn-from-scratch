"""Loss functions."""
from __future__ import annotations

import numpy as np


class SoftmaxCrossEntropy:
    """Softmax + negative log-likelihood, fused for numerical stability.

    forward(logits (N,K), labels (N,) int) -> mean loss (float)
    backward() -> dL/dlogits = (softmax - onehot) / N
    """

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        self.probs = np.exp(log_probs)
        self.labels = labels
        n = logits.shape[0]
        return float(-log_probs[np.arange(n), labels].mean())

    __call__ = forward

    def backward(self) -> np.ndarray:
        n = self.probs.shape[0]
        d = self.probs.copy()
        d[np.arange(n), self.labels] -= 1.0
        return d / n
