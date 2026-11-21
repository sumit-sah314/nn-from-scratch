"""The NumPy and PyTorch models, given identical weights, must agree on
logits, loss, and every parameter gradient."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from nnbench.architectures import build_numpy, build_torch, copy_numpy_to_torch, get_spec  # noqa: E402
from nnbench.numpy_nn import SoftmaxCrossEntropy  # noqa: E402


@pytest.mark.parametrize("arch", ["mlp", "cnn"])
@pytest.mark.parametrize("shape", [(1, 28, 28), (3, 32, 32)])
def test_forward_backward_parity(arch, shape):
    torch.set_default_dtype(torch.float64)
    try:
        spec = get_spec(arch, shape)
        nm = build_numpy(spec, seed=1, dtype=np.float64)
        tm = build_torch(spec, seed=1).double()
        copy_numpy_to_torch(nm, tm)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, *shape))
        y = rng.integers(0, 10, 4)

        lf = SoftmaxCrossEntropy()
        logits_np = nm.forward(x)
        loss_np = lf(logits_np, y)
        nm.backward(lf.backward())

        logits_t = tm(torch.from_numpy(x))
        loss_t = F.cross_entropy(logits_t, torch.from_numpy(y))
        loss_t.backward()

        np.testing.assert_allclose(logits_np, logits_t.detach().numpy(), rtol=1e-9, atol=1e-9)
        assert abs(loss_np - loss_t.item()) < 1e-10

        t_params = [m for m in tm if any(True for _ in m.parameters(recurse=False))]
        n_layers = [l for l in nm.layers if l.params]
        for n, t in zip(n_layers, t_params):
            gW = n.grads["W"].T if n.grads["W"].ndim == 2 else n.grads["W"]
            np.testing.assert_allclose(gW, t.weight.grad.numpy(), rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(n.grads["b"], t.bias.grad.numpy(), rtol=1e-7, atol=1e-10)

        assert nm.num_params() == sum(p.numel() for p in tm.parameters())
    finally:
        torch.set_default_dtype(torch.float32)
