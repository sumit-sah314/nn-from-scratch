"""Each from-scratch optimizer must match torch.optim step-for-step."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nnbench.numpy_nn import make_optimizer  # noqa: E402
from nnbench.torch_nn import make_optimizer as torch_make  # noqa: E402

CASES = [("sgd", 0.1, 0.0), ("momentum", 0.05, 0.0), ("rmsprop", 0.01, 0.0), ("adam", 0.01, 0.0),
         ("adam", 0.01, 1e-2), ("momentum", 0.05, 1e-2)]


@pytest.mark.parametrize("name,lr,wd", CASES)
def test_optimizer_matches_torch(name, lr, wd):
    rng = np.random.default_rng(0)
    p0 = rng.standard_normal((4, 3))
    target = rng.standard_normal((4, 3))

    # NumPy: minimise 0.5*||p - target||^2 * (row weights) -> grad computed by hand
    w = np.arange(1, 5)[:, None].astype(np.float64)
    p_np = p0.copy()
    g_np = np.zeros_like(p_np)
    opt = make_optimizer(name, [(p_np, g_np)], lr=lr, weight_decay=wd)

    p_t = torch.tensor(p0.copy(), requires_grad=True)
    topt = torch_make(name, [p_t], lr=lr, weight_decay=wd)
    tw, tt = torch.from_numpy(w), torch.from_numpy(target)

    for _ in range(25):
        g_np[...] = w * (p_np - target)
        opt.step()
        topt.zero_grad()
        (0.5 * (tw * (p_t - tt) ** 2).sum()).backward()
        topt.step()

    np.testing.assert_allclose(p_np, p_t.detach().numpy(), rtol=1e-10, atol=1e-12)
