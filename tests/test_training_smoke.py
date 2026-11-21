"""Short training runs on synthetic data: loss must fall and accuracy must beat chance."""
import pytest

from nnbench.data import synthetic
from nnbench.flops import summarize
from nnbench.architectures import get_spec
from nnbench.training import train_numpy


@pytest.mark.parametrize("arch,opt,lr", [("mlp", "adam", 1e-3), ("mlp", "sgd", 0.05), ("cnn", "momentum", 0.01)])
def test_numpy_learns(arch, opt, lr):
    ds = synthetic("mnist", n_train=512, n_test=256)
    h = train_numpy(ds, arch, opt, lr, epochs=2, batch_size=32, verbose=False)
    first, last = h["step_loss"][0][1], h["epochs"][-1]["train_loss"]
    assert last < first
    assert h["epochs"][-1]["test_acc"] > 0.5


def test_torch_learns():
    pytest.importorskip("torch")
    from nnbench.training import train_torch

    ds = synthetic("mnist", n_train=512, n_test=256)
    h = train_torch(ds, "cnn", "adam", 1e-3, epochs=4, batch_size=32, device="cpu", verbose=False)
    assert h["epochs"][-1]["test_acc"] > 0.5


def test_flop_counts_lenet_mnist():
    s = summarize(get_spec("cnn", (1, 28, 28)), (1, 28, 28))
    # conv1: 16*28*28*1*9 MACs, conv2: 32*14*14*16*9, fc1: 1568*128, fc2: 128*10
    macs = 16 * 28 * 28 * 9 + 32 * 14 * 14 * 16 * 9 + 1568 * 128 + 128 * 10
    assert s["fwd_macs_per_sample"] >= macs  # plus ReLU/pool ops
    assert s["params"] == (16 * 9 + 16) + (32 * 16 * 9 + 32) + (1568 * 128 + 128) + (128 * 10 + 10)
