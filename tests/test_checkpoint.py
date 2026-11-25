"""Saved NumPy weights must reload into a fresh model and reproduce its outputs."""
import numpy as np

from nnbench.architectures import build_numpy, get_spec


def test_numpy_weights_round_trip(tmp_path):
    shape = (1, 28, 28)
    spec = get_spec("cnn", shape)
    trained = build_numpy(spec, seed=1)
    x = np.random.default_rng(0).standard_normal((3, *shape)).astype(np.float32)

    path = tmp_path / "weights.npz"
    trained.save(str(path))

    restored = build_numpy(spec, seed=2)  # different init, so the load must do the work
    assert not np.allclose(restored.forward(x, training=False), trained.forward(x, training=False))
    restored.load(str(path))
    np.testing.assert_array_equal(restored.forward(x, training=False), trained.forward(x, training=False))
