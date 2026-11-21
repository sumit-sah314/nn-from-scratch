"""Dataset loading with no torchvision dependency.

Downloads raw MNIST / CIFAR-10 once, converts to float32 NCHW arrays normalised
per channel with training-set statistics, and caches them as .npz under `root`.
Both the NumPy and PyTorch pipelines consume exactly these arrays, so data
loading never biases the timing comparison.
"""
from __future__ import annotations

import gzip
import os
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass

import numpy as np

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


@dataclass
class Dataset:
    x_train: np.ndarray  # (N, C, H, W) float32
    y_train: np.ndarray  # (N,) int64
    x_test: np.ndarray
    y_test: np.ndarray
    name: str
    num_classes: int = 10

    @property
    def in_shape(self) -> tuple[int, int, int]:
        return tuple(self.x_train.shape[1:])  # type: ignore[return-value]


def _download(url: str, dest: str) -> str:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest + ".part")
        os.replace(dest + ".part", dest)
    return dest


def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def _normalise(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std = x_train.std(axis=(0, 2, 3), keepdims=True)
    return (x_train - mean) / std, (x_test - mean) / std


def load_mnist(root: str = "data") -> Dataset:
    cache = os.path.join(root, "mnist.npz")
    if not os.path.exists(cache):
        raw = {k: _read_idx(_download(MNIST_URL + f, os.path.join(root, "raw", f))) for k, f in MNIST_FILES.items()}
        xtr, xte = _normalise(raw["x_train"][:, None], raw["x_test"][:, None])
        np.savez_compressed(cache, x_train=xtr, y_train=raw["y_train"].astype(np.int64),
                            x_test=xte, y_test=raw["y_test"].astype(np.int64))
    with np.load(cache) as d:
        return Dataset(d["x_train"], d["y_train"], d["x_test"], d["y_test"], "mnist")


def load_cifar10(root: str = "data") -> Dataset:
    cache = os.path.join(root, "cifar10.npz")
    if not os.path.exists(cache):
        tgz = _download(CIFAR_URL, os.path.join(root, "raw", "cifar-10-python.tar.gz"))
        xs, ys = {}, {}
        with tarfile.open(tgz) as tar:
            for m in tar.getmembers():
                base = os.path.basename(m.name)
                if base.startswith("data_batch") or base == "test_batch":
                    d = pickle.load(tar.extractfile(m), encoding="bytes")  # noqa: S301 (trusted official archive)
                    xs[base] = d[b"data"].reshape(-1, 3, 32, 32)
                    ys[base] = np.array(d[b"labels"], dtype=np.int64)
        train = sorted(k for k in xs if k.startswith("data_batch"))
        xtr, xte = _normalise(np.concatenate([xs[k] for k in train]), xs["test_batch"])
        np.savez_compressed(cache, x_train=xtr, y_train=np.concatenate([ys[k] for k in train]),
                            x_test=xte, y_test=ys["test_batch"])
    with np.load(cache) as d:
        return Dataset(d["x_train"], d["y_train"], d["x_test"], d["y_test"], "cifar10")


def synthetic(name: str = "mnist", n_train: int = 2048, n_test: int = 512, seed: int = 0) -> Dataset:
    """Random-but-learnable data with the right shapes (for tests / offline smoke runs).

    Each class has a fixed random template; samples are template + noise.
    """
    shape = (1, 28, 28) if name == "mnist" else (3, 32, 32)
    rng = np.random.default_rng(seed)
    templates = rng.standard_normal((10, *shape)).astype(np.float32)

    def make(n):
        y = rng.integers(0, 10, n)
        x = templates[y] + 1.5 * rng.standard_normal((n, *shape)).astype(np.float32)
        return x.astype(np.float32), y.astype(np.int64)

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return Dataset(xtr, ytr, xte, yte, f"synthetic-{name}")


def load(name: str, root: str = "data", subset: int | None = None, synthetic_data: bool = False) -> Dataset:
    if synthetic_data:
        ds = synthetic(name)
    elif name == "mnist":
        ds = load_mnist(root)
    elif name in ("cifar10", "cifar-10"):
        ds = load_cifar10(root)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    if subset:
        ds = Dataset(ds.x_train[:subset], ds.y_train[:subset], ds.x_test, ds.y_test, ds.name, ds.num_classes)
    return ds


def batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, rng: np.random.Generator | None = None):
    idx = np.arange(len(x))
    if shuffle:
        (rng or np.random.default_rng()).shuffle(idx)
    for i in range(0, len(x), batch_size):
        j = idx[i:i + batch_size]
        yield x[j], y[j]
