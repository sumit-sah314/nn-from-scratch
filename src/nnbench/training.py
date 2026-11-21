"""Training loops for both frameworks with identical logging.

Each loop returns a history dict that is saved as JSON and later consumed by
scripts/plot_results.py and scripts/cost_report.py.
"""
from __future__ import annotations

import time

import numpy as np

from .architectures import build_numpy, build_torch, get_spec
from .data import Dataset, batches


def _history(cfg: dict) -> dict:
    return {"config": cfg, "epochs": [], "step_loss": []}


# --------------------------------------------------------------------------- #
# NumPy
# --------------------------------------------------------------------------- #
def evaluate_numpy(model, x, y, batch_size=1000):
    from .numpy_nn import SoftmaxCrossEntropy

    loss_fn, total, correct, n = SoftmaxCrossEntropy(), 0.0, 0, 0
    for xb, yb in batches(x, y, batch_size, shuffle=False):
        logits = model.forward(xb, training=False)
        total += loss_fn(logits, yb) * len(yb)
        correct += int((logits.argmax(1) == yb).sum())
        n += len(yb)
    return total / n, correct / n


def train_numpy(ds: Dataset, arch: str, optimizer: str, lr: float, epochs: int, batch_size: int,
                seed: int = 0, weight_decay: float = 0.0, log_every: int = 20, verbose: bool = True) -> dict:
    from .numpy_nn import SoftmaxCrossEntropy, make_optimizer

    spec = get_spec(arch, ds.in_shape, ds.num_classes)
    model = build_numpy(spec, seed=seed)
    opt = make_optimizer(optimizer, model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = SoftmaxCrossEntropy()
    rng = np.random.default_rng(seed)
    hist = _history({"framework": "numpy", "arch": arch, "dataset": ds.name, "optimizer": optimizer, "lr": lr,
                     "epochs": epochs, "batch_size": batch_size, "seed": seed, "device": "cpu",
                     "params": model.num_params()})
    step, total_train = 0, 0.0
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        run_loss, run_correct, seen = 0.0, 0, 0
        for xb, yb in batches(ds.x_train, ds.y_train, batch_size, rng=rng):
            logits = model.forward(xb)
            loss = loss_fn(logits, yb)
            model.backward(loss_fn.backward())
            opt.step()
            run_loss += loss * len(yb)
            run_correct += int((logits.argmax(1) == yb).sum())
            seen += len(yb)
            if step % log_every == 0:
                hist["step_loss"].append([step, loss])
            step += 1
        ep_time = time.perf_counter() - t0
        total_train += ep_time
        test_loss, test_acc = evaluate_numpy(model, ds.x_test, ds.y_test)
        rec = {"epoch": ep, "train_loss": run_loss / seen, "train_acc": run_correct / seen,
               "test_loss": test_loss, "test_acc": test_acc, "epoch_seconds": ep_time}
        hist["epochs"].append(rec)
        if verbose:
            print(f"[numpy {arch}] epoch {ep}: loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
                  f"test loss {test_loss:.4f} acc {test_acc:.4f} | {ep_time:.1f}s")
    hist["train_seconds"] = total_train
    hist["model"] = model
    return hist


# --------------------------------------------------------------------------- #
# PyTorch
# --------------------------------------------------------------------------- #
def evaluate_torch(model, x, y, device, batch_size=1000):
    import torch
    import torch.nn.functional as F

    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in batches(x, y, batch_size, shuffle=False):
            xb_t, yb_t = torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device)
            logits = model(xb_t)
            total += F.cross_entropy(logits, yb_t, reduction="sum").item()
            correct += (logits.argmax(1) == yb_t).sum().item()
            n += len(yb)
    model.train()
    return total / n, correct / n


def train_torch(ds: Dataset, arch: str, optimizer: str, lr: float, epochs: int, batch_size: int,
                seed: int = 0, weight_decay: float = 0.0, device: str | None = None, log_every: int = 20,
                verbose: bool = True) -> dict:
    import torch
    import torch.nn.functional as F

    from .torch_nn import make_optimizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch(get_spec(arch, ds.in_shape, ds.num_classes), seed=seed).to(device)
    opt = make_optimizer(optimizer, model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    n_params = sum(p.numel() for p in model.parameters())
    hist = _history({"framework": "torch", "arch": arch, "dataset": ds.name, "optimizer": optimizer, "lr": lr,
                     "epochs": epochs, "batch_size": batch_size, "seed": seed, "device": device,
                     "params": n_params})
    # Move the whole dataset to the device once so per-batch host->device copies
    # don't dominate GPU timings (MNIST/CIFAR fit easily in GPU memory).
    x_all = torch.from_numpy(ds.x_train).to(device)
    y_all = torch.from_numpy(ds.y_train).to(device)
    step, total_train = 0, 0.0
    for ep in range(1, epochs + 1):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx = torch.from_numpy(rng.permutation(len(ds.x_train))).to(device)
        run_loss = torch.zeros((), device=device)
        run_correct = torch.zeros((), device=device, dtype=torch.long)
        for i in range(0, len(idx), batch_size):
            j = idx[i:i + batch_size]
            xb, yb = x_all[j], y_all[j]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.detach() * len(j)
            run_correct += (logits.argmax(1) == yb).sum()
            if step % log_every == 0:
                hist["step_loss"].append([step, loss.item()])
            step += 1
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - t0
        total_train += ep_time
        test_loss, test_acc = evaluate_torch(model, ds.x_test, ds.y_test, device)
        n = len(ds.x_train)
        rec = {"epoch": ep, "train_loss": run_loss.item() / n, "train_acc": run_correct.item() / n,
               "test_loss": test_loss, "test_acc": test_acc, "epoch_seconds": ep_time}
        hist["epochs"].append(rec)
        if verbose:
            print(f"[torch/{device} {arch}] epoch {ep}: loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
                  f"test loss {test_loss:.4f} acc {test_acc:.4f} | {ep_time:.1f}s")
    hist["train_seconds"] = total_train
    hist["model"] = model
    return hist
