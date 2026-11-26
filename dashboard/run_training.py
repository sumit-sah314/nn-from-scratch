"""Run the existing NumPy trainer and save one dashboard run's artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from filelock import FileLock
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
# This runs as its own process (`python -m dashboard.run_training`), so it does not
# inherit app.py's sys.path changes. Where nnbench isn't pip-installed (e.g. Streamlit
# Community Cloud, which only installs requirements.txt), add src/ and the repo root.
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nnbench.data import batches, load  # noqa: E402
from nnbench.training import train_numpy, train_torch  # noqa: E402


CIFAR10_LABELS = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def save_confusion_plot(counts: np.ndarray, labels: list[str], path: Path) -> None:
    figure = Figure(figsize=(7.5, 6), layout="constrained")
    axis = figure.subplots()
    image = axis.imshow(counts, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Test samples")
    ticks = range(len(labels))
    axis.set_xticks(ticks, labels, rotation=35, ha="right")
    axis.set_yticks(ticks, labels)
    axis.set(xlabel="Predicted class", ylabel="True class")
    midpoint = counts.max() / 2
    for (true_class, predicted_class), count in np.ndenumerate(counts):
        color = "white" if count > midpoint else "#17324d"
        axis.text(predicted_class, true_class, str(count), ha="center", va="center", color=color, fontsize=8)
    figure.savefig(path, dpi=130)


def save_run_plot(history: dict, path: Path) -> None:
    """One figure per run: step loss, epoch loss, and epoch accuracy, with the run's own title."""
    config, epochs = history["config"], history["epochs"]
    framework = "NumPy (from scratch)" if config["framework"] == "numpy" else "PyTorch (built-in)"
    figure = Figure(figsize=(12, 3.8), layout="constrained")
    step_axis, loss_axis, acc_axis = figure.subplots(1, 3)
    figure.suptitle(f"{framework} · {config['arch'].upper()} · {config['dataset'].upper()} · "
                    f"{config['optimizer']} · lr {config['lr']:g} · batch {config['batch_size']}", fontsize=11)

    if history.get("step_loss"):
        steps, losses = zip(*history["step_loss"])
        step_axis.plot(steps, losses, color="#8fb8e8", linewidth=1, label="train loss (per logged step)")
        step_axis.set_yscale("log")
    step_axis.set(title="Training loss per step", xlabel="step", ylabel="cross-entropy (log scale)")

    x = [e["epoch"] for e in epochs]
    loss_axis.plot(x, [e["train_loss"] for e in epochs], marker="o", label="train")
    loss_axis.plot(x, [e["test_loss"] for e in epochs], marker="o", label="test")
    loss_axis.set(title="Loss per epoch", xlabel="epoch", ylabel="cross-entropy")
    acc_axis.plot(x, [e["train_acc"] for e in epochs], marker="o", label="train")
    acc_axis.plot(x, [e["test_acc"] for e in epochs], marker="o", label="test")
    acc_axis.set(title="Accuracy per epoch", xlabel="epoch", ylabel="accuracy")
    acc_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    for axis in (loss_axis, acc_axis):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(frameon=False)
        axis.grid(alpha=0.3)
    step_axis.grid(alpha=0.3)
    figure.savefig(path, dpi=130)


def predict(model, images: np.ndarray, framework: str) -> np.ndarray:
    """Class predictions from either the from-scratch NumPy model or the PyTorch model."""
    if framework == "numpy":
        return model.forward(images, training=False).argmax(axis=1)
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(images).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def save_weights(model, framework: str, path: Path) -> None:
    if framework == "numpy":
        model.save(str(path))  # NumPy arrays in an .npz file
    else:
        import torch

        torch.save(model.state_dict(), path)


def run_training(out: Path, data_root: Path, dataset: str, framework: str = "numpy", **training_options) -> None:
    print(f"Loading {dataset.upper()}…")
    data_root.mkdir(parents=True, exist_ok=True)
    # The upstream loader uses a shared cache and fixed download filenames.
    with FileLock(str(data_root / f".{dataset}.lock")):
        data = load(dataset, root=str(data_root))

    name = "NumPy (from scratch)" if framework == "numpy" else "PyTorch (built-in)"
    print(f"Training the {name} model…")
    trainer = train_numpy if framework == "numpy" else train_torch
    history = trainer(data, **training_options)

    # The CLI drops the fitted model. Use it here for held-out predictions
    # before persisting the CLI-compatible history.
    print("Evaluating test predictions and generating plots…")
    model = history.pop("model")
    history["config"]["framework"] = framework
    counts = np.zeros((data.num_classes, data.num_classes), dtype=int)
    for images, targets in batches(data.x_test, data.y_test, 128, shuffle=False):
        np.add.at(counts, (targets, predict(model, images, framework)), 1)
    labels = list(CIFAR10_LABELS) if dataset == "cifar10" else [str(i) for i in range(data.num_classes)]
    history["confusion_matrix"] = {"labels": labels, "counts": counts.tolist()}

    # Isolate each run so the batch plotter never mixes epochs or other
    # hyperparameters across dashboard runs.
    out.mkdir(parents=True, exist_ok=False)
    save_run_plot(history, out / "run_curves.png")
    save_confusion_plot(counts, labels, out / "confusion_matrix.png")
    save_weights(model, framework, out / ("weights.npz" if framework == "numpy" else "weights.pt"))
    pending = out / "history.tmp"
    pending.write_text(json.dumps(history, indent=1), encoding="utf-8")
    pending.replace(out / "history.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=json.loads, required=True)
    args = parser.parse_args()
    run_training(args.out, args.data_root, **args.config)


if __name__ == "__main__":
    main()
