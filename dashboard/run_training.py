"""Run the existing NumPy trainer and save one dashboard run's artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from filelock import FileLock
from matplotlib.figure import Figure

from nnbench.data import batches, load
from nnbench.training import train_numpy
from scripts.plot_results import plot_convergence


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


def run_training(out: Path, data_root: Path, dataset: str, **training_options) -> None:
    print(f"Loading {dataset.upper()}…")
    data_root.mkdir(parents=True, exist_ok=True)
    # The upstream loader uses a shared cache and fixed download filenames.
    with FileLock(str(data_root / f".{dataset}.lock")):
        data = load(dataset, root=str(data_root))

    print("Training the NumPy model…")
    history = train_numpy(data, **training_options)

    # The CLI drops the fitted model. Use it here for held-out predictions
    # before persisting the CLI-compatible history.
    print("Evaluating test predictions and generating plots…")
    model = history.pop("model")
    counts = np.zeros((data.num_classes, data.num_classes), dtype=int)
    for images, targets in batches(data.x_test, data.y_test, 128, shuffle=False):
        predictions = model.forward(images, training=False).argmax(axis=1)
        np.add.at(counts, (targets, predictions), 1)
    labels = list(CIFAR10_LABELS) if dataset == "cifar10" else [str(i) for i in range(data.num_classes)]
    history["confusion_matrix"] = {"labels": labels, "counts": counts.tolist()}

    # Isolate each run so the batch plotter never mixes epochs or other
    # hyperparameters across dashboard runs.
    out.mkdir(parents=True, exist_ok=False)
    plot_convergence([history], str(out))
    save_confusion_plot(counts, labels, out / "confusion_matrix.png")
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
