"""nn-from-scratch dashboard: train NumPy and PyTorch models, then inspect and compare the runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# Streamlit puts dashboard/ on sys.path, not the repo root, so `scripts` (and `nnbench`
# when the package isn't installed) would fail to import unless we add them here.
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nnbench.architectures import ARCHITECTURES  # noqa: E402
from scripts.train import DEFAULT_LR  # noqa: E402


RESULTS = ROOT / "results"
BASELINE = {"dataset": "mnist", "arch": "cnn", "optimizer": "adam", "epochs": 5, "batch_size": 64}
DATASET_NAMES = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
FRAMEWORK_NAMES = {"numpy": "NumPy (from scratch)", "torch": "PyTorch (built-in)"}
FRAMEWORK_SHORT = {"numpy": "NumPy", "torch": "PyTorch"}


def framework_of(path: Path, history: dict) -> str:
    """Runs saved before the framework was recorded are identified by their file name."""
    framework = history["config"].get("framework")
    if framework:
        return framework
    return "torch" if path.name.startswith("torch") or path.parent.name.startswith("torch") else "numpy"


def training_controls() -> tuple[bool, dict]:
    with st.sidebar:
        st.header("New training run")
        framework = st.radio("Implementation", list(FRAMEWORK_NAMES), format_func=FRAMEWORK_NAMES.get)
        datasets = list(DATASET_NAMES)
        dataset = st.selectbox("Dataset", datasets, datasets.index(BASELINE["dataset"]), format_func=DATASET_NAMES.get)
        architectures = list(ARCHITECTURES)
        arch = st.selectbox("Architecture", architectures, architectures.index(BASELINE["arch"]), format_func=str.upper)
        optimizers = list(DEFAULT_LR)
        optimizer = st.selectbox("Optimizer", optimizers, optimizers.index(BASELINE["optimizer"]))
        epochs = st.number_input("Epochs", min_value=1, value=BASELINE["epochs"])
        lr = st.number_input(
            "Learning rate", min_value=0.0, value=DEFAULT_LR[optimizer],
            step=0.0001, format="%.8g", key=f"learning_rate_{optimizer}",
        )
        batch_size = st.number_input("Batch size", min_value=1, value=BASELINE["batch_size"])
        train = st.button("Train model", type="primary", width="stretch")
        st.caption("On an Apple M4 CPU, a CNN takes about 20 s per epoch on CIFAR-10, and an MLP takes seconds on MNIST.")

    return train, {"framework": framework, "dataset": dataset, "arch": arch, "optimizer": optimizer,
                   "epochs": epochs, "lr": lr, "batch_size": batch_size}


def launch_training(config: dict) -> Path | None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS / "runs" / "dashboard" / f"{timestamp}-{uuid4().hex[:8]}"
    command = [
        sys.executable, "-u", "-m", "dashboard.run_training",
        "--out", str(out), "--data-root", str(ROOT / "data"), "--config", json.dumps(config),
    ]
    # Unbuffered stdout carries the trainer's existing epoch logs into the UI.
    # Force UTF-8 in the child: on Windows its piped stdout otherwise defaults to the
    # ANSI code page (cp1252), so "…" arrives as a byte that decodes to "�" below.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with st.status("Training in progress", expanded=True) as status:
        try:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    status.text(line.rstrip())
                exit_code = process.wait()
        except OSError as error:
            status.text(str(error))
            status.update(label="Training could not start", state="error", expanded=True)
            return None

        history_path = out / "history.json"
        if exit_code or not history_path.is_file():
            status.update(label="Training failed", state="error", expanded=True)
            return None
        status.update(label="Training complete", state="complete", expanded=False)
    return history_path


def load_runs() -> dict[Path, dict]:
    runs = {}
    paths = sorted(RESULTS.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # Other scripts may write JSON under results, too.
        if isinstance(history, dict) and history.get("config") and history.get("epochs"):
            runs[path] = history
    return runs


def run_label(runs: dict[Path, dict], path: Path) -> str:
    config = runs[path]["config"]
    dataset = DATASET_NAMES.get(config["dataset"], config["dataset"])
    implementation = FRAMEWORK_SHORT[framework_of(path, runs[path])]
    run_id = path.parent.name if path.name == "history.json" else path.stem
    return f"{implementation} · {dataset} · {config['arch'].upper()} · {config['optimizer']} · {run_id}"


def weights_file(path: Path) -> Path | None:
    for name in ("weights.npz", "weights.pt"):
        candidate = path.with_name(name)
        if candidate.is_file():
            return candidate
    # CLI runs keep weights next to their JSON, under the same stem.
    for suffix in (".npz", ".pt"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def show_summary(runs: dict[Path, dict]) -> None:
    best_path = max(runs, key=lambda path: runs[path]["epochs"][-1]["test_acc"])
    frameworks = [framework_of(path, runs[path]) for path in runs]
    trained = sum(1 for path in runs if weights_file(path))
    accuracy, counts, best = st.columns(3)
    best_accuracy = runs[best_path]["epochs"][-1]["test_acc"]
    accuracy.metric("Best test accuracy", f"{best_accuracy:.2%}",
                    help=run_label(runs, best_path))
    counts.metric("Saved runs", len(runs),
                  help=f"{frameworks.count('numpy')} NumPy · {frameworks.count('torch')} PyTorch · "
                       f"{trained} with saved weights")
    best.metric("Best run", run_label(runs, best_path).split(" · ", 1)[0],
                help=run_label(runs, best_path))


def show_run(path: Path, history: dict) -> None:
    config, epochs = history["config"], history["epochs"]
    final = epochs[-1]
    framework = FRAMEWORK_NAMES[framework_of(path, history)]
    st.markdown(f"**{framework}** · {DATASET_NAMES.get(config['dataset'], config['dataset'])} · "
                f"{config['arch'].upper()} · {config['optimizer']}")
    st.caption(f"{len(epochs)} epochs · batch size {config['batch_size']} · learning rate {config['lr']:g}")

    accuracy, loss, duration = st.columns(3)
    accuracy.metric("Test accuracy", f"{final['test_acc']:.2%}")
    loss.metric("Test loss", f"{final['test_loss']:.4f}")
    duration.metric("Training time", f"{history['train_seconds']:.1f} s")

    st.subheader("Training curves")
    curves = path.with_name("run_curves.png")
    if path.name == "history.json" and curves.is_file():
        st.image(str(curves), width="stretch")
    else:
        # Legacy CLI runs contain epoch metrics even when no plots were saved.
        chart = st.scatter_chart if len(epochs) == 1 else st.line_chart
        loss_chart, accuracy_chart = st.columns(2)
        with loss_chart:
            chart(epochs, x="epoch", y=["train_loss", "test_loss"], x_label="Epoch", y_label="Cross-entropy")
        with accuracy_chart:
            chart(epochs, x="epoch", y=["train_acc", "test_acc"], x_label="Epoch", y_label="Accuracy")

    st.subheader("Confusion matrix")
    matrix = path.with_name("confusion_matrix.png")
    if history.get("confusion_matrix") and matrix.is_file():
        _, centre, _ = st.columns([1, 2, 1])
        centre.image(str(matrix), width="stretch")
    else:
        st.info("This run has no saved test predictions, so there is no confusion matrix.")

    st.subheader("Artifacts")
    downloads = st.columns(3)
    downloads[0].download_button("Download history (JSON)", path.read_bytes(), file_name=path.name,
                                 mime="application/json", width="stretch")
    weights = weights_file(path)
    if weights:
        downloads[1].download_button("Download weights", weights.read_bytes(), file_name=weights.name,
                                     mime="application/octet-stream", width="stretch")
        downloads[2].caption(f"Weights: `{weights.name}`")
    else:
        downloads[1].caption("No saved weights: this run predates weight saving.")
    st.caption(path.relative_to(ROOT).as_posix())


def compare_runs(runs: dict[Path, dict]) -> None:
    """Put runs from the from-scratch NumPy and built-in PyTorch implementations side by side."""
    st.markdown("Pick runs from either implementation. Compare accuracy, cost and learning curves.")
    # Default to the newest run from each implementation on the same dataset and architecture:
    # the first view should be a fair comparison. Runs are sorted newest first.
    newest = next(iter(runs))
    setup = (runs[newest]["config"]["dataset"], runs[newest]["config"]["arch"])
    default = {}
    for path in runs:
        config = runs[path]["config"]
        if (config["dataset"], config["arch"]) == setup:
            default.setdefault(framework_of(path, runs[path]), path)
    chosen = st.multiselect("Runs to compare", list(runs), default=list(default.values()),
                            format_func=lambda path: run_label(runs, path), key="compared_runs")
    if not chosen:
        st.info("Select one or more runs to compare.")
        return

    configs = [runs[path]["config"] for path in chosen]
    if len({(c["dataset"], c["arch"]) for c in configs}) > 1:
        st.warning("These runs use different datasets or architectures, so their accuracies are not directly comparable.")

    rows = []
    for path in chosen:
        history = runs[path]
        config, final = history["config"], history["epochs"][-1]
        rows.append({
            "Run": run_label(runs, path),
            "Test accuracy (%)": 100 * final["test_acc"],
            "Test loss": final["test_loss"],
            "Training time (s)": history["train_seconds"],
            "Epochs": len(history["epochs"]),
            "Batch size": config["batch_size"],
            "Learning rate": config["lr"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Test accuracy (%)": st.column_config.NumberColumn(format="%.2f"),
        "Test loss": st.column_config.NumberColumn(format="%.4f"),
        "Training time (s)": st.column_config.NumberColumn(format="%.1f"),
        "Learning rate": st.column_config.NumberColumn(format="%.2g"),
    })

    st.subheader("Test accuracy per epoch")
    curves = pd.DataFrame([
        {"Run": run_label(runs, path), "Epoch": epoch["epoch"], "Test accuracy": epoch["test_acc"]}
        for path in chosen for epoch in runs[path]["epochs"]
    ])
    # Zoom the axis to the data: differences of a few tenths of a percent are the whole story here.
    chart = alt.Chart(curves).mark_line(point=True).encode(
        x=alt.X("Epoch:O", title="Epoch"),
        y=alt.Y("Test accuracy:Q", title="Test accuracy", scale=alt.Scale(zero=False), axis=alt.Axis(format=".1%")),
        color=alt.Color("Run:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Run", "Epoch", alt.Tooltip("Test accuracy:Q", format=".2%")],
    ).properties(height=340)
    st.altair_chart(chart, width="stretch")


def show_benchmark_figures() -> None:
    figures = sorted((RESULTS / "figures").glob("*.png"))
    if not figures:
        st.info("No benchmark figures yet. Run `python scripts/plot_results.py` to generate them.")
        return
    st.markdown("Figures generated from the full MNIST and CIFAR-10 grid (3 seeds × 10 epochs, CPU). "
                "See the README for the discussion.")
    for start in range(0, len(figures), 2):
        columns = st.columns(2)
        for column, figure in zip(columns, figures[start:start + 2]):
            column.image(str(figure), caption=figure.stem.replace("_", " "), width="stretch")


def show_about() -> None:
    st.markdown(
        """
**nn-from-scratch** implements an MLP and a LeNet-style CNN, plus SGD, momentum, RMSProp and Adam,
using only NumPy: forward and backward passes, im2col convolution, max-pooling and the weight updates.

The same architectures are rebuilt in PyTorch from one shared spec, so both sides have identical layers,
parameter counts and initialisation. The test suite checks that they compute the same function:

- **Gradient checks** against finite differences for every layer and the full CNN.
- **Framework parity**: with weights copied across, logits, loss and every gradient match PyTorch.
- **Optimizer parity**: each optimizer matches `torch.optim` after 25 steps.

Use the **Train** panel to run either implementation. Each run saves its history, plots, confusion matrix
and trained weights (`weights.npz` for NumPy, `weights.pt` for PyTorch) under `results/runs/dashboard/`.
        """
    )
    st.link_button("View the project on GitHub", "https://github.com/sumit-sah314/nn-from-scratch",
                   width="content")


def main() -> None:
    st.set_page_config(page_title="nn-from-scratch", page_icon="🧠", layout="wide")
    st.title("nn-from-scratch")
    st.markdown("Neural networks written by hand in NumPy, benchmarked against PyTorch on MNIST and CIFAR-10.")

    train, config = training_controls()
    if train:
        path = launch_training(config)
        if path is not None:
            st.session_state.selected_run = path

    runs = load_runs()
    if runs:
        show_summary(runs)

    explore, compare, figures, about = st.tabs(["Explore runs", "Compare", "Benchmark figures", "About"])
    with explore:
        if runs:
            selected = st.selectbox("Saved run", list(runs), format_func=lambda path: run_label(runs, path),
                                    key="selected_run")
            show_run(selected, runs[selected])
        else:
            st.info("No saved runs yet. Choose a configuration in the sidebar and select **Train model**.")
    with compare:
        if runs:
            compare_runs(runs)
        else:
            st.info("Train at least one model to compare runs.")
    with figures:
        show_benchmark_figures()
    with about:
        show_about()


if __name__ == "__main__":
    main()
