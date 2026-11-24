"""Train and inspect nnbench runs with Streamlit."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from nnbench.architectures import ARCHITECTURES
from scripts.train import DEFAULT_LR


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASELINE = {"dataset": "mnist", "arch": "cnn", "optimizer": "adam", "epochs": 5, "batch_size": 64}
DATASET_NAMES = {"mnist": "MNIST", "cifar10": "CIFAR-10"}


def training_controls() -> tuple[bool, dict]:
    with st.sidebar:
        st.header("Training")
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
        train = st.button("Train", type="primary", width="stretch")

    return train, {"dataset": dataset, "arch": arch, "optimizer": optimizer,
                   "epochs": epochs, "lr": lr, "batch_size": batch_size}


def launch_training(config: dict) -> Path | None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS / "runs" / "dashboard" / f"{timestamp}-{uuid4().hex[:8]}"
    command = [
        sys.executable, "-u", "-m", "dashboard.run_training",
        "--out", str(out), "--data-root", str(ROOT / "data"), "--config", json.dumps(config),
    ]
    # Unbuffered stdout carries the trainer's existing epoch logs into the UI.
    with st.status("Training in progress", expanded=True) as status:
        try:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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


def show_run(path: Path, history: dict) -> None:
    config, epochs = history["config"], history["epochs"]
    final = epochs[-1]
    st.caption(f"Epochs {len(epochs)} · Batch size {config['batch_size']} · Learning rate {config['lr']:g}")
    accuracy, loss, duration = st.columns(3)
    accuracy.metric("Test accuracy", f"{final['test_acc']:.2%}")
    loss.metric("Test loss", f"{final['test_loss']:.4f}")
    duration.metric("Training time", f"{history['train_seconds']:.1f} s")

    st.subheader("Loss and accuracy")
    convergence = path.with_name(f"convergence_{config['arch']}_{config['dataset']}.png")
    if path.name == "history.json" and convergence.is_file():
        st.image(str(convergence), width="stretch")
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
        st.image(str(matrix), width=650)
    else:
        st.info("This run's history has no saved test predictions or confusion matrix.")
    st.caption(path.relative_to(ROOT).as_posix())


def main() -> None:
    st.set_page_config(page_title="Neural Network Benchmarks", layout="wide")
    st.title("Neural Network Benchmarks")
    train, config = training_controls()
    if train:
        path = launch_training(config)
        if path is not None:
            st.session_state.selected_run = path

    runs = load_runs()
    st.header("Run results")
    if runs:
        def run_label(path: Path) -> str:
            config = runs[path]["config"]
            dataset = DATASET_NAMES.get(config["dataset"], config["dataset"])
            run_id = path.parent.name if path.name == "history.json" else path.stem
            return f"{dataset} · {config['arch'].upper()} · {config['optimizer']} · {run_id}"

        selected = st.selectbox("Saved run", list(runs), format_func=run_label, key="selected_run")
        show_run(selected, runs[selected])
    else:
        st.info("No saved runs yet. Choose a configuration and select Train.")

    figures = sorted((RESULTS / "figures").glob("*.png"))
    if figures:
        with st.expander("Existing result figures", expanded=not runs):
            for figure in figures:
                st.image(str(figure), caption=figure.stem.replace("_", " "), width="stretch")


if __name__ == "__main__":
    main()
