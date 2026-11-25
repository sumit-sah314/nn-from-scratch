# nn-from-scratch

**Neural networks from scratch in NumPy, benchmarked against PyTorch.**

We implement a fully connected network (MLP), a convolutional network (CNN), and four optimizers (SGD, SGD + Momentum, RMSProp, Adam) using **only NumPy**: forward propagation, backpropagation, im2col convolution, max-pooling and weight updates, all written by hand. We then build the **same architectures in PyTorch** and compare both on **MNIST** and **CIFAR-10** across:

| Axis | How it's measured |
|---|---|
| Accuracy & loss convergence | per-step training loss, per-epoch train/test loss & accuracy (`scripts/train.py`) |
| Training & inference time | median of N timed repeats after warm-up; `cuda.synchronize()` on GPU (`nnbench/profiling.py`) |
| Memory usage | `tracemalloc` peak (NumPy), PyTorch allocator profiler / `cuda.max_memory_allocated` (PyTorch), plus an analytic estimate |
| Parameter count | counted from both models; asserted equal in tests |
| Compute cost (FLOPs) | analytic per-layer counter (`nnbench/flops.py`), cross-checked with `ptflops` / `fvcore` |
| Dollar cost | measured hours × cloud hourly rates in `configs/pricing.json` (`scripts/cost_report.py`) |

[![tests](https://github.com/sumit-sah314/nn-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/sumit-sah314/nn-from-scratch/actions)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumit-sah314/nn-from-scratch/blob/main/notebooks/colab_quickstart.ipynb)

![Streamlit dashboard: run metrics, training curves and confusion matrix](docs/dashboard-explore.png)

**Highlights**
- Forward and backward passes, im2col convolution, max-pooling and four optimizers written by hand in NumPy, checked against finite differences and against `torch.optim`.
- The same MLP and CNN are rebuilt in PyTorch from one shared spec, so the comparison is between identical models.
- A Streamlit dashboard trains either implementation, saves each model's weights, and compares runs side by side.
- Full MNIST and CIFAR-10 grids (3 seeds × 10 epochs), with timing, memory, FLOP and cost tables.

---

## Correctness first

A speed comparison is meaningless if the two implementations compute different things, so the test suite proves they don't:

- **Gradient checks** (`tests/test_gradcheck.py`): every backward pass (Dense, ReLU, Conv2D with several strides/paddings, MaxPool, softmax cross-entropy, a full CNN) matches centered finite differences to relative error < 1e-6 in float64.
- **Framework parity** (`tests/test_torch_parity.py`): with weights copied across, the NumPy and PyTorch MLP/CNN produce identical logits, loss and **every parameter gradient** (rtol 1e-7) on both MNIST and CIFAR-10 input shapes.
- **Optimizer parity** (`tests/test_optim_parity.py`): each from-scratch optimizer, with and without weight decay, matches `torch.optim` after 25 steps to 1e-10.

- **Weight round trip** (`tests/test_checkpoint.py`): saved NumPy weights reload into a fresh model and reproduce its outputs exactly.

```bash
pytest -q        # 24 tests, ~5 s on CPU (torch tests run when torch is installed)
```

## Quickstart

```bash
git clone https://github.com/sumit-sah314/nn-from-scratch.git
cd nn-from-scratch
pip install -e ".[torch,dev,dashboard]"   # or: pip install -r requirements.txt && pip install -e .

make smoke                              # offline end-to-end run on synthetic data
python scripts/train.py --framework numpy --arch cnn --dataset mnist --optimizer adam --epochs 5
python scripts/train.py --framework torch --arch cnn --dataset mnist --optimizer adam --epochs 5
python scripts/benchmark.py             # timing / memory / FLOPs table
python scripts/plot_results.py          # figures -> results/figures/
python scripts/cost_report.py --scale-epochs 20
streamlit run dashboard/app.py          # interactive dashboard (see below)
```

Datasets download automatically on first use (no torchvision needed) and are cached as normalised `.npz` arrays in `data/`. Both frameworks train on exactly the same arrays.

## Dashboard

```bash
streamlit run dashboard/app.py
```

- **Train** either implementation from the sidebar: NumPy (from scratch) or PyTorch (built-in), on MNIST or CIFAR-10, with the MLP or CNN, any of the four optimizers, and your own learning rate, epochs and batch size. Training output streams into the page.
- **Explore runs** shows the summary metrics, a per-run figure (step loss, epoch loss, epoch accuracy), the confusion matrix, and download buttons for the history and the weights.
- **Compare** puts runs side by side in a table and an accuracy-per-epoch chart. It warns when the runs use different datasets or architectures, since those numbers are not comparable.
- **Benchmark figures** shows the generated plots from `results/figures/`.

Each training run is saved under `results/runs/dashboard/<timestamp>/`:

| file | contents |
|---|---|
| `history.json` | config, per-step loss, per-epoch metrics, confusion matrix |
| `weights.npz` / `weights.pt` | trained parameters (NumPy arrays / PyTorch `state_dict`) |
| `run_curves.png`, `confusion_matrix.png` | the figures shown in the dashboard |

`scripts/train.py` saves the same weights next to its JSON history.

`make all` runs the full grid: 2 frameworks × 2 architectures × 2 datasets × 4 optimizers, then benchmarks, plots and the cost report. On Colab, open `notebooks/colab_quickstart.ipynb` with a T4 GPU runtime to get CPU and GPU numbers in one session.

## Repository layout

```
src/nnbench/
├── numpy_nn/              # the from-scratch library
│   ├── layers.py          #   Dense, ReLU, Flatten, Conv2D (im2col), MaxPool2D
│   ├── losses.py          #   fused softmax + cross-entropy
│   ├── optim.py           #   SGD, Momentum, RMSProp, Adam
│   └── model.py           #   Sequential container, save/load, per-layer FLOPs
├── torch_nn/models.py     # PyTorch reference models + optimizer factory
├── architectures.py       # one spec -> identical NumPy and PyTorch models
├── data.py                # MNIST / CIFAR-10 download, normalisation, batching
├── training.py            # matched training loops with identical logging
├── flops.py               # analytic FLOPs / params / memory + ptflops/fvcore cross-check
├── profiling.py           # timers, tracemalloc, torch memory profiler
└── cost.py                # dollars from measured time x hourly rate
scripts/                   # train.py, benchmark.py, profile_numpy.py, plot_results.py, cost_report.py
dashboard/                 # Streamlit app (app.py) and its training runner (run_training.py)
tests/                     # gradcheck, torch parity, optimizer parity, checkpoints, training smoke tests
configs/pricing.json       # cloud hourly rates (edit me)
notebooks/                 # Colab quickstart
docs/                      # screenshots used in this README
```

## Architectures

Both are defined once in `architectures.py` and instantiated in both frameworks.

| Model | Layers | Params (MNIST) | Params (CIFAR-10) | Fwd MFLOPs/sample (MNIST / CIFAR) |
|---|---|---|---|---|
| MLP | Flatten → 256 → ReLU → 128 → ReLU → 10 | 235,146 | 820,874 | 0.47 / 1.64 |
| CNN | [Conv3×3 → ReLU → MaxPool2] × 2 (16, 32 ch) → FC 128 → ReLU → FC 10 | 206,922 | 268,650 | 2.47 / 3.81 |

## Implementation notes

**Convolution via im2col.** Patches are unfolded with one strided copy per kernel offset (9 copies for 3×3) instead of one per output pixel, turning convolution into a single matrix multiply that NumPy hands to BLAS. The backward pass is two more GEMMs (for dW and dX) plus `col2im`, which scatter-adds overlapping patches back. MaxPool reuses the same unfolding and routes gradients through the stored `argmax`.

**Optimizers mirror `torch.optim` exactly**: momentum uses PyTorch's `v = μv + g` convention, RMSProp puts ε outside the square root, Adam uses bias-corrected moments, and weight decay is L2 added to the gradient (not AdamW).

**Same initialisation** (He/Kaiming normal, zero bias) on both sides so convergence curves start comparably. They won't be bit-identical across frameworks because the random number generators differ; use several `--seed` values and report mean ± std.

## Methodology and caveats

- **FLOPs convention:** 1 multiply-accumulate = 2 FLOPs. `ptflops` and `fvcore` report MACs, so divide our numbers by 2 to compare. Backward is estimated as 2× forward for Dense/Conv (one GEMM for dW, one for dX) and 1× for elementwise/pooling layers, so a training step ≈ 3× forward.
- **Threads:** NumPy's BLAS and PyTorch both use multiple threads by default. `benchmark.py --threads 1` pins both to one thread, which isolates algorithmic/implementation efficiency from parallelism. Report both.
- **Memory:** `tracemalloc` sees NumPy buffers but **not** PyTorch's C++ allocator, so the PyTorch CPU number comes from `torch.profiler` memory events. Treat cross-framework memory as approximate and lean on the analytic estimate (weights + grads + optimizer state + activations) for a like-for-like figure.
- **GPU:** only PyTorch runs on GPU; the NumPy code is CPU-only by design. A natural extension is swapping `numpy` for `cupy` to separate "framework overhead" from "hardware".
- **Data loading** is excluded from timing: both frameworks iterate over the same preloaded arrays, and the PyTorch loop moves the dataset to the device once.

## Cost estimation

`configs/pricing.json` lists on-demand hourly rates (checked 2026-11-20, us-east-1 where applicable):

| Instance | Hardware | USD/hr |
|---|---|---|
| AWS c5.xlarge | 4 vCPU | 0.17 |
| AWS c5.4xlarge | 16 vCPU | 0.68 |
| AWS g4dn.xlarge | NVIDIA T4 | 0.526 |
| AWS p3.2xlarge | NVIDIA V100 | 3.06 |
| Colab Enterprise T4 | accelerator fee only | 0.42 |
| Colab pay-as-you-go T4 | ~1.19 compute units/hr × $0.10 | ~0.12 |

`scripts/cost_report.py` prices CPU runs on CPU instances and CUDA runs on GPU instances, optionally extrapolating to more epochs from the measured mean epoch time. Prices change; update the file and its `checked` date before reporting numbers.

## Results

`results/benchmark.md` and `results/figures/` are generated automatically by `make all`.

Measured on an Apple M4 (CPU only, multithreaded BLAS, batch 64, Adam, 20 timed repeats after warm-up; `results/benchmark.md`):

| arch | dataset | impl | params | fwd MFLOPs/sample | train step (ms) | inference (µs/sample) | achieved GFLOP/s | peak mem (MB) |
|---|---|---|---|---|---|---|---|---|
| MLP | MNIST | NumPy | 235,146 | 0.47 | 0.43 | 0.57 | 210 | 4.0 |
| MLP | MNIST | PyTorch CPU | 235,146 | 0.47 | 0.44 | 0.51 | 205 | 5.1 |
| CNN | MNIST | NumPy | 206,922 | 2.47 | 15.5 | 141 | 30 | 23.5 |
| CNN | MNIST | PyTorch CPU | 206,922 | 2.47 | 6.5 | 20.1 | 73 | 53.6 |
| MLP | CIFAR-10 | NumPy | 820,874 | 1.64 | 1.47 | 1.37 | 215 | 15.1 |
| MLP | CIFAR-10 | PyTorch CPU | 820,874 | 1.64 | 0.91 | 1.39 | 347 | 16.2 |
| CNN | CIFAR-10 | NumPy | 268,650 | 3.81 | 22.0 | 198 | 33 | 36.5 |
| CNN | CIFAR-10 | PyTorch CPU | 268,650 | 3.81 | 9.6 | 46.3 | 76 | 69.9 |

The analytic FLOP counts match `ptflops` and `fvcore` (MLP MNIST: 235,146 params, 235,914 MACs from `ptflops`).

**Single thread vs multi thread** (`results/single_thread/benchmark.md`, `OMP/OPENBLAS_NUM_THREADS=1`): the NumPy CNN CIFAR-10 step is 22.1 ms single-threaded versus 22.0 ms multithreaded, and PyTorch's is 11.1 ms versus 9.6 ms. On this machine the batch-64 matrices are too small for BLAS threading to help the NumPy side, so the numbers are nearly identical.

**Where the NumPy CNN time goes** (cProfile over 50 CIFAR-10 training steps, `results/profile_cnn_cifar10.txt`, one-step ≈ 22 ms):

| operation | share of profiled time |
|---|---|
| `col2im` (scatter-add in conv and pool backward) | 26% |
| `argmax` (max-pool forward) | 18% |
| `reshape` / transpose copies | 13% |
| `im2col` (patch unfolding) | 11% |
| `put_along_axis` / `take_along_axis` (pool routing) | 12% |
| remaining Python and layer bodies, including the GEMMs | ~20% |

cProfile does not record the `@` operator separately, so the matrix multiplies sit inside the layer bodies in the last row. The shares are approximate because cProfile adds per-call overhead. The main finding is that about 80% of the NumPy CNN's time is memory movement (unfolding, scatter-add, copies, and argmax), not arithmetic. Pooling is a large share on its own, because it is implemented through `im2col` plus `argmax` rather than a direct kernel.

Takeaway: the from-scratch MLP is within ~1–1.6× of PyTorch on training steps, because both are dominated by the same BLAS GEMMs. The CNN is ~2.3–2.4× slower on training steps, and ~7× slower on MNIST inference, which comes from im2col memory traffic and Python-level overhead, which PyTorch avoids with fused convolution kernels (oneDNN on CPU, cuDNN on GPU).

### Accuracy and convergence (3 seeds × 10 epochs, CPU)

Full table: `results/summary.md`. Mean ± std over seeds, final epoch, test set:

| dataset | arch | NumPy (from scratch), best optimizer | PyTorch, best optimizer | NumPy train time (s) | PyTorch train time (s) |
|---|---|---|---|---|---|
| MNIST | MLP | 98.20 ± 0.12% (momentum) | 98.17 ± 0.02% (momentum) | 1.9–4.3 | 2.5–4.8 |
| MNIST | CNN | 98.96 ± 0.17% (momentum) | 98.94 ± 0.18% (momentum) | 136–142 | 63–67 |
| CIFAR-10 | MLP | 50.98 ± 0.13% (adam) | 50.85 ± 0.34% (adam) | 5–14 | 5–12 |
| CIFAR-10 | CNN | 67.45 ± 0.52% (adam) | 67.63 ± 0.26% (adam) | 181–219 | 80–118 |

Figures: `results/figures/convergence_{mlp,cnn}_{mnist,cifar10}.png` (mean over seeds, ±1 std bands). The benchmark plot is `results/figures/benchmark.png`.

**What the results show**
- **The from-scratch implementation matches PyTorch on accuracy.** For the best optimizer in each row, the two frameworks differ by less than 0.2 points. Across all optimizers, the gaps are mostly within one standard deviation.
- **Optimizers differ little in final accuracy.** On MNIST all four are within about 0.6% for each architecture. On CIFAR-10 the CNN's ranking is Adam ≈ momentum > SGD ≈ RMSProp by roughly 2–3 points, and the spread across seeds is comparable to those gaps.
- **Test loss is lowest around epochs 3–6 and then rises** on CIFAR-10, while accuracy plateaus. The network is overfitting the training set. There is no augmentation or regularization here, so this is expected.
- **Training is 2–3× slower in NumPy** for the CNN (about 180–220 s versus 80–120 s for 10 epochs on CIFAR-10), and the gap is larger for inference (see the benchmark above). The MLP training time is close to PyTorch's.

**Limitations**
- Everything was measured on a single Apple M4 CPU. The Colab GPU runs in the checklist have not been done.
- Only 3 seeds were used, so differences of under about 1 point should not be read as real.
- Weights are saved for runs made after this feature was added (dashboard runs and `scripts/train.py`). The runs in `results/runs/` from before then contain only their histories.

## Project checklist

- [x] NumPy layers, loss, optimizers + gradient checks
- [x] PyTorch mirror + parity tests
- [x] Training / benchmark / plotting / cost scripts
- [x] Full MNIST runs (3 seeds × 10 epochs, CPU)
- [x] Full CIFAR-10 runs (3 seeds × 10 epochs, CPU)
- [ ] Colab GPU runs of the same grid
- [x] Single-thread vs multi-thread benchmark comparison
- [x] cProfile breakdown of the NumPy CNN (`scripts/profile_numpy.py`)
- [x] Final report: plots, tables, discussion
- [x] Streamlit dashboard: train either implementation, inspect and compare runs
- [x] Trained weights saved for every new run (NumPy `.npz`, PyTorch `.pt`)

## References

- I. Goodfellow, Y. Bengio, A. Courville. *Deep Learning*. MIT Press, 2016. Chapters 6, 8, 9.
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition, course notes (backprop, im2col, optimizers). https://cs231n.github.io/
- D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization*. ICLR 2015.
- G. Hinton. Lecture 6e, *RMSProp*, Neural Networks for Machine Learning (Coursera), 2012.
- PyTorch documentation, `torch.optim`. https://pytorch.org/docs/stable/optim.html

## License

MIT
