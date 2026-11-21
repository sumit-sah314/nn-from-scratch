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

---

## Correctness first

A speed comparison is meaningless if the two implementations compute different things, so the test suite proves they don't:

- **Gradient checks** (`tests/test_gradcheck.py`): every backward pass (Dense, ReLU, Conv2D with several strides/paddings, MaxPool, softmax cross-entropy, a full CNN) matches centered finite differences to relative error < 1e-6 in float64.
- **Framework parity** (`tests/test_torch_parity.py`): with weights copied across, the NumPy and PyTorch MLP/CNN produce identical logits, loss and **every parameter gradient** (rtol 1e-7) on both MNIST and CIFAR-10 input shapes.
- **Optimizer parity** (`tests/test_optim_parity.py`): each from-scratch optimizer, with and without weight decay, matches `torch.optim` after 25 steps to 1e-10.

```bash
pytest -q        # 23 tests, ~5 s on CPU
```

## Quickstart

```bash
git clone https://github.com/sumit-sah314/nn-from-scratch.git
cd nn-from-scratch
pip install -e ".[torch,dev]"          # or: pip install -r requirements.txt && pip install -e .

make smoke                              # offline end-to-end run on synthetic data
python scripts/train.py --framework numpy --arch cnn --dataset mnist --optimizer adam --epochs 5
python scripts/train.py --framework torch --arch cnn --dataset mnist --optimizer adam --epochs 5
python scripts/benchmark.py             # timing / memory / FLOPs table
python scripts/plot_results.py          # figures -> results/figures/
python scripts/cost_report.py --scale-epochs 20
```

Datasets download automatically on first use (no torchvision needed) and are cached as normalised `.npz` arrays in `data/`. Both frameworks train on exactly the same arrays.

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
scripts/                   # train.py, benchmark.py, plot_results.py, cost_report.py
tests/                     # gradcheck, torch parity, optimizer parity, training smoke tests
configs/pricing.json       # cloud hourly rates (edit me)
notebooks/                 # Colab quickstart
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

> Fill in after running `make all`. `results/benchmark.md` and `results/figures/` are generated automatically.

Sample from a sandbox CPU run (batch 64, multithreaded, synthetic inputs; your hardware will differ):

| arch | dataset | impl | train step (ms) | inference (µs/sample) | achieved GFLOP/s |
|---|---|---|---|---|---|
| MLP | MNIST | NumPy | 2.6 | 5.5 | 35 |
| MLP | MNIST | PyTorch CPU | 1.8 | 3.0 | 50 |
| CNN | MNIST | NumPy | 53 | 399 | 8.9 |
| CNN | MNIST | PyTorch CPU | 10.4 | 79 | 46 |
| CNN | CIFAR-10 | NumPy | 77 | 619 | 9.5 |
| CNN | CIFAR-10 | PyTorch CPU | 23.9 | 115 | 31 |

Early takeaway: the from-scratch MLP stays within ~1.5–2.5× of PyTorch because both are dominated by the same BLAS GEMMs, while the CNN gap (~3–5×) comes from im2col memory traffic and Python-level overhead that PyTorch avoids with fused convolution kernels (oneDNN on CPU, cuDNN on GPU).

## Project checklist

- [x] NumPy layers, loss, optimizers + gradient checks
- [x] PyTorch mirror + parity tests
- [x] Training / benchmark / plotting / cost scripts
- [ ] Full MNIST + CIFAR-10 runs (≥ 3 seeds) on CPU and Colab GPU
- [ ] Single-thread vs multi-thread benchmark comparison
- [ ] cProfile breakdown of the NumPy CNN (where does the time go?)
- [ ] Final report: plots, tables, discussion

## References

- I. Goodfellow, Y. Bengio, A. Courville. *Deep Learning*. MIT Press, 2016. Chapters 6, 8, 9.
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition, course notes (backprop, im2col, optimizers). https://cs231n.github.io/
- D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization*. ICLR 2015.
- G. Hinton. Lecture 6e, *RMSProp*, Neural Networks for Machine Learning (Coursera), 2012.
- PyTorch documentation, `torch.optim`. https://pytorch.org/docs/stable/optim.html

## License

MIT
