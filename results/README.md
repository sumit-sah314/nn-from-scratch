# results/

Generated files land here (git-ignored by default; commit final ones deliberately):

| path | produced by |
|---|---|
| `runs/*.json` | `scripts/train.py`: config, per-step loss, per-epoch train/test loss & accuracy, epoch times |
| `benchmark.csv`, `benchmark.md` | `scripts/benchmark.py`: params, FLOPs, step/inference time, peak memory |
| `figures/*.png` | `scripts/plot_results.py` |
| `cost.csv` | `scripts/cost_report.py` |
