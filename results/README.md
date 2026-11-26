# results/

Generated files land here. Everything under `results/` except this file is git-ignored (`results/*` in `.gitignore`), so nothing new is committed by accident. Files already in git stay tracked. To publish a new one, add it explicitly, for example `git add -f results/cost.csv`.

| path | produced by |
|---|---|
| `runs/*.json` | `scripts/train.py`: config, per-step loss, per-epoch train/test loss & accuracy, epoch times |
| `runs/dashboard/<timestamp>/` | the dashboard: `history.json`, weights, run figures |
| `smoke/` | `make smoke` (synthetic data) |
| `benchmark.csv`, `benchmark.md` | `scripts/benchmark.py`: params, FLOPs, step/inference time, peak memory |
| `figures/*.png` | `scripts/plot_results.py` |
| `cost.csv` | `scripts/cost_report.py` |
