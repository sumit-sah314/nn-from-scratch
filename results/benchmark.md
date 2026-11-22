| arch | dataset | impl | device | params | fwd_mflops_per_sample | train_step_ms | infer_us_per_sample | achieved_gflops | peak_mem_mb |
|---|---|---|---|---|---|---|---|---|---|
| mlp | mnist | numpy | cpu | 235146 | 0.47 | 0.429 | 0.57 | 210 | 3.95 |
| mlp | mnist | torch | cpu | 235146 | 0.47 | 0.44 | 0.509 | 205 | 5.06 |
| cnn | mnist | numpy | cpu | 206922 | 2.47 | 15.5 | 141 | 30.4 | 23.5 |
| cnn | mnist | torch | cpu | 206922 | 2.47 | 6.5 | 20.1 | 72.6 | 53.6 |
| mlp | cifar10 | numpy | cpu | 820874 | 1.64 | 1.47 | 1.37 | 215 | 15.1 |
| mlp | cifar10 | torch | cpu | 820874 | 1.64 | 0.909 | 1.39 | 347 | 16.2 |
| cnn | cifar10 | numpy | cpu | 268650 | 3.81 | 22 | 198 | 33.2 | 36.5 |
| cnn | cifar10 | torch | cpu | 268650 | 3.81 | 9.57 | 46.3 | 76.2 | 69.9 |
