| arch | dataset | impl | device | params | fwd_mflops_per_sample | train_step_ms | infer_us_per_sample | achieved_gflops | peak_mem_mb |
|---|---|---|---|---|---|---|---|---|---|
| mlp | mnist | numpy | cpu | 235146 | 0.47 | 0.43 | 0.568 | 210 | 3.95 |
| mlp | mnist | torch | cpu | 235146 | 0.47 | 0.344 | 0.428 | 262 | 5.06 |
| cnn | mnist | numpy | cpu | 206922 | 2.47 | 15.7 | 142 | 30 | 23.5 |
| cnn | mnist | torch | cpu | 206922 | 2.47 | 8.1 | 61.4 | 58.3 | 53.6 |
| mlp | cifar10 | numpy | cpu | 820874 | 1.64 | 1.44 | 1.37 | 219 | 15.1 |
| mlp | cifar10 | torch | cpu | 820874 | 1.64 | 1.21 | 1.3 | 261 | 16.2 |
| cnn | cifar10 | numpy | cpu | 268650 | 3.81 | 22.1 | 200 | 33 | 36.5 |
| cnn | cifar10 | torch | cpu | 268650 | 3.81 | 11.1 | 91.4 | 65.5 | 69.9 |
