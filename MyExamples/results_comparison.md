# Framework Timing Results Comparison

Date: 2026-03-26
Source: `MyExamples/test_results/*.log`

This report summarizes the latest run timings for `torcpy` and `StarPU Python` in easy-to-read tables.

## 1) Consolidated Timing Table

| Example | Workload | Framework | Config | Elapsed Time (s) |
|---|---|---|---|---:|
| app00 | CMA-ES + PDE | StarPU | 0w | 451.8759 |
| app00 | CMA-ES + PDE | StarPU | 1w | 450.8932 |
| app00 | CMA-ES + PDE | StarPU | 2w | 469.2443 |
| app00 | CMA-ES + PDE | torcpy | 1p 1w | 104.8515 |
| app00 | CMA-ES + PDE | torcpy | 1p 2w | 67.4655 |
| app00 | CMA-ES + PDE | torcpy | 2p 1w | 60.0608 |
| app00 | CMA-ES + PDE | torcpy | 2p 2w | 41.7412 |
| app01 | Heavy image processing | StarPU | 0w | 42.806228 |
| app01 | Heavy image processing | StarPU | 1w | 42.753906 |
| app01 | Heavy image processing | StarPU | 2w | 42.495097 |
| app01 | Heavy image processing | torcpy | 1p 1w | 104.670402 |
| app01 | Heavy image processing | torcpy | 1p 2w | 62.696408 |
| app01 | Heavy image processing | torcpy | 2p 1w | 63.061268 |
| app01 | Heavy image processing | torcpy | 2p 2w | 39.167607 |
| app02 | Heavy Monte Carlo | StarPU | 0w | 11.3643 |
| app02 | Heavy Monte Carlo | StarPU | 1w | 11.3345 |
| app02 | Heavy Monte Carlo | StarPU | 2w | 11.1876 |
| app02 | Heavy Monte Carlo | torcpy | 1p 1w | 57.0115 |
| app02 | Heavy Monte Carlo | torcpy | 1p 2w | 37.4608 |
| app02 | Heavy Monte Carlo | torcpy | 2p 1w | 35.3252 |
| app02 | Heavy Monte Carlo | torcpy | 2p 2w | 21.7681 |
| ex00 | Master-worker | StarPU | 0w | 1.0029 |
| ex00 | Master-worker | StarPU | 1w | 1.0022 |
| ex00 | Master-worker | StarPU | 2w | 1.0022 |
| ex00 | Master-worker | torcpy | 1p 1w | 4.0130 |
| ex00 | Master-worker | torcpy | 1p 2w | 2.0115 |
| ex00 | Master-worker | torcpy | 2p 1w | 2.0424 |
| ex00 | Master-worker | torcpy | 2p 2w | 1.0419 |
| ex01 | Submit/wait | StarPU | 0w | 0.0062 |
| ex01 | Submit/wait | StarPU | 1w | 0.0053 |
| ex01 | Submit/wait | StarPU | 2w | 0.0051 |
| ex01 | Submit/wait | torcpy | 1p 1w | 0.0104 |
| ex01 | Submit/wait | torcpy | 1p 2w | 0.0103 |
| ex01 | Submit/wait | torcpy | 2p 1w | 0.0420 |
| ex01 | Submit/wait | torcpy | 2p 2w | 0.0424 |
| ex02 | Map | StarPU | 0w | 0.0045 |
| ex02 | Map | StarPU | 1w | 0.0043 |
| ex02 | Map | StarPU | 2w | 0.0054 |
| ex02 | Map | torcpy | 1p 1w | 0.0104 |
| ex02 | Map | torcpy | 1p 2w | 0.0104 |
| ex02 | Map | torcpy | 2p 1w | 0.0421 |
| ex02 | Map | torcpy | 2p 2w | 0.0420 |
| ex03 | Fibonacci | StarPU | 0w | 4.1543 |
| ex03 | Fibonacci | StarPU | 1w | 3.5371 |
| ex03 | Fibonacci | StarPU | 2w | 3.6653 |
| ex03 | Fibonacci | torcpy | 1p 1w | 17.5201 |
| ex03 | Fibonacci | torcpy | 1p 2w | 2.1198 |
| ex03 | Fibonacci | torcpy | 2p 1w | 1.1263 |
| ex03 | Fibonacci | torcpy | 2p 2w | 1.1242 |
| ex04 | Image processing | StarPU | 0w | 0.2423 |
| ex04 | Image processing | StarPU | 1w | 0.2217 |
| ex04 | Image processing | StarPU | 2w | 0.2244 |
| ex04 | Image processing | torcpy | 1p 1w | 0.2795 |
| ex04 | Image processing | torcpy | 1p 2w | 0.1620 |
| ex04 | Image processing | torcpy | 2p 1w | 0.2268 |
| ex04 | Image processing | torcpy | 2p 2w | 0.1445 |
| ex05 | CMA-ES | StarPU | 0w | 0.2767 |
| ex05 | CMA-ES | StarPU | 1w | 0.2757 |
| ex05 | CMA-ES | StarPU | 2w | 0.2828 |
| ex05 | CMA-ES | torcpy | 1p 1w | 0.5480 |
| ex05 | CMA-ES | torcpy | 1p 2w | 0.5401 |
| ex05 | CMA-ES | torcpy | 2p 1w | 3.1990 |
| ex05 | CMA-ES | torcpy | 2p 2w | 2.9734 |
| ex06 | kwargs (torcpy only) | torcpy | 1p 1w | 4.0138 |
| ex06 | kwargs (torcpy only) | torcpy | 1p 2w | 2.0119 |
| ex06 | kwargs (torcpy only) | torcpy | 2p 1w | 2.0327 |
| ex06 | kwargs (torcpy only) | torcpy | 2p 2w | 1.0427 |
| ex06 | kwargs (starpupy only) | StarPU | 0w | 1.0027 |
| ex06 | kwargs (starpupy only) | StarPU | 1w | 1.0025 |
| ex06 | kwargs (starpupy only) | StarPU | 2w | 1.0023 |
| ex_monitor | Monitor workload | StarPU | 0w | 16.1140 |
| ex_monitor | Monitor workload | StarPU | 1w | 16.1125 |
| ex_monitor | Monitor workload | StarPU | 2w | 16.0867 |
| ex_monitor | Monitor workload | torcpy | 1p 1w | 64.0114 |
| ex_monitor | Monitor workload | torcpy | 1p 2w | 32.0229 |
| ex_monitor | Monitor workload | torcpy | 2p 1w | 32.0420 |
| ex_monitor | Monitor workload | torcpy | 2p 2w | 16.0618 |

## 2) Best Configuration Comparison (per common example)

For each example that exists in both frameworks, this table shows the fastest timing seen for each framework and who wins.

| Example | StarPU Best (s) | torcpy Best (s) | Faster Framework | Speedup |
|---|---:|---:|---|---:|
| app00 | 450.8932 | 41.7412 | torcpy | 10.80x |
| app01 | 42.495097 | 39.167607 | torcpy | 1.08x |
| app02 | 11.1876 | 21.7681 | StarPU | 1.95x |
| ex00 | 1.0022 | 1.0419 | StarPU | 1.04x |
| ex01 | 0.0051 | 0.0103 | StarPU | 2.02x |
| ex02 | 0.0043 | 0.0104 | StarPU | 2.42x |
| ex03 | 3.5371 | 1.1242 | torcpy | 3.15x |
| ex04 | 0.2217 | 0.1445 | torcpy | 1.53x |
| ex05 | 0.2757 | 0.5401 | StarPU | 1.96x |
| ex06 | 1.0023 | 1.0427 | StarPU | 1.04x |
| ex_monitor | 16.0867 | 16.0618 | torcpy | 1.00x |

## 3) Interpretation

- Fine-grained overhead-dominated workloads (`ex01`, `ex02`) favor StarPU clearly.
- Heavy CPU workloads and process-level scaling (`app00`, `ex03`, and often `ex04`) favor torcpy in its multi-process configurations.
- For `app01` (heavy image pipeline), both are close, with a slight torcpy advantage in best configuration.
- `ex_monitor` is essentially tied at best config (`~16.1s` each), showing comparable throughput when torcpy is configured as `2p 2w`.

