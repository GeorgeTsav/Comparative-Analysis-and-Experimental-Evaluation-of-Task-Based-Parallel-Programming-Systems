# Test Results Summary

Date: 2026-03-10

This report collects the elapsed times from the test runs, presents per-example tables, computes concise comparisons and speedup ratios, and provides interpretation and recommendations.

## Overview
- Collected elapsed times from logs under `MyExamples/test_results`.
- Included per-example tables for both frameworks (StarPU Python and `torcpy`) and for multiple process/worker configurations.

---

## Per-example results
 
### Consolidated per-example table

| Example | Name | Framework | Config | Elapsed time (s) |
|---|---|---|---:|---:|
| ex00 | Master-Worker | StarPU | 0 workers | 1.0041 |
| ex00 | Master-Worker | StarPU | 1 worker | 1.0068 |
| ex00 | Master-Worker | StarPU | 2 workers | 1.0039 |
| ex00 | Master-Worker | torcpy | 1p 1w | 4.0141 |
| ex00 | Master-Worker | torcpy | 1p 2w | 2.0232 |
| ex00 | Master-Worker | torcpy | 2p 1w | 2.0448 |
| ex00 | Master-Worker | torcpy | 2p 2w | 1.0436 |
| ex01 | Submit/Wait | StarPU | 0w | 0.0070 |
| ex01 | Submit/Wait | StarPU | 1w | 0.0067 |
| ex01 | Submit/Wait | StarPU | 2w | 0.0067 |
| ex01 | Submit/Wait | torcpy | 1p 1w | 0.0109 |
| ex01 | Submit/Wait | torcpy | 1p 2w | 0.0106 |
| ex01 | Submit/Wait | torcpy | 2p 1w | 0.0427 |
| ex01 | Submit/Wait | torcpy | 2p 2w | 0.0440 |
| ex02 | Map | StarPU | 0w | 0.0065 |
| ex02 | Map | StarPU | 1w | 0.0069 |
| ex02 | Map | StarPU | 2w | 0.0060 |
| ex02 | Map | torcpy | 1p 1w | 0.0108 |
| ex02 | Map | torcpy | 1p 2w | 0.0105 |
| ex02 | Map | torcpy | 2p 1w | 0.0426 |
| ex02 | Map | torcpy | 2p 2w | 0.0426 |
| ex04 | Fib | StarPU | 0w | 6.5011 |
| ex04 | Fib | StarPU | 1w | 6.5292 |
| ex04 | Fib | StarPU | 2w | 6.4215 |
| ex04 | Fib | torcpy | 1p 1w | 33.9466 |
| ex04 | Fib | torcpy | 1p 2w | 3.6765 |
| ex04 | Fib | torcpy | 2p 1w | 2.0765 |
| ex04 | Fib | torcpy | 2p 2w | 2.0933 |
| ex07 | Image processing | StarPU | 0w | 0.2373 |
| ex07 | Image processing | StarPU | 1w | 0.2680 |
| ex07 | Image processing | StarPU | 2w | 0.2540 |
| ex07 | Image processing | torcpy | 1p 1w | 0.4554 |
| ex07 | Image processing | torcpy | 1p 2w | 0.2632 |
| ex07 | Image processing | torcpy | 2p 1w | 0.3162 |
| ex07 | Image processing | torcpy | 2p 2w | 0.2208 |
| ex08 | CMA-ES | StarPU | 0w | 0.4268 |
| ex08 | CMA-ES | StarPU | 1w | 0.4360 |
| ex08 | CMA-ES | StarPU | 2w | 0.4283 |
| ex08 | CMA-ES | torcpy | 1p 1w | 0.6209 |
| ex08 | CMA-ES | torcpy | 1p 2w | 0.6179 |
| ex08 | CMA-ES | torcpy | 2p 1w | 3.0996 |
| ex08 | CMA-ES | torcpy | 2p 2w | 2.9564 |
| ex11 | kwargs example | StarPU | 0w | 1.0038 |
| ex11 | kwargs example | StarPU | 1w | 1.0041 |
| ex11 | kwargs example | StarPU | 2w | 1.0038 |
| ex11 | kwargs example | torcpy | 1p 1w | 4.0135 |
| ex11 | kwargs example | torcpy | 1p 2w | 2.0113 |
| ex11 | kwargs example | torcpy | 2p 1w | 2.0434 |
| ex11 | kwargs example | torcpy | 2p 2w | 1.0425 |

---

## Concise comparison (best times and ratios)

For each example the best (minimum) StarPU time and best torcpy time were selected. The ratio is defined as StarPU_time / torcpy_time; values > 1 imply torcpy is faster by that factor, values < 1 imply StarPU is faster (factor = 1/ratio).

| Example | StarPU best (s) | torcpy best (s) | Ratio (StarPU/torcpy) | Interpretation |
|---|---:|---:|---:|---|
| ex00 | 1.0039 | 1.0436 | 0.96 | StarPU slightly faster (~1.04× slower for torcpy) |
| ex01 | 0.0067 | 0.0106 | 0.63 | StarPU ~1.58× faster |
| ex02 | 0.0060 | 0.0105 | 0.57 | StarPU ~1.75× faster |
| ex04 | 6.4215 | 2.0765 | 3.09 | torcpy ~3.09× faster (CPU-bound benefit) |
| ex07 | 0.2373 | 0.2208 | 1.08 | torcpy slightly faster (~1.08×) |
| ex08 | 0.4268 | 0.6179 | 0.69 | StarPU ~1.45× faster |
| ex11 | 1.0038 | 1.0425 | 0.96 | StarPU slightly faster |


## Detailed interpretation & takeaways

- Overhead and granularity:
  - Small, fine-grained tasks (ex01, ex02) are dominated by framework overhead. StarPU's in-process scheduling shows lower overhead than torcpy's distributed MPI-based approach, so StarPU wins on very small tasks.
  - torcpy shows large variability across configurations; choosing an appropriate combination of MPI processes (`p`) and internal workers (`w`) is crucial.

- CPU-bound tasks:
  - For compute-heavy Python tasks (ex04 fib), multi-process torcpy (2p configurations) is significantly faster because it avoids the Python GIL, permitting parallel CPU work across processes. That yields ~3× speedup over StarPU for ex04.

- Mixed workloads:
  - Image processing (ex07) and CMA-ES (ex08) show mixed results. torcpy wins for ex07 in its best config (2p 2w), while StarPU is faster for ex08 in these runs. These outcomes depend on per-task cost, communication overhead and process startup/serialization cost.

- Stability across worker counts:
  - StarPU times are relatively stable across 0/1/2 worker settings in these runs; torcpy can be much slower in single-process configs but can outperform when configured for multi-process parallelism.

## Recommendations

- Use torcpy with multiple MPI processes for CPU-bound, pure-Python workloads to bypass the GIL.
- Use StarPU for many small tasks or low-overhead in-process scheduling; consider batching fine-grained work when using torcpy.
- When comparing frameworks, run per-example tuning sweeping `p` and `w` (processes/workers) to find the operating point for your hardware and workload.
