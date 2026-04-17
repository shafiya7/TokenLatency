# Goal 1 Benchmark Summary

**Source file:** `results/single_benchmark_kv_on_20260416_232200.json`

## Benchmark Setup

- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Device: `mps`
- Runtime dtype: `float16`
- Prompt length (chars): `88`
- Max new tokens: `24`
- Warm-up runs: `2`
- Measured trials: `5`
- Outlier method: `iqr`
- IQR multiplier: `1.5`
- KV cache enabled: `True`
- Sampling enabled: `False`

## Example Result Table

| Metric | Raw Mean (ms) | Raw Std (ms) | Filtered Mean (ms) | Filtered Std (ms) | Trials Kept |
|---|---:|---:|---:|---:|---:|
| First-token latency (TTFT) | 115.338 | 47.919 | 115.338 | 47.919 | 5.000 |
| Steady-state per-token latency | 57.026 | 1.955 | 57.026 | 1.955 | 5.000 |
| End-to-end response time | 1426.927 | 85.953 | 1426.927 | 85.953 | 5.000 |

## Additional Summary Statistics

| Metric | Raw Median (ms) | Raw Min (ms) | Raw Max (ms) | Raw P95 (ms) |
|---|---:|---:|---:|---:|
| First-token latency (TTFT) | 84.498 | 81.140 | 188.253 | 178.742 |
| Steady-state per-token latency | 56.792 | 54.572 | 59.645 | 59.344 |
| End-to-end response time | 1388.327 | 1339.654 | 1525.492 | 1522.902 |

## Outlier Bounds Used

| Metric | Lower Bound (ms) | Upper Bound (ms) |
|---|---:|---:|
| First-token latency (TTFT) | -5.789 | 228.591 |
| Steady-state per-token latency | 52.732 | 61.386 |
| End-to-end response time | 1152.734 | 1728.427 |

## Measurement Notes

- TTFT is measured from the start of timed model execution to completion of first-token selection.
- Steady-state per-token latency is computed only for decode steps after the first generated token.
- End-to-end latency is computed as TTFT plus the sum of steady-state token latencies.
- Warm-up runs are excluded from reported statistics.
- Outliers are filtered using IQR-based bounds before computing filtered statistics.
- Tokenization and host-side prompt preparation are excluded from the timed region.
- Raw standard deviation is computed from all measured trials.
- Filtered standard deviation is recomputed from the subset of trials that fall within the recorded IQR bounds.
