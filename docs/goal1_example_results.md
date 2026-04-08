# Goal 1 Benchmark Summary

**Source file:** `results/single_benchmark_kv_on_20260406_001037.json`

## Benchmark Setup

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Device: `mps`
- Runtime dtype: `float32`
- Prompt length (chars): `1148`
- Max new tokens: `24`
- Warm-up runs: `1`
- Measured trials: `3`
- Outlier method: `iqr`
- IQR multiplier: `1.5`
- KV cache enabled: `True`
- Sampling enabled: `False`

## Example Result Table

| Metric | Raw Mean (ms) | Raw Std (ms) | Filtered Mean (ms) | Filtered Std (ms) | Trials Kept |
|---|---:|---:|---:|---:|---:|
| First-token latency (TTFT) | 220.567 | 2.909 | 220.567 | - | 3.000 |
| Steady-state per-token latency | 80.765 | 9.278 | 80.765 | - | 3.000 |
| End-to-end response time | 2078.153 | 214.956 | 2078.153 | - | 3.000 |

## Additional Summary Statistics

| Metric | Raw Median (ms) | Raw Min (ms) | Raw Max (ms) | Raw P95 (ms) |
|---|---:|---:|---:|---:|
| First-token latency (TTFT) | 221.941 | 217.225 | 222.535 | 222.476 |
| Steady-state per-token latency | 76.718 | 74.197 | 91.379 | 89.913 |
| End-to-end response time | 1987.039 | 1923.761 | 2323.659 | 2289.997 |

## Outlier Bounds Used

| Metric | Lower Bound (ms) | Upper Bound (ms) |
|---|---:|---:|
| First-token latency (TTFT) | - | - |
| Steady-state per-token latency | - | - |
| End-to-end response time | - | - |

## Measurement Notes

- TTFT is measured from the start of timed model execution to completion of first-token selection.
- Steady-state per-token latency is computed only for decode steps after the first generated token.
- End-to-end latency is computed as TTFT plus the sum of steady-state token latencies.
- Warm-up runs are excluded from reported statistics.
- Outliers are filtered using IQR-based bounds before computing filtered statistics.
- Tokenization and host-side prompt preparation are excluded from the timed region.
- Raw standard deviation is computed from all measured trials.
- Filtered standard deviation is recomputed from the subset of trials that fall within the recorded IQR bounds.
