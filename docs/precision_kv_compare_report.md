# Precision + KV Cache Comparison

| Precision | KV Cache | TTFT Mean (ms) | Per-token Mean (ms) | E2E Mean (ms) |
|---|---|---:|---:|---:|
| float32 | off | 89.781 | 104.142 | 2485.052 |
| float32 | on | 86.425 | 62.492 | 1523.746 |
| float16 | off | 85.671 | 89.108 | 2135.166 |
| float16 | on | 75.440 | 51.797 | 1266.776 |

## Pairwise KV-cache benefit by precision

- float32: with KV cache gave 1.666x per-token speedup, 41.650 ms lower per-token latency, 3.356 ms TTFT delta, and 961.305 ms E2E delta.
- float16: with KV cache gave 1.720x per-token speedup, 37.311 ms lower per-token latency, 10.232 ms TTFT delta, and 868.390 ms E2E delta.
