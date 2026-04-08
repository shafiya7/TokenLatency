# Goal 3 Scaling Analysis

**Source file:** `results/scaling_20260406_001038.json`

## Experiment Coverage

- Models: 1
- Precisions: float16, float32
- Prompt lengths: 64, 128, 256

## Scaling Table

| Model | Precision | Prompt Tokens | TTFT Mean (ms) | Per-token Mean (ms) | E2E Mean (ms) |
|---|---|---:|---:|---:|---:|
| TinyLlama-1.1B-Chat-v1.0 | float16 | 64 | 142.311 | 72.097 | 1800.539 |
| TinyLlama-1.1B-Chat-v1.0 | float16 | 128 | 146.243 | 70.790 | 1774.402 |
| TinyLlama-1.1B-Chat-v1.0 | float16 | 256 | 209.934 | 73.117 | 1891.615 |
| TinyLlama-1.1B-Chat-v1.0 | float32 | 64 | 118.087 | 72.621 | 1788.367 |
| TinyLlama-1.1B-Chat-v1.0 | float32 | 128 | 147.041 | 77.412 | 1927.511 |
| TinyLlama-1.1B-Chat-v1.0 | float32 | 256 | 220.567 | 80.765 | 2078.153 |

## Dominant Trends

- TinyLlama-1.1B-Chat-v1.0 [float16] from 64→256 tokens: TTFT changed by 67.623 ms and per-token latency changed by 1.020 ms.
- TinyLlama-1.1B-Chat-v1.0 [float32] from 64→256 tokens: TTFT changed by 102.480 ms and per-token latency changed by 8.144 ms.

## Interpretation

- Sequence length typically affects TTFT first through the prompt pass, then increasingly affects steady-state decode as the KV-cache grows.
- Model size changes both TTFT and per-token latency because larger variants have more layers and wider hidden states, increasing attention and MLP cost.
- Precision differences matter most when the backend truly accelerates lower precision or reduces memory traffic; otherwise, gains may be modest.
- The strongest bottleneck should be confirmed with the Goal 2 breakdown. If attention grows faster than MLP as context increases, that points to decode-time memory pressure and KV-cache traffic.
