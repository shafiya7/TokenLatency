# Goal 3 Scaling Analysis

**Source file:** `results/scaling_20260416_233056.json`

## Experiment Coverage

- Models: 3
- Precisions: auto
- Prompt lengths: 16, 32, 64, 128, 256

## Scaling Table

| Model | Precision | Prompt Tokens | TTFT Mean (ms) | Per-token Mean (ms) | E2E Mean (ms) |
|---|---|---:|---:|---:|---:|
| TinyLlama-1.1B-Chat-v1.0 | auto | 16 | 86.158 | 69.307 | 1680.211 |
| TinyLlama-1.1B-Chat-v1.0 | auto | 32 | 102.118 | 67.236 | 1648.535 |
| TinyLlama-1.1B-Chat-v1.0 | auto | 64 | 117.482 | 69.133 | 1707.535 |
| TinyLlama-1.1B-Chat-v1.0 | auto | 128 | 145.100 | 68.162 | 1712.838 |
| TinyLlama-1.1B-Chat-v1.0 | auto | 256 | 210.887 | 69.758 | 1815.311 |
| Llama-3.2-1B-Instruct | auto | 16 | 83.072 | 55.509 | 1359.778 |
| Llama-3.2-1B-Instruct | auto | 32 | 103.976 | 59.098 | 1463.235 |
| Llama-3.2-1B-Instruct | auto | 64 | 137.379 | 56.847 | 1444.868 |
| Llama-3.2-1B-Instruct | auto | 128 | 183.179 | 60.297 | 1570.004 |
| Llama-3.2-1B-Instruct | auto | 256 | 233.262 | 64.633 | 1719.821 |
| Llama-3.2-3B-Instruct | auto | 16 | 206.079 | 99.169 | 2486.964 |
| Llama-3.2-3B-Instruct | auto | 32 | 172.946 | 99.838 | 2469.224 |
| Llama-3.2-3B-Instruct | auto | 64 | 261.042 | 108.731 | 2761.849 |
| Llama-3.2-3B-Instruct | auto | 128 | 312.831 | 107.802 | 2792.271 |
| Llama-3.2-3B-Instruct | auto | 256 | 507.761 | 109.666 | 3030.076 |

## Dominant Trends

- TinyLlama-1.1B-Chat-v1.0 [auto] from 16→256 tokens: TTFT changed by 124.729 ms and per-token latency changed by 0.451 ms.
- Llama-3.2-1B-Instruct [auto] from 16→256 tokens: TTFT changed by 150.190 ms and per-token latency changed by 9.124 ms.
- Llama-3.2-3B-Instruct [auto] from 16→256 tokens: TTFT changed by 301.682 ms and per-token latency changed by 10.497 ms.

## Interpretation

- Sequence length typically affects TTFT first through the prompt pass, then increasingly affects steady-state decode as the KV-cache grows.
- Model size changes both TTFT and per-token latency because larger variants have more layers and wider hidden states, increasing attention and MLP cost.
- Precision differences matter most when the backend truly accelerates lower precision or reduces memory traffic; otherwise, gains may be modest.
- The strongest bottleneck should be confirmed with the Goal 2 breakdown. If attention grows faster than MLP as context increases, that points to decode-time memory pressure and KV-cache traffic.
