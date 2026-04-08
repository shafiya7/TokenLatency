# Goal 2 Latency Decomposition Report

**Source file:** `results/single_benchmark_kv_on_20260406_001037.json`

## Setup

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Device: `mps`
- Dtype: `float32`
- Generated tokens in example trial: `24`
- Effective bandwidth used for KV estimate: `120.0` GB/s

## First Token Breakdown

| Component | Time (ms) |
|---|---:|
| `embedding_lookup_ms` | 0.204 |
| `attention_qkv_ms` | 25.489 |
| `attention_softmax_ms` | 30.825 |
| `attention_projection_ms` | 16.336 |
| `kv_cache_rw_estimated_ms` | 0.097 |
| `mlp_total_ms` | 111.495 |
| `layernorm_total_ms` | 13.888 |
| `residual_ms` | 7.623 |
| `lm_head_ms` | 7.215 |
| `sampling_decoding_ms` | 0.287 |
| `framework_overhead_ms` | 3.767 |
| `total_token_ms` | 217.225 |

## Steady-State Average Breakdown

| Component | Avg Time (ms) |
|---|---:|
| `embedding_lookup_ms` | 0.180 |
| `attention_qkv_ms` | 10.758 |
| `attention_softmax_ms` | 13.748 |
| `attention_projection_ms` | 4.495 |
| `kv_cache_rw_estimated_ms` | 0.101 |
| `mlp_total_ms` | 27.129 |
| `layernorm_total_ms` | 7.911 |
| `residual_ms` | 5.833 |
| `lm_head_ms` | 1.189 |
| `sampling_decoding_ms` | 0.197 |
| `framework_overhead_ms` | 2.656 |
| `total_token_ms` | 74.197 |

## Per-Token Average Recomputed from Stored Token Rows

| Component | Avg Time (ms) |
|---|---:|
| `embedding_lookup_ms` | 0.181 |
| `attention_qkv_ms` | 11.371 |
| `attention_softmax_ms` | 14.460 |
| `attention_projection_ms` | 4.988 |
| `kv_cache_rw_estimated_ms` | 0.101 |
| `mlp_total_ms` | 30.645 |
| `layernorm_total_ms` | 8.160 |
| `residual_ms` | 5.907 |
| `lm_head_ms` | 1.440 |
| `sampling_decoding_ms` | 0.201 |
| `framework_overhead_ms` | 2.703 |
| `total_token_ms` | 80.157 |

## Example Per-Token Rows

| Token | Total (ms) | QKV | Softmax | MLP | Residual | Sampling | Overhead |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 217.225 | 25.489 | 30.825 | 111.495 | 7.623 | 0.287 | 3.767 |
| 2.000 | 76.465 | 10.522 | 14.176 | 27.282 | 6.087 | 0.212 | 4.518 |
| 3.000 | 75.519 | 10.713 | 14.946 | 27.158 | 5.731 | 0.183 | 2.974 |
| 4.000 | 74.444 | 10.809 | 13.579 | 27.094 | 5.791 | 0.183 | 3.046 |
| 5.000 | 73.509 | 10.776 | 13.733 | 27.150 | 5.825 | 0.178 | 2.038 |
