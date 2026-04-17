# Goal 2 Latency Decomposition Report

**Source file:** `results/single_benchmark_kv_on_20260415_011020.json`

## Setup

- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Device: `mps`
- Dtype: `float16`
- Generated tokens in example trial: `24`
- Effective bandwidth used for KV estimate: `120.0` GB/s

## First Token Breakdown

| Component | Time (ms) |
|---|---:|
| `embedding_lookup_ms` | 0.188 |
| `attention_qkv_ms` | 8.945 |
| `attention_softmax_ms` | 11.756 |
| `attention_projection_ms` | 3.531 |
| `kv_cache_rw_estimated_ms` | 0.006 |
| `mlp_total_ms` | 24.928 |
| `layernorm_total_ms` | 10.328 |
| `residual_ms` | 4.062 |
| `lm_head_ms` | 14.616 |
| `sampling_decoding_ms` | 0.333 |
| `framework_overhead_ms` | 5.430 |
| `total_token_ms` | 84.124 |

## Steady-State Average Breakdown

| Component | Avg Time (ms) |
|---|---:|
| `embedding_lookup_ms` | 0.179 |
| `attention_qkv_ms` | 7.221 |
| `attention_softmax_ms` | 14.481 |
| `attention_projection_ms` | 2.696 |
| `kv_cache_rw_estimated_ms` | 0.009 |
| `mlp_total_ms` | 17.353 |
| `layernorm_total_ms` | 7.234 |
| `residual_ms` | 4.090 |
| `lm_head_ms` | 2.291 |
| `sampling_decoding_ms` | 0.232 |
| `framework_overhead_ms` | 3.642 |
| `total_token_ms` | 59.430 |

## Per-Token Average Recomputed from Stored Token Rows

| Component | Avg Time (ms) |
|---|---:|
| `embedding_lookup_ms` | 0.180 |
| `attention_qkv_ms` | 7.293 |
| `attention_softmax_ms` | 14.367 |
| `attention_projection_ms` | 2.731 |
| `kv_cache_rw_estimated_ms` | 0.009 |
| `mlp_total_ms` | 17.669 |
| `layernorm_total_ms` | 7.363 |
| `residual_ms` | 4.089 |
| `lm_head_ms` | 2.805 |
| `sampling_decoding_ms` | 0.236 |
| `framework_overhead_ms` | 3.717 |
| `total_token_ms` | 60.459 |

## Example Per-Token Rows

| Token | Total (ms) | QKV | Softmax | MLP | Residual | Sampling | Overhead |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 84.124 | 8.945 | 11.756 | 24.928 | 4.062 | 0.333 | 5.430 |
| 2.000 | 63.477 | 7.653 | 14.761 | 18.558 | 4.484 | 0.302 | 4.524 |
| 3.000 | 59.922 | 7.344 | 15.302 | 17.159 | 3.854 | 0.251 | 3.135 |
| 4.000 | 65.625 | 7.177 | 15.200 | 18.545 | 4.327 | 0.230 | 6.574 |
| 5.000 | 58.326 | 8.290 | 11.762 | 17.505 | 4.599 | 0.234 | 3.190 |
