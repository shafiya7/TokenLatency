# Generated Findings Report

## Configuration
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Device: mps
- Dtype: float32
- Prompt tokens: 257
- Output tokens: 24
- Warmups: 1
- Measured trials: 3
- KV cache enabled: True

## Benchmark Summary
- TTFT mean: 220.567 ms
- Per-token mean: 80.765 ms
- End-to-end mean: 2078.153 ms
- TTFT p95: 222.476 ms
- Per-token p95: 89.913 ms

## Outlier Handling
- Method: iqr
- IQR multiplier: 1.5
- Kept TTFT trials: 3
- Kept per-token trials: 3
- Kept E2E trials: 3

## First-Step Breakdown
- attention_projection_ms: 16.336 ms
- attention_qkv_ms: 25.489 ms
- attention_softmax_ms: 30.825 ms
- attention_total_ms: 72.649 ms
- decoder_layer_total_ms: 205.655 ms
- embedding_lookup_ms: 0.204 ms
- framework_overhead_ms: 3.767 ms
- is_first_token: 1.000 ms
- kv_cache_rw_estimated_ms: 0.097 ms
- layernorm_total_ms: 13.888 ms
- lm_head_ms: 7.215 ms
- mlp_projection_ms: 101.660 ms
- mlp_total_ms: 111.495 ms
- residual_ms: 7.623 ms
- sampling_decoding_ms: 0.287 ms
- token_index: 1.000 ms
- total_token_ms: 217.225 ms

## Steady-State Breakdown
- attention_projection_ms: 4.495 ms
- attention_qkv_ms: 10.758 ms
- attention_softmax_ms: 13.748 ms
- attention_total_ms: 29.001 ms
- decoder_layer_total_ms: 69.874 ms
- embedding_lookup_ms: 0.180 ms
- framework_overhead_ms: 2.656 ms
- is_first_token: 0.000 ms
- kv_cache_rw_estimated_ms: 0.101 ms
- layernorm_total_ms: 7.911 ms
- lm_head_ms: 1.189 ms
- mlp_projection_ms: 20.751 ms
- mlp_total_ms: 27.129 ms
- residual_ms: 5.833 ms
- sampling_decoding_ms: 0.197 ms
- token_index: 13.000 ms
- total_token_ms: 74.197 ms

## KV-Cache Traffic Estimate
- KV cache bytes per token: 45056
- First decode-step KV total bytes: 11624448
- Last decode-step KV total bytes: 12660736

## KV-Cache Comparison
- Avg per-token with KV cache: 65.588 ms
- Avg per-token without KV cache: 103.092 ms
- Speedup: 1.572x
- TTFT delta (no-cache - cache): 30.543 ms

## Scaling Snapshot
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [float16] @ prompt 64 tokens: TTFT=142.31 ms, per-token=72.10 ms, E2E=1800.54 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [float16] @ prompt 128 tokens: TTFT=146.24 ms, per-token=70.79 ms, E2E=1774.40 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [float16] @ prompt 256 tokens: TTFT=209.93 ms, per-token=73.12 ms, E2E=1891.62 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [float32] @ prompt 64 tokens: TTFT=118.09 ms, per-token=72.62 ms, E2E=1788.37 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [float32] @ prompt 128 tokens: TTFT=147.04 ms, per-token=77.41 ms, E2E=1927.51 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [float32] @ prompt 256 tokens: TTFT=220.57 ms, per-token=80.76 ms, E2E=2078.15 ms

## Interpretation
- TTFT is higher because the full prompt path runs before any token is emitted.
- Steady-state decode benefits from KV-cache reuse, but KV-cache traffic still grows with sequence length.
- Attention, MLP, and framework overhead should be interpreted together: if attention time grows with context and KV-cache traffic grows linearly, decode becomes increasingly memory-sensitive.
- Precision changes can shift both arithmetic cost and memory traffic, so scaling experiments should be interpreted jointly by model size, prompt length, and dtype.
