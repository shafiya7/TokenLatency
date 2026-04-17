# Generated Findings Report

## Configuration
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Device: mps
- Dtype: float16
- Prompt tokens: 257
- Output tokens: 24
- Warmups: 1
- Measured trials: 3
- KV cache enabled: True

## Benchmark Summary
- TTFT mean: 210.887 ms
- Per-token mean: 69.758 ms
- End-to-end mean: 1815.311 ms
- TTFT p95: 212.269 ms
- Per-token p95: 69.771 ms

## Outlier Handling
- Method: iqr
- IQR multiplier: 1.5
- Kept TTFT trials: 3
- Kept per-token trials: 3
- Kept E2E trials: 3

## First-Step Breakdown
- attention_projection_ms: 15.260 ms
- attention_qkv_ms: 25.579 ms
- attention_softmax_ms: 32.895 ms
- attention_total_ms: 73.734 ms
- decoder_layer_total_ms: 202.419 ms
- embedding_lookup_ms: 0.194 ms
- framework_overhead_ms: 3.072 ms
- is_first_token: 1.000 ms
- kv_cache_rw_estimated_ms: 0.048 ms
- layernorm_total_ms: 18.484 ms
- lm_head_ms: 6.527 ms
- mlp_projection_ms: 92.613 ms
- mlp_total_ms: 102.437 ms
- residual_ms: 7.765 ms
- sampling_decoding_ms: 0.226 ms
- token_index: 1.000 ms
- total_token_ms: 212.487 ms

## Steady-State Breakdown
- attention_projection_ms: 3.767 ms
- attention_qkv_ms: 9.792 ms
- attention_softmax_ms: 15.929 ms
- attention_total_ms: 29.488 ms
- decoder_layer_total_ms: 66.210 ms
- embedding_lookup_ms: 0.173 ms
- framework_overhead_ms: 2.447 ms
- is_first_token: 0.000 ms
- kv_cache_rw_estimated_ms: 0.051 ms
- layernorm_total_ms: 9.541 ms
- lm_head_ms: 0.694 ms
- mlp_projection_ms: 14.935 ms
- mlp_total_ms: 21.349 ms
- residual_ms: 5.832 ms
- sampling_decoding_ms: 0.198 ms
- token_index: 13.000 ms
- total_token_ms: 69.772 ms

## KV-Cache Traffic Estimate
- KV cache bytes per token: 22528
- First decode-step KV total bytes: 5812224
- Last decode-step KV total bytes: 6330368

## KV-Cache Comparison
- Avg per-token with KV cache: 56.347 ms
- Avg per-token without KV cache: 86.341 ms
- Speedup: 1.532x
- TTFT delta (no-cache - cache): 3.369 ms

## Scaling Snapshot
- meta-llama/Llama-3.2-1B-Instruct [auto] @ prompt 16 tokens: TTFT=83.07 ms, per-token=55.51 ms, E2E=1359.78 ms
- meta-llama/Llama-3.2-1B-Instruct [auto] @ prompt 32 tokens: TTFT=103.98 ms, per-token=59.10 ms, E2E=1463.23 ms
- meta-llama/Llama-3.2-1B-Instruct [auto] @ prompt 64 tokens: TTFT=137.38 ms, per-token=56.85 ms, E2E=1444.87 ms
- meta-llama/Llama-3.2-1B-Instruct [auto] @ prompt 128 tokens: TTFT=183.18 ms, per-token=60.30 ms, E2E=1570.00 ms
- meta-llama/Llama-3.2-1B-Instruct [auto] @ prompt 256 tokens: TTFT=233.26 ms, per-token=64.63 ms, E2E=1719.82 ms
- meta-llama/Llama-3.2-3B-Instruct [auto] @ prompt 16 tokens: TTFT=206.08 ms, per-token=99.17 ms, E2E=2486.96 ms
- meta-llama/Llama-3.2-3B-Instruct [auto] @ prompt 32 tokens: TTFT=172.95 ms, per-token=99.84 ms, E2E=2469.22 ms
- meta-llama/Llama-3.2-3B-Instruct [auto] @ prompt 64 tokens: TTFT=261.04 ms, per-token=108.73 ms, E2E=2761.85 ms
- meta-llama/Llama-3.2-3B-Instruct [auto] @ prompt 128 tokens: TTFT=312.83 ms, per-token=107.80 ms, E2E=2792.27 ms
- meta-llama/Llama-3.2-3B-Instruct [auto] @ prompt 256 tokens: TTFT=507.76 ms, per-token=109.67 ms, E2E=3030.08 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [auto] @ prompt 16 tokens: TTFT=86.16 ms, per-token=69.31 ms, E2E=1680.21 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [auto] @ prompt 32 tokens: TTFT=102.12 ms, per-token=67.24 ms, E2E=1648.54 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [auto] @ prompt 64 tokens: TTFT=117.48 ms, per-token=69.13 ms, E2E=1707.53 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [auto] @ prompt 128 tokens: TTFT=145.10 ms, per-token=68.16 ms, E2E=1712.84 ms
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 [auto] @ prompt 256 tokens: TTFT=210.89 ms, per-token=69.76 ms, E2E=1815.31 ms

## Interpretation
- TTFT is higher because the full prompt path runs before any token is emitted.
- Steady-state decode benefits from KV-cache reuse, but KV-cache traffic still grows with sequence length.
- Attention, MLP, and framework overhead should be interpreted together: if attention time grows with context and KV-cache traffic grows linearly, decode becomes increasingly memory-sensitive.
- Precision changes can shift both arithmetic cost and memory traffic, so scaling experiments should be interpreted jointly by model size, prompt length, and dtype.
