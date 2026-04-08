# KV-Cache Analytical Report

## Purpose
This report uses an analytical KV-cache model to explain how KV memory footprint scales with:
- model architecture
- sequence length
- KV precision

It supports:
- **Goal 4**: architectural bottleneck analysis
- **Goal 5**: optimization proposal for reduced-precision KV-cache

## Key Equations
For one token, KV storage added across all layers is approximated as:

`KV bytes per token = num_layers × num_kv_heads × head_dim × (bytes(K) + bytes(V))`

For a sequence of length `L`:

`Total KV cache bytes = L × KV bytes per token`

For one decode step, an approximate lower-bound for KV transfer time is:

`transfer time ≈ KV bytes read / memory bandwidth`

## Model: meta-llama/Llama-3.2-1B-Instruct

### FP16 KV-cache footprint by sequence length

| Seq Len | KV Bytes/Token | Total KV Size | Decode Read / Step | Estimated Transfer @ 100 GB/s |
| --- | --- | --- | --- | --- |
| 16 | 32768 | 0.500 MB | 0.500 MB | 0.0049 ms |
| 32 | 32768 | 1.000 MB | 1.000 MB | 0.0098 ms |
| 64 | 32768 | 2.000 MB | 2.000 MB | 0.0195 ms |
| 128 | 32768 | 4.000 MB | 4.000 MB | 0.0391 ms |
| 256 | 32768 | 8.000 MB | 8.000 MB | 0.0781 ms |
| 512 | 32768 | 16.000 MB | 16.000 MB | 0.1562 ms |

### Cache-capacity thresholds

| Cache Size | KV Exceeds Cache At Seq Len |
| --- | --- |
| 256KB | 8 |
| 1MB | 32 |
| 8MB | 256 |
| 32MB | 1024 |
| 64MB | 2048 |

### Precision comparison

| Precision | Seq Len | KV Bytes/Token | Total KV Size |
| --- | --- | --- | --- |
| fp16 | 16 | 32768 | 0.500 MB |
| fp16 | 32 | 32768 | 1.000 MB |
| fp16 | 64 | 32768 | 2.000 MB |
| fp16 | 128 | 32768 | 4.000 MB |
| fp16 | 256 | 32768 | 8.000 MB |
| fp16 | 512 | 32768 | 16.000 MB |
| int8 | 16 | 16384 | 0.250 MB |
| int8 | 32 | 16384 | 0.500 MB |
| int8 | 64 | 16384 | 1.000 MB |
| int8 | 128 | 16384 | 2.000 MB |
| int8 | 256 | 16384 | 4.000 MB |
| int8 | 512 | 16384 | 8.000 MB |
| int4 | 16 | 8192 | 0.125 MB |
| int4 | 32 | 8192 | 0.250 MB |
| int4 | 64 | 8192 | 0.500 MB |
| int4 | 128 | 8192 | 1.000 MB |
| int4 | 256 | 8192 | 2.000 MB |
| int4 | 512 | 8192 | 4.000 MB |

### Interpretation
- KV-cache footprint grows **linearly with sequence length**.
- Decode-time KV reads also grow with sequence length, which increases memory traffic.
- Lower-precision KV formats reduce both **storage size** and **per-step read volume**.
- Once KV-cache exceeds smaller on-chip caches, accesses are more likely to spill to slower memory levels.

## Model: meta-llama/Llama-3.2-3B-Instruct

### FP16 KV-cache footprint by sequence length

| Seq Len | KV Bytes/Token | Total KV Size | Decode Read / Step | Estimated Transfer @ 100 GB/s |
| --- | --- | --- | --- | --- |
| 16 | 114688 | 1.750 MB | 1.750 MB | 0.0171 ms |
| 32 | 114688 | 3.500 MB | 3.500 MB | 0.0342 ms |
| 64 | 114688 | 7.000 MB | 7.000 MB | 0.0684 ms |
| 128 | 114688 | 14.000 MB | 14.000 MB | 0.1367 ms |
| 256 | 114688 | 28.000 MB | 28.000 MB | 0.2734 ms |
| 512 | 114688 | 56.000 MB | 56.000 MB | 0.5469 ms |

### Cache-capacity thresholds

| Cache Size | KV Exceeds Cache At Seq Len |
| --- | --- |
| 256KB | 2 |
| 1MB | 9 |
| 8MB | 73 |
| 32MB | 292 |
| 64MB | 585 |

### Precision comparison

| Precision | Seq Len | KV Bytes/Token | Total KV Size |
| --- | --- | --- | --- |
| fp16 | 16 | 114688 | 1.750 MB |
| fp16 | 32 | 114688 | 3.500 MB |
| fp16 | 64 | 114688 | 7.000 MB |
| fp16 | 128 | 114688 | 14.000 MB |
| fp16 | 256 | 114688 | 28.000 MB |
| fp16 | 512 | 114688 | 56.000 MB |
| int8 | 16 | 57344 | 0.875 MB |
| int8 | 32 | 57344 | 1.750 MB |
| int8 | 64 | 57344 | 3.500 MB |
| int8 | 128 | 57344 | 7.000 MB |
| int8 | 256 | 57344 | 14.000 MB |
| int8 | 512 | 57344 | 28.000 MB |
| int4 | 16 | 28672 | 0.438 MB |
| int4 | 32 | 28672 | 0.875 MB |
| int4 | 64 | 28672 | 1.750 MB |
| int4 | 128 | 28672 | 3.500 MB |
| int4 | 256 | 28672 | 7.000 MB |
| int4 | 512 | 28672 | 14.000 MB |

### Interpretation
- KV-cache footprint grows **linearly with sequence length**.
- Decode-time KV reads also grow with sequence length, which increases memory traffic.
- Lower-precision KV formats reduce both **storage size** and **per-step read volume**.
- Once KV-cache exceeds smaller on-chip caches, accesses are more likely to spill to slower memory levels.

## Summary

### Goal 4 connection
This model shows why KV-cache becomes an architectural bottleneck:
- larger sequence lengths increase KV-cache size
- larger KV-cache increases memory traffic during decode
- once cache capacity is exceeded, memory access becomes more expensive
- this contributes to memory-bandwidth saturation and slower decoding

### Goal 5 connection
This model also supports the reduced-precision KV proposal:
- FP16 → INT8 roughly halves KV size
- FP16 → INT4 reduces KV size further
- smaller KV-cache can reduce memory pressure and improve decode latency

### Final takeaway
The analytical KV model gives a concrete explanation for why decode latency grows with sequence length and why reduced-precision KV-cache is a reasonable optimization proposal.

