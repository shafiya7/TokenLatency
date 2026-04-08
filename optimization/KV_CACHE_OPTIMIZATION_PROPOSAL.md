# KV-Cache Optimization Proposal

## Baseline
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Device: mps
- Dtype: float32
- TTFT mean: 220.567 ms
- Per-token mean: 80.765 ms
- Estimated KV-cache bytes per token: 45056
- Measured steady-state attention time: 29.001 ms
- Estimated KV-cache read/write time: 0.101 ms
- Effective bandwidth used for KV estimate: 120.0

## Concrete Optimization
Use a more cache-friendly KV organization together with reduced-precision KV storage:
- keep compute activations in fp16/bf16
- store KV-cache in a reduced-precision format when accuracy allows
- reorganize KV layout so head-wise decode reads are more contiguous

## Why this bottleneck appears
- During decode, each new token revisits the cached context.
- As prompt length grows, KV traffic grows roughly linearly even though arithmetic per new token stays small.
- That pushes decode toward a bandwidth-sensitive regime with poor arithmetic intensity.

## Before / After Estimate
- Current steady-state per-token latency: 80.765 ms
- Memory-sensitive share used in estimate: 36.0%
- Estimated improvement from layout + reduced-precision KV: 12.6%
- Estimated new steady-state per-token latency: 70.579 ms
- Measured cache-vs-no-cache speedup from this project: 1.572x


## Hardware / System Cost
- extra cache-manager complexity
- additional validation work for long-context numerical stability
- potential quantize/dequantize overhead if reduced-precision KV is used

## Expected Best Case
The largest gains should appear at longer prompt lengths and larger models, where cached-context reads dominate steady-state decode time.
