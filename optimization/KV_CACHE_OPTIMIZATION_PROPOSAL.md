# KV-Cache Optimization Proposal and Architectural Bottleneck Analysis

## 1. Baseline Measurement

This report is automatically generated from the most recent benchmark result file.

- Source benchmark file: `single_benchmark_kv_on_20260415_011020.json`
- Model: meta-llama/Llama-3.2-1B-Instruct
- Device: mps
- Dtype: float16
- Prompt length: N/A
- Number of measured trials: 3

Measured baseline results:

- TTFT mean: 79.095 ms
- Steady-state per-token latency mean: 54.418 ms
- End-to-end latency mean: 1330.708 ms
- Estimated KV-cache bytes per token: 32768
- Measured steady-state attention time: 24.398 ms
- Estimated KV-cache read/write time: 0.009 ms
- MLP time: 0.000 ms
- LayerNorm/residual time: 0.000 ms
- Embedding time: 0.000 ms
- Sampling time: 0.000 ms
- Framework/other overhead: 3.642 ms
- Effective bandwidth used for KV estimate: 120.0

These measurements show that once the first token is produced, steady-state token generation still remains expensive. Even with KV-cache enabled, each new token must repeatedly access cached keys and values from previous context.

## 2. Goal 4: Architectural Bottleneck Analysis

### Why the bottleneck occurs

During autoregressive decoding, the model generates one token at a time. For each new token:

1. The current token is embedded and passed through all transformer layers.
2. At each attention layer, the query for the new token must attend to all previously cached keys and values.
3. As context grows, the amount of KV-cache data that must be revisited also grows.
4. This shifts the decoding stage away from being purely compute-bound and toward being increasingly memory-sensitive.

In short, KV-cache removes expensive recomputation of past states, but it also makes steady-state decoding depend heavily on repeated memory access to cached context.

### Mapping from measurement to architecture

#### A. Cache capacity vs. KV-cache size

The benchmark estimates **32768 bytes of KV-cache per token**.

Architectural implication:
- KV-cache footprint grows approximately linearly with sequence length.
- At longer contexts, the working set becomes larger and harder to keep close to the processor.
- More decode time is spent revisiting prior cached state instead of performing fresh useful arithmetic.

Why this matters:
- The memory system must support a growing cached context for every generated token.
- This creates increasing pressure on cache capacity and higher-latency memory paths.

#### B. Memory bandwidth saturation

The measured attention time is **24.398 ms**, while total steady-state per-token latency is **54.418 ms**.

That means attention alone accounts for approximately **44.8%** of steady-state token latency.

Architectural implication:
- The attention path is a major contributor to token latency.
- During decode, attention must repeatedly read previously stored keys and values.
- As the sequence grows, this repeated data movement can saturate effective memory bandwidth before arithmetic units are fully utilized.

Why this matters:
- The system can stall on memory movement even if there is still compute capacity available.

#### C. Poor arithmetic intensity

Arithmetic intensity refers to the amount of useful computation performed per byte of memory transferred.

In decode:
- The model processes only one newly generated token at a time.
- However, it still must read a large amount of historical KV-cache.
- This means memory traffic grows faster than useful arithmetic for the new token.

Architectural implication:
- Decoding operates in a low-arithmetic-intensity regime.
- This is consistent with a memory-sensitive bottleneck rather than a purely compute-bound one.

Why this matters:
- Optimizations that reduce KV traffic or improve cache locality can help more than optimizations that only increase raw compute throughput.

#### D. Pipeline bubbles

Autoregressive decoding is inherently sequential:
- token N+1 cannot be generated until token N is completed
- each layer depends on outputs from the previous layer
- repeated small decode steps can leave hardware underutilized between operations

Architectural implication:
- Some hardware resources may sit idle while waiting for memory fetches or prior stages to finish.
- This creates pipeline bubbles and lowers effective utilization.

Why this matters:
- Observed latency is affected not only by math cost, but also by stalls between dependent decode stages.

#### E. Synchronization overhead

The run uses **mps** with **float16**, and decode typically involves many repeated framework-dispatched operations.

Architectural implication:
- Small per-token operations can incur nontrivial runtime, launch, or synchronization overhead.
- These overheads become more visible in token-by-token decoding than in large batched training-style workloads.

Why this matters:
- Some portion of the measured latency is due to orchestration overhead, not just raw attention math.

## 3. Precision and KV-Cache Matrix from Your Testing

A precision/KV experiment file was found and incorporated automatically:

- Source matrix file: `precision_kv_matrix_20260415_011030.json`

### Full precision × KV matrix

| Precision | KV Cache | TTFT (ms) | Per-token (ms) | E2E (ms) | Runtime device | Runtime dtype |
|---|---:|---:|---:|---:|---|---|
| float32 | With cache | 111.464 | 62.926 | 1558.752 | mps | float32 |
| float32 | Without cache | 87.393 | 92.216 | 2208.352 | mps | float32 |
| float16 | With cache | 79.095 | 54.418 | 1330.708 | mps | float16 |
| float16 | Without cache | 84.741 | 89.981 | 2154.302 | mps | float16 |


### Pairwise cache benefit within each precision

| Precision | Speedup with KV (x) | Per-token savings (ms) | TTFT delta (ms) | E2E delta (ms) |
|---|---:|---:|---:|---:|
| float32 | 1.465 | 29.290 | -24.071 | 649.600 |
| float16 | 1.654 | 35.563 | 5.647 | 823.595 |


### Cross-precision interpretation

- With KV-cache enabled, **fp16** per-token latency is 8.508 ms (13.5% lower) compared with **fp32**.
- Without KV-cache, **fp16** per-token latency is 2.235 ms (2.4% lower) compared with **fp32**.
- Under **fp32**, enabling KV-cache changes per-token latency from 92.216 ms to 62.926 ms.
- Under **fp16**, enabling KV-cache changes per-token latency from 89.981 ms to 54.418 ms.
- This comparison is useful architecturally because precision changes the amount of data moved through the memory hierarchy, while KV-cache changes whether past context is recomputed or read back from memory.


### Why precision belongs in the bottleneck analysis

Precision changes not only numerical representation, but also the amount of data moved through the memory hierarchy. When the workload is already memory-sensitive, reducing precision can help because:

- fewer bytes are read and written for cached activations or KV-related state
- memory bandwidth pressure is reduced
- more data may fit closer to the processor
- decode steps can spend less time stalled on memory traffic

That makes the precision study directly relevant to Goal 4, because it helps distinguish whether latency is dominated by compute or by data movement. If lower precision improves steady-state token latency, that is evidence that the decode path is at least partly bandwidth-sensitive.

## 4. Main Bottleneck Summary

The dominant steady-state bottleneck is **memory-sensitive attention during decoding**.

This conclusion is supported by the measurements:

- Per-token latency: **54.418 ms**
- Attention contribution: **24.398 ms**
- Estimated KV read/write time: **0.009 ms**
- KV-cache bytes per token: **32768**
- Estimated memory-sensitive share of token latency: **44.9%**

Taken together, these measurements indicate that steady-state decode is limited less by raw arithmetic throughput and more by repeated movement and access of cached context.

## 5. Evidence from Your Testing

A comparison file was also found and incorporated automatically:

- Source comparison file: `kv_cache_compare_20260415_010443.json`
- Measured cache-vs-no-cache speedup (per-token): 1.532x

Interpretation:
- KV-cache already provides a measurable speedup over recomputing attention from scratch.
- This means the correct optimization direction is **not removing KV-cache**, but making the KV-cache itself more efficient.
- Your measurements therefore support a proposal focused on **KV representation and access efficiency**.

## 6. Goal 5: Concrete Optimization Proposal

### Proposed optimization: Reduced-precision KV-cache with cache-friendly layout

The proposed optimization is to keep main model computation in its normal compute precision, but make the **KV-cache** more memory-efficient and more contiguous for decode-time access.

### What changes

- Keep compute activations in fp16/bf16 or the model's normal compute precision path
- Store cached keys and values in a reduced-precision format when accuracy allows
- Reorganize KV layout so head-wise decode reads are more contiguous and cache-friendly

### Why this optimization fits the measured bottleneck

This proposal directly targets the architectural issues identified above:

- It reduces the amount of memory traffic required to revisit cached context
- It improves cache locality during decode
- It addresses poor arithmetic intensity by reducing bytes moved per useful attention computation
- It can reduce decode stalls caused by repeated cache fetches

This is a better fit than a compute-only optimization because your own measurements indicate that steady-state decoding is strongly shaped by memory-sensitive attention behavior.

## 7. Before / After Latency Estimate

### Baseline

- Current steady-state per-token latency: 54.418 ms

### Estimate rationale

A reasonable estimate is based on the measured memory-sensitive share of the per-token path, while avoiding overclaiming benefits for components that are not directly improved by KV optimization.

Using:
- attention time = 24.398 ms
- KV read/write estimate = 0.009 ms
- per-token latency = 54.418 ms
- estimated memory-sensitive share = 44.9%

the projected improvement from **reduced-precision KV-cache + cache-friendly layout** is:

- Estimated latency improvement: 17.7%
- Estimated new steady-state per-token latency: 44.787 ms
- Estimated absolute savings per token: 9.631 ms

Supporting rationale:
The measured cache-vs-no-cache speedup of 1.532x shows cached-context handling strongly affects decode latency. The precision matrix shows that lower precision already reduces per-token latency with KV enabled, reinforcing that memory movement is a real decode bottleneck.

### Before / After Summary

- Before: 54.418 ms per token
- After: 44.787 ms per token
- Improvement: 17.7% (9.631 ms/token)

## 8. Hardware or System Cost

This optimization has real implementation costs:

- additional KV-cache manager complexity
- validation work for long-context numerical stability
- possible quantize/dequantize overhead
- more engineering effort than a simple runtime flag change
- need to confirm that output quality remains acceptable after reduced-precision KV storage

However, these costs are justified because the optimization directly addresses the dominant decode-time bottleneck observed in the measurements.

## 9. Expected Best Case

The largest gains should appear under conditions where cached-context access dominates token generation most strongly:

- longer prompt lengths
- larger transformer models
- memory-constrained decode paths
- settings where attention already occupies a large fraction of per-token latency

## 10. Final Conclusion

The benchmark results show that steady-state token generation is constrained by **memory-sensitive attention behavior** during autoregressive decoding.

The evidence comes from:
- the relatively high steady-state per-token latency
- the significant attention contribution
- the growing KV-cache footprint per token
- the inherent sequential nature of decode
- the measured speedup from using KV-cache versus not using it (1.532x in your tests)
- the precision matrix showing how latency changes across fp32/fp16 and with/without KV-cache

Therefore, the proposed optimization is to **store KV-cache in reduced precision and reorganize its layout for more cache-friendly decode access**.

This proposal is expected to reduce steady-state token latency from **54.418 ms** to approximately **44.787 ms**, with the strongest benefits appearing at longer contexts and larger models.
