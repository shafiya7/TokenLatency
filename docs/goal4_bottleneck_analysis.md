# Goal 4: Architectural Bottleneck Analysis

## Objective
This section explains why the measured latency bottlenecks occur, using architectural reasoning.

## 1. TTFT vs Steady-State Decode
Measured results show TTFT is much higher than steady-state per-token latency.

### Why?
The first token requires processing the entire prompt (prefill phase), which includes:
- embedding all prompt tokens
- full attention over the prompt
- all transformer layers across the full input length

After the first token, decoding becomes incremental:
- only one new token is processed at a time
- prior keys/values are reused via KV cache

### Architectural implication
TTFT is dominated by prompt-side compute and setup overhead, while steady-state latency reflects recurrent decode cost.

---

## 2. KV-Cache Growth and Sequence Length Scaling
Latency increases with sequence length.

### Why?
In autoregressive decoding, each new token attends over all previous tokens. Even with KV caching:
- we avoid recomputing old K/V tensors
- but we still must read cached keys and values for all previous positions

Thus, as sequence length grows:
- KV-cache traffic grows
- attention becomes increasingly memory-bound

### Architectural implication
The decode path has growing memory-read pressure, especially for long contexts.

---

## 3. Cache Capacity vs KV-Cache Size
At shorter sequences, KV-cache may fit more effectively within faster cache levels or have lower working-set pressure. As sequence length increases, the working set grows.

### Why this matters
When KV-cache footprint becomes large relative to effective cache capacity:
- more reads are served from slower memory levels
- memory latency and bandwidth pressure rise
- token latency increases more sharply

### Architectural implication
An inflection point may appear once KV-cache size exceeds favorable cache behavior.

---

## 4. Memory Bandwidth Saturation
Attention during decode often becomes bandwidth-limited rather than compute-limited.

### Why?
For each generated token:
- only a small amount of arithmetic is performed per byte fetched
- large KV-cache reads are required
- arithmetic intensity is low

This means performance depends heavily on memory bandwidth.

### Architectural implication
Even if theoretical FLOPs are high, actual decode performance can be limited by memory movement, not math throughput.

---

## 5. Poor Arithmetic Intensity
Steady-state single-token decoding has poor arithmetic intensity.

### Why?
Arithmetic intensity = useful computation / memory traffic.

In batch-1 decode:
- only one token is processed at a time
- matrix operations are smaller
- attention repeatedly reads large cached state
- hardware utilization drops

### Architectural implication
This is why token decoding often achieves poor utilization compared with large-batch training or prefill.

---

## 6. Pipeline Bubbles and Small-Batch Inefficiency
Batch size 1 exposes inefficiencies in the inference pipeline.

### Why?
With tiny workloads:
- kernels are small
- device occupancy is lower
- fixed launch/scheduling costs matter more
- software overhead becomes visible

### Architectural implication
Interactive decoding is harder to optimize than bulk throughput workloads.

---

## 7. Synchronization and Framework Overhead
Measured framework overhead is non-trivial.

### Why?
Overhead may come from:
- kernel launch latency
- synchronization barriers
- Python dispatch
- tensor shape handling
- memory allocation / bookkeeping

### Architectural implication
At batch size 1, overhead that is negligible in large workloads becomes a significant fraction of per-token latency.

---

## 8. Model Size Scaling
Larger models show higher per-token latency.

### Why?
Larger models increase:
- hidden dimension
- number of layers
- parameter count
- KV-cache size per token

This increases both:
- compute cost
- memory traffic

### Architectural implication
Scaling model size increases decode latency due to both arithmetic and memory-system pressure.

---

## 9. Precision Effects
Reduced precision often lowers latency.

### Why?
Lower precision reduces:
- bytes transferred
- cache pressure
- memory bandwidth demand
- sometimes compute cost

This is especially useful when memory traffic dominates.

### Architectural implication
When decode is memory-bound, reduced precision can yield meaningful speedups.

---

## 10. Mapping Measurements to Architecture

| Measurement Observation | Architectural Cause |
|---|---|
| High TTFT | Prompt prefill over full sequence |
| Latency rises with context length | KV-cache read growth |
| Larger models slower | More layers + larger hidden state + bigger KV |
| Framework overhead visible | Batch-1 decode exposes fixed overheads |
| Precision improves latency | Lower memory traffic and bandwidth demand |

## Conclusion
The measurements are consistent with an inference regime in which:
- TTFT is dominated by prompt processing
- steady-state decode is increasingly memory-bound
- KV-cache traffic becomes dominant at long sequence lengths
- batch-1 decoding exposes framework and synchronization overhead
- reduced precision and KV-cache optimizations are promising directions