# Measurement Methodology

## Objective
This project benchmarks token-generation latency in LLaMA-style autoregressive decoding and measures:

- First-token latency (TTFT)
- Steady-state per-token latency
- End-to-end response time

## Experimental Setup
### Hardware
- Device: Apple M4 Pro (Apple Silicon CPU + integrated GPU via MPS backend)
- Memory: 24 GB
- OS: macOS 26.3 (arm64)


### Software Stack
- Python: 3.14.3
- PyTorch: 2.11.0
- Transformers: 5.5.0
- Optional CUDA: Not used (no CUDA-capable GPU available)


### Model Configuration
Models tested:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-3B-Instruct

Decoding configuration:
- Batch size = 1
- Fixed prompt lengths = 16, 32, 64, 128, 256
- Fixed generation length = [fill in your output tokens, e.g. 64 or 128]
- Greedy decoding unless otherwise stated

## Measurement Definitions

### 1. First-Token Latency (TTFT)
TTFT is the elapsed time from the start of model inference on the prompt until the first output token is produced.

This includes:
- embedding lookup
- prefill attention over prompt tokens
- MLP and normalization layers
- logits computation
- sampling / argmax
- framework overhead

### 2. Steady-State Per-Token Latency
Steady-state latency is the average time to generate each token after the first token.

This isolates autoregressive decoding behavior where:
- one new query token is processed
- prior keys/values are reused from KV cache
- attention cost grows with context length

### 3. End-to-End Response Time
Total wall-clock time from inference start to completion of all generated tokens.

## Benchmark Procedure
1. Load model and tokenizer
2. Prepare prompt of fixed length
3. Run warm-up iterations
4. Run timed trials
5. Record:
   - TTFT
   - per-token latencies
   - total elapsed time
6. Remove obvious outliers
7. Compute:
   - mean
   - median
   - std. dev. where applicable

## Warm-Up Policy
Warm-up runs are used before timed runs to reduce distortion from:
- lazy kernel initialization
- model graph setup
- allocator startup
- cache cold-start effects

## Outlier Handling
Outliers are filtered using [fill your actual rule here, for example]:
- discard first trial, or
- remove points beyond 1.5×IQR, or
- report median alongside mean

## Attribution Methodology
Latency decomposition is performed using latency_breakdown hooks and timing wrappers around major decoder components:

- embedding lookup
- attention QKV projection
- attention score/softmax
- attention output projection
- KV-cache read/write
- MLP
- layernorm and residual connections
- sampling / decoding logic
- framework overhead

Framework overhead is estimated as:

`total_token_time - sum(all explicitly measured operator times)`

## Code Path to Measurement Mapping

| Measurement | Code Path / Script |
|---|---|
| TTFT | `scripts/run_single.py` |
| Steady-state token latency | `scripts/run_single.py` |
| Scaling vs sequence/model | `scripts/run_scaling.py` |
| Precision sweep | `scripts/run_precision_sweep.py` |
| Breakdown CSV generation | `profiling/generate_breakdown_csv.py` |
| Goal 2 report | `analysis/goal2_report.py` |
| KV-cache analysis | `analysis/kv_cache_model.py`, `analysis/kv_cache_report.py` |

## Reproducibility Notes
To reproduce:
1. Create virtual environment
2. Install requirements
3. Set `PYTHONPATH` to repo root
4. Run the full pipeline script
5. Inspect generated JSON, CSV, PNG, and Markdown reports

## Limitations
- Fine-grained operator attribution may be approximate on CPU
- KV-cache read/write may partly overlap with attention timing depending on latency_breakdown
- Framework overhead is measured indirectly