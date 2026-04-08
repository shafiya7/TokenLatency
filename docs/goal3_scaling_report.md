# Goal 3 Scaling Summary

## Sequence-Length Scaling Snapshot

| Model | Precision | Prompt Tokens | TTFT Mean (ms) | Per-token Mean (ms) | E2E Mean (ms) | Dominant Bottleneck |
|---|---|---:|---:|---:|---:|---|
| TinyLlama-1.1B-Chat-v1.0 | auto | 16 | 80.013 | 65.638 | 1589.681 | attention_total |
| TinyLlama-1.1B-Chat-v1.0 | auto | 32 | 96.321 | 66.338 | 1622.097 | attention_total |
| TinyLlama-1.1B-Chat-v1.0 | auto | 64 | 125.062 | 67.747 | 1683.237 | attention_total |
| TinyLlama-1.1B-Chat-v1.0 | auto | 128 | 143.262 | 69.128 | 1733.217 | attention_total |
| TinyLlama-1.1B-Chat-v1.0 | auto | 256 | 211.380 | 75.297 | 1943.221 | attention_total |
| Llama-3.2-1B-Instruct | auto | 16 | 127.709 | 64.515 | 1611.544 | attention_total |
| Llama-3.2-1B-Instruct | auto | 32 | 89.550 | 56.587 | 1391.053 | attention_total |
| Llama-3.2-1B-Instruct | auto | 64 | 116.280 | 56.581 | 1417.631 | attention_total |
| Llama-3.2-1B-Instruct | auto | 128 | 140.860 | 56.017 | 1429.243 | attention_total |
| Llama-3.2-1B-Instruct | auto | 256 | 214.246 | 57.152 | 1528.733 | attention_total |
| Llama-3.2-3B-Instruct | auto | 16 | 156.509 | 102.017 | 2502.890 | attention_total |
| Llama-3.2-3B-Instruct | auto | 32 | 174.635 | 98.672 | 2444.086 | attention_total |
| Llama-3.2-3B-Instruct | auto | 64 | 253.039 | 106.166 | 2694.865 | attention_total |
| Llama-3.2-3B-Instruct | auto | 128 | 313.332 | 102.826 | 2678.336 | attention_total |
| Llama-3.2-3B-Instruct | auto | 256 | 499.482 | 111.736 | 3069.410 | attention_total |

## Inflection-Point Notes

- TinyLlama-1.1B-Chat-v1.0 [auto] shows a possible inflection near 256 tokens: per-token slope rose from 0.0216 to 0.0482 ms/token (x2.23).
- Llama-3.2-1B-Instruct [auto] shows a possible inflection near 128 tokens: per-token slope rose from -0.0002 to -0.0088 ms/token (x42.73).

## Dominant Bottleneck Interpretation

- Increasing prompt length primarily stresses attention and KV-cache traffic, so per-token latency often rises faster than TTFT after context becomes moderately large.
- Larger models increase both attention and MLP cost because there are more parameters, wider hidden states, and usually more layers.
- Reduced precision can improve latency when the backend and hardware meaningfully accelerate lower-precision kernels or reduce memory traffic.
- If framework overhead becomes visible at short prompts, that usually means launch/synchronization costs are non-trivial relative to the actual compute.
