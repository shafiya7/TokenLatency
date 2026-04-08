# Timing Diagram of Measurement Process

```text
                +---------------- BENCHMARK RUN ----------------+
                |                                               |
Prompt Load --->| Tokenize Prompt                               |
                |                                               |
                | Warm-up Run(s)                                |
                |   - model forward                             |
                |   - cache/init effects                        |
                |                                               |
                +---------------- TIMED REGION -----------------+
                |                                               |
                | t0                                             |
                | |                                              |
                | +--> Prefill / Prompt Processing               |
                |       - embedding                              |
                |       - full attention over prompt             |
                |       - MLP / LN / residual                    |
                |       - logits                                 |
                |       - sample first token                     |
                |                                                |
                | t1 = first token emitted                       |
                | TTFT = t1 - t0                                 |
                |                                                |
                | Decode token 2                                 |
                |   - embed previous token                       |
                |   - attention with KV-cache                    |
                |   - KV read/write                              |
                |   - MLP / LN / residual                        |
                |   - logits + sample                            |
                |                                                |
                | Decode token 3                                 |
                |   ...                                          |
                |                                                |
                | Decode token N                                 |
                |                                                |
                | tN = final token emitted                       |
                | End-to-End Time = tN - t0                      |
                | Steady-State Per-Token = avg(t[i]-t[i-1])      |
                |   for i >= 2                                   |
                +------------------------------------------------+