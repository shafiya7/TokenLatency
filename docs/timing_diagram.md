User Prompt Submitted
        |
        v
+----------------------+
| Tokenization/Input   |
| preparation          |
+----------------------+
        |
        v
+----------------------+
| Model inference      |
| starts               |
+----------------------+
        |
        |<------ TTFT ------>|
        |                    |
        v                    v
   first token          first token received
   requested

After first token:
        |
        v
+--------------------------------------------------------------+
| Generate token 2 | Generate token 3 | ... | Generate token N |
+--------------------------------------------------------------+
        |-------------- steady-state decoding -----------------|

Per-token latency:
measured across tokens after the first token
= average time between consecutive generated tokens

End-to-end response time:
|<---------------------- total time --------------------------->|
from prompt submission to final token generated



OVERALL FLOW



|--------------------------- End-to-End Latency ----------------------------|

t0                 t1                      t2         t3         t4       tN
|------------------|-----------------------|----------|----------|--------|
Prompt received    Generation starts       Token 1    Token 2    Token 3  Final token

                     <------ TTFT ------>

                                          <--- steady-state token generation --->
                                          |--Δ2--|--Δ3--|--Δ4--| ... |--ΔN--|

Per-token latency = average(Δ2, Δ3, Δ4, ..., ΔN)