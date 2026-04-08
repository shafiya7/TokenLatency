# LLaMA Token Latency Benchmark Lab

This project is organized around the five benchmark goals in the assignment:

1. **Benchmark harness design**: repeatable TTFT, steady-state per-token, and end-to-end timing with warmups, multiple trials, and IQR outlier handling.
2. **Latency decomposition**: stage-level timing hooks for embeddings, attention, MLP, layernorm, LM head, KV-cache transfer estimate, sampling/decoding, and framework overhead.
3. **Scaling analysis**: prompt-length sweeps across at least two model variants, plus optional precision sweeps.
4. **Architectural bottleneck analysis**: generated report and notes tying measured growth to memory traffic and decode-time bandwidth pressure.
5. **Optimization proposal**: concrete KV-cache optimization writeup with before/after estimate and optional measured cache-vs-no-cache comparison.

## Main scripts

- `python scripts/run_single.py`
- `python scripts/run_kv_cache_compare.py`
- `python scripts/run_scaling.py`
- `python scripts/run_precision_sweep.py`
- `python scripts/run_full_pipeline.py`

## Outputs

- `results/single_benchmark_*.json`
- `results/kv_cache_compare_*.json`
- `results/scaling_*.json`
- `results/breakdown_summary.csv`
- `plots/*.png`
- `plots/inflection_points.md`
- `docs/generated_findings_report.md`
- `optimization/KV_CACHE_OPTIMIZATION_PROPOSAL.md`

## Notes

- Use small/open models if gated Meta Llama weights are unavailable.
- On Apple Silicon or consumer GPUs, `float16` is often the practical choice.
- The decomposition is based on forward hooks and transfer estimates, so it is best treated as an attribution methodology for coursework rather than a hardware-counter-accurate microarchitectural model.


## Goal 3 workflow

1. `python scripts/run_scaling.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct --prompt-lengths 16 32 64 128 256 --precisions auto`
2. `python scripts/run_precision_sweep.py`
3. `python analysis/plot_scaling.py`
4. `python analysis/goal3_report.py`

This produces Goal 3 plots and markdown reports in `plots/` and `docs/`.
# TokenLatency
