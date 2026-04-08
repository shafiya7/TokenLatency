from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_dir = Path("results")
    paths = sorted(results_dir.glob("single_benchmark_*.json"))
    scaling_paths = sorted(results_dir.glob("scaling_*.json"))
    compare_paths = sorted(results_dir.glob("kv_cache_compare_*.json"))
    if not paths:
        print("No single benchmark results found.")
        return

    latest = paths[-1]
    data = json.loads(latest.read_text(encoding="utf-8"))
    summary = data["summary"]["filtered"]
    trials = data["trials"]

    scaling_block = ""
    if scaling_paths:
        scaling = json.loads(scaling_paths[-1].read_text(encoding="utf-8"))
        scaling_lines = []
        for row in scaling.get("rows", []):
            scaling_lines.append(
                f"- {row['model_name']} [{row.get('precision','auto')}] @ prompt {row['prompt_length_tokens']} tokens: "
                f"TTFT={row['ttft_mean_ms']:.2f} ms, per-token={row['per_token_mean_ms']:.2f} ms, E2E={row['e2e_mean_ms']:.2f} ms"
            )
        scaling_block = "\n## Scaling Snapshot\n" + "\n".join(scaling_lines) + "\n"

    compare_block = ""
    if compare_paths:
        compare = json.loads(compare_paths[-1].read_text(encoding="utf-8"))
        cmp = compare["comparison"]
        compare_block = f"""
## KV-Cache Comparison
- Avg per-token with KV cache: {compare['with_kv_cache']['summary']['filtered']['avg_per_token_ms']['mean']:.3f} ms
- Avg per-token without KV cache: {compare['without_kv_cache']['summary']['filtered']['avg_per_token_ms']['mean']:.3f} ms
- Speedup: {cmp.get('avg_per_token_ms_speedup'):.3f}x
- TTFT delta (no-cache - cache): {cmp.get('ttft_ms_delta'):.3f} ms
"""

    out = Path("docs/generated_findings_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    first = trials[0]
    md = f"""# Generated Findings Report

## Configuration
- Model: {data['config']['model_name']}
- Device: {data['runtime']['device']}
- Dtype: {data['runtime']['dtype']}
- Prompt tokens: {first['prompt_tokens']}
- Output tokens: {first['generated_tokens']}
- Warmups: {data['config']['warmup_runs']}
- Measured trials: {data['config']['measured_trials']}
- KV cache enabled: {data['config'].get('use_kv_cache', True)}

## Benchmark Summary
- TTFT mean: {summary['ttft_ms']['mean']:.3f} ms
- Per-token mean: {summary['avg_per_token_ms']['mean']:.3f} ms
- End-to-end mean: {summary['e2e_ms']['mean']:.3f} ms
- TTFT p95: {summary['ttft_ms']['p95']:.3f} ms
- Per-token p95: {summary['avg_per_token_ms']['p95']:.3f} ms

## Outlier Handling
- Method: {data['summary']['outlier_handling']['method']}
- IQR multiplier: {data['summary']['outlier_handling']['iqr_multiplier']}
- Kept TTFT trials: {data['summary']['outlier_handling']['kept_trials']['ttft']}
- Kept per-token trials: {data['summary']['outlier_handling']['kept_trials']['avg_per_token']}
- Kept E2E trials: {data['summary']['outlier_handling']['kept_trials']['e2e']}

## First-Step Breakdown
"""
    for k, v in sorted(first["first_step_breakdown_ms"].items()):
        md += f"- {k}: {v:.3f} ms\n"

    md += "\n## Steady-State Breakdown\n"
    for k, v in sorted(first["steady_state_breakdown_avg_ms"].items()):
        md += f"- {k}: {v:.3f} ms\n"

    if first.get("kv_cache_rw_per_step"):
        first_kv = first["kv_cache_rw_per_step"][0]
        last_kv = first["kv_cache_rw_per_step"][-1]
        md += f"""
## KV-Cache Traffic Estimate
- KV cache bytes per token: {first.get('kv_cache_bytes_per_token')}
- First decode-step KV total bytes: {first_kv.get('kv_total_bytes')}
- Last decode-step KV total bytes: {last_kv.get('kv_total_bytes')}
"""

    md += compare_block
    md += scaling_block
    md += """
## Interpretation
- TTFT is higher because the full prompt path runs before any token is emitted.
- Steady-state decode benefits from KV-cache reuse, but KV-cache traffic still grows with sequence length.
- Attention, MLP, and framework overhead should be interpreted together: if attention time grows with context and KV-cache traffic grows linearly, decode becomes increasingly memory-sensitive.
- Precision changes can shift both arithmetic cost and memory traffic, so scaling experiments should be interpreted jointly by model size, prompt length, and dtype.
"""
    out.write_text(md, encoding="utf-8")
    print(f"Saved report to {out}")


if __name__ == "__main__":
    main()
