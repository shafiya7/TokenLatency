from __future__ import annotations

import json
from pathlib import Path


def _latest(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern))
    return files[-1] if files else None


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def main() -> None:
    results_dir = Path("results")
    latest = _latest(results_dir, "single_benchmark_*.json")
    if latest is None:
        print("No benchmark results found.")
        return

    data = json.loads(latest.read_text(encoding="utf-8"))
    summary = data["summary"]["filtered"]
    first = data["trials"][0]

    kv_bytes = first.get("kv_cache_bytes_per_token")
    ttft = float(summary["ttft_ms"]["mean"])
    per_token = float(summary["avg_per_token_ms"]["mean"])
    steady = first.get("steady_state_breakdown_avg_ms", {})
    attention_ms = float(steady.get("attention_total_ms", 0.0))
    kv_rw_ms = float(steady.get("kv_cache_rw_estimated_ms", steady.get("kv_cache_rw_ms", 0.0)))
    effective_bandwidth = data.get("measurement_metadata", {}).get("effective_bandwidth_gbps")

    compare_path = _latest(results_dir, "kv_cache_compare_*.json")
    compare = json.loads(compare_path.read_text(encoding="utf-8")) if compare_path else None
    measured_speedup = None
    if compare is not None:
        measured_speedup = compare["comparison"].get("avg_per_token_ms_speedup")

    memory_sensitive_fraction = _safe_fraction(attention_ms + kv_rw_ms, per_token)
    # Assume optimization improves only a subset of the memory-sensitive share.
    est_improvement_pct = min(30.0, max(8.0, 35.0 * memory_sensitive_fraction))
    improved_per_token = per_token * (1.0 - est_improvement_pct / 100.0)

    out = Path("optimization/KV_CACHE_OPTIMIZATION_PROPOSAL.md")
    md = f"""# KV-Cache Optimization Proposal

## Baseline
- Model: {data['config']['model_name']}
- Device: {data['runtime']['device']}
- Dtype: {data['runtime']['dtype']}
- TTFT mean: {ttft:.3f} ms
- Per-token mean: {per_token:.3f} ms
- Estimated KV-cache bytes per token: {kv_bytes}
- Measured steady-state attention time: {attention_ms:.3f} ms
- Estimated KV-cache read/write time: {kv_rw_ms:.3f} ms
- Effective bandwidth used for KV estimate: {effective_bandwidth}

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
- Current steady-state per-token latency: {per_token:.3f} ms
- Memory-sensitive share used in estimate: {memory_sensitive_fraction * 100:.1f}%
- Estimated improvement from layout + reduced-precision KV: {est_improvement_pct:.1f}%
- Estimated new steady-state per-token latency: {improved_per_token:.3f} ms
"""
    if measured_speedup is not None:
        md += f"- Measured cache-vs-no-cache speedup from this project: {measured_speedup:.3f}x\n"
    md += f"""

## Hardware / System Cost
- extra cache-manager complexity
- additional validation work for long-context numerical stability
- potential quantize/dequantize overhead if reduced-precision KV is used

## Expected Best Case
The largest gains should appear at longer prompt lengths and larger models, where cached-context reads dominate steady-state decode time.
"""
    out.write_text(md, encoding="utf-8")
    print(f"Saved proposal to {out}")


if __name__ == "__main__":
    main()
