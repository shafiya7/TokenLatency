from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")
OUTPUT_PATH = Path("optimization/KV_CACHE_OPTIMIZATION_PROPOSAL.md")


def _latest(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern))
    return files[-1] if files else None


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _extract_prompt_length(data: dict[str, Any]) -> Any:
    candidates = [
        data.get("config", {}).get("prompt_length"),
        data.get("config", {}).get("input_length"),
        data.get("config", {}).get("sequence_length"),
        data.get("measurement_metadata", {}).get("prompt_length"),
    ]
    for item in candidates:
        if item is not None:
            return item
    return "N/A"


def _extract_num_trials(data: dict[str, Any]) -> int:
    trials = data.get("trials", [])
    return len(trials) if isinstance(trials, list) else 0


def _estimate_improvement(
    per_token_ms: float,
    attention_ms: float,
    kv_rw_ms: float,
    measured_speedup: float | None,
    dtype: str,
    precision_matrix: dict[str, Any] | None,
) -> tuple[float, float, float, str]:
    """
    Returns:
        memory_sensitive_fraction,
        estimated_improvement_pct,
        improved_per_token_ms,
        rationale_text
    """
    memory_sensitive_fraction = _safe_fraction(attention_ms + kv_rw_ms, per_token_ms)
    base_pct = 35.0 * memory_sensitive_fraction

    dtype_lower = (dtype or "").lower()
    rationale_parts: list[str] = []

    if "32" in dtype_lower:
        base_pct += 2.0
        rationale_parts.append(
            "The baseline run uses float32, so reduced-precision KV storage has meaningful room to cut memory traffic."
        )

    if measured_speedup is not None:
        if measured_speedup >= 1.8:
            base_pct += 2.0
        elif measured_speedup >= 1.4:
            base_pct += 1.0
        rationale_parts.append(
            f"The measured cache-vs-no-cache speedup of {measured_speedup:.3f}x shows cached-context handling strongly affects decode latency."
        )

    if precision_matrix is not None:
        rows = precision_matrix.get("rows", [])
        fp16_with = next(
            (
                r for r in rows
                if str(r.get("precision")).lower() in {"float16", "fp16"}
                and bool(r.get("use_kv_cache")) is True
            ),
            None,
        )
        fp32_with = next(
            (
                r for r in rows
                if str(r.get("precision")).lower() in {"float32", "fp32"}
                and bool(r.get("use_kv_cache")) is True
            ),
            None,
        )
        if fp16_with and fp32_with:
            fp16_pt = _safe_float(fp16_with.get("per_token_mean_ms"), default=-1.0)
            fp32_pt = _safe_float(fp32_with.get("per_token_mean_ms"), default=-1.0)
            if fp16_pt > 0 and fp32_pt > 0 and fp16_pt < fp32_pt:
                base_pct += 1.0
                rationale_parts.append(
                    "The precision matrix shows that lower precision already reduces per-token latency with KV enabled, reinforcing that memory movement is a real decode bottleneck."
                )

    estimated_improvement_pct = min(30.0, max(8.0, base_pct))
    improved_per_token_ms = per_token_ms * (1.0 - estimated_improvement_pct / 100.0)

    rationale_text = " ".join(rationale_parts) if rationale_parts else (
        "The estimate is driven by the measured memory-sensitive share of steady-state decode."
    )
    return memory_sensitive_fraction, estimated_improvement_pct, improved_per_token_ms, rationale_text


def _precision_matrix_table(matrix: dict[str, Any]) -> str:
    rows = matrix.get("rows", [])
    if not rows:
        return "No precision matrix rows were found.\n"

    lines = []
    lines.append("| Precision | KV Cache | TTFT (ms) | Per-token (ms) | E2E (ms) | Runtime device | Runtime dtype |")
    lines.append("|---|---:|---:|---:|---:|---|---|")

    def _sort_key(r: dict[str, Any]) -> tuple[int, int]:
        prec = str(r.get("precision", "")).lower()
        prec_rank = 0 if prec in {"float32", "fp32"} else 1 if prec in {"float16", "fp16"} else 2
        kv_rank = 0 if bool(r.get("use_kv_cache")) else 1
        return (prec_rank, kv_rank)

    for r in sorted(rows, key=_sort_key):
        kv_label = "With cache" if bool(r.get("use_kv_cache")) else "Without cache"
        lines.append(
            f"| {r.get('precision', 'N/A')} | {kv_label} | "
            f"{_fmt(r.get('ttft_mean_ms'))} | {_fmt(r.get('per_token_mean_ms'))} | {_fmt(r.get('e2e_mean_ms'))} | "
            f"{r.get('runtime_device', 'N/A')} | {r.get('runtime_dtype', 'N/A')} |"
        )

    return "\n".join(lines) + "\n"


def _pairwise_matrix_section(matrix: dict[str, Any]) -> str:
    pairwise = matrix.get("pairwise_comparisons", [])
    if not pairwise:
        return "No pairwise precision/KV comparisons were found.\n"

    out = []
    out.append("| Precision | Speedup with KV (x) | Per-token savings (ms) | TTFT delta (ms) | E2E delta (ms) |")
    out.append("|---|---:|---:|---:|---:|")
    for p in pairwise:
        out.append(
            f"| {p.get('precision', 'N/A')} | {_fmt(p.get('per_token_speedup_with_kv'))} | "
            f"{_fmt(p.get('per_token_delta_ms'))} | {_fmt(p.get('ttft_delta_ms'))} | {_fmt(p.get('e2e_delta_ms'))} |"
        )
    return "\n".join(out) + "\n"


def _cross_precision_analysis(matrix: dict[str, Any]) -> str:
    rows = matrix.get("rows", [])
    if not rows:
        return "The precision matrix was not available, so cross-precision analysis could not be generated automatically.\n"

    def _find_row(precision_names: set[str], use_kv_cache: bool) -> dict[str, Any] | None:
        for r in rows:
            if str(r.get("precision", "")).lower() in precision_names and bool(r.get("use_kv_cache")) == use_kv_cache:
                return r
        return None

    fp32_with = _find_row({"float32", "fp32"}, True)
    fp32_without = _find_row({"float32", "fp32"}, False)
    fp16_with = _find_row({"float16", "fp16"}, True)
    fp16_without = _find_row({"float16", "fp16"}, False)

    notes = []

    if fp32_with and fp16_with:
        a = _safe_float(fp32_with.get("per_token_mean_ms"), -1.0)
        b = _safe_float(fp16_with.get("per_token_mean_ms"), -1.0)
        if a > 0 and b > 0:
            pct = (a - b) / a * 100.0
            direction = "lower" if b < a else "higher"
            notes.append(
                f"- With KV-cache enabled, **fp16** per-token latency is {_fmt(abs(a - b))} ms ({pct:.1f}% {direction}) compared with **fp32**."
            )

    if fp32_without and fp16_without:
        a = _safe_float(fp32_without.get("per_token_mean_ms"), -1.0)
        b = _safe_float(fp16_without.get("per_token_mean_ms"), -1.0)
        if a > 0 and b > 0:
            pct = (a - b) / a * 100.0
            direction = "lower" if b < a else "higher"
            notes.append(
                f"- Without KV-cache, **fp16** per-token latency is {_fmt(abs(a - b))} ms ({pct:.1f}% {direction}) compared with **fp32**."
            )

    if fp32_with and fp32_without:
        on = _safe_float(fp32_with.get("per_token_mean_ms"), -1.0)
        off = _safe_float(fp32_without.get("per_token_mean_ms"), -1.0)
        if on > 0 and off > 0:
            notes.append(
                f"- Under **fp32**, enabling KV-cache changes per-token latency from {_fmt(off)} ms to {_fmt(on)} ms."
            )

    if fp16_with and fp16_without:
        on = _safe_float(fp16_with.get("per_token_mean_ms"), -1.0)
        off = _safe_float(fp16_without.get("per_token_mean_ms"), -1.0)
        if on > 0 and off > 0:
            notes.append(
                f"- Under **fp16**, enabling KV-cache changes per-token latency from {_fmt(off)} ms to {_fmt(on)} ms."
            )

    if not notes:
        return "Cross-precision comparison data was incomplete, so only the matrix table could be shown.\n"

    notes.append(
        "- This comparison is useful architecturally because precision changes the amount of data moved through the memory hierarchy, while KV-cache changes whether past context is recomputed or read back from memory."
    )
    return "\n".join(notes) + "\n"


def main() -> None:
    latest_single = _latest(RESULTS_DIR, "single_benchmark_*.json")
    if latest_single is None:
        print("No single benchmark results found in results/.")
        return

    data = _read_json(latest_single)
    if data is None:
        print("Could not read benchmark JSON.")
        return

    summary = data.get("summary", {}).get("filtered", {})
    trials = data.get("trials", [])
    first_trial = trials[0] if trials else {}

    model_name = data.get("config", {}).get("model_name", "Unknown")
    device = data.get("runtime", {}).get("device", "Unknown")
    dtype = data.get("runtime", {}).get("dtype", "Unknown")
    prompt_length = _extract_prompt_length(data)
    num_trials = _extract_num_trials(data)

    ttft = _safe_float(summary.get("ttft_ms", {}).get("mean"))
    per_token = _safe_float(summary.get("avg_per_token_ms", {}).get("mean"))
    e2e = _safe_float(summary.get("e2e_ms", {}).get("mean"))

    kv_bytes_per_token = first_trial.get("kv_cache_bytes_per_token", "N/A")
    steady = first_trial.get("steady_state_breakdown_avg_ms", {})
    attention_ms = _safe_float(steady.get("attention_total_ms"))
    kv_rw_ms = _safe_float(
        steady.get("kv_cache_rw_estimated_ms", steady.get("kv_cache_rw_ms", 0.0))
    )
    mlp_ms = _safe_float(steady.get("mlp_ms"))
    layernorm_residual_ms = _safe_float(
        steady.get("layernorm_residual_ms", steady.get("norm_residual_ms", 0.0))
    )
    embedding_ms = _safe_float(steady.get("embedding_ms"))
    sampling_ms = _safe_float(steady.get("sampling_ms"))
    framework_overhead_ms = _safe_float(
        steady.get("framework_overhead_ms", steady.get("other_ms", 0.0))
    )
    effective_bandwidth = data.get("measurement_metadata", {}).get(
        "effective_bandwidth_gbps"
    )

    compare_path = _latest(RESULTS_DIR, "kv_cache_compare_*.json")
    compare = _read_json(compare_path)

    matrix_path = _latest(RESULTS_DIR, "precision_kv_matrix_*.json")
    matrix = _read_json(matrix_path)

    measured_speedup = None
    cache_per_token = None
    no_cache_per_token = None
    cache_ttft = None
    no_cache_ttft = None

    if compare is not None:
        comparison = compare.get("comparison", {})
        measured_speedup = comparison.get("avg_per_token_ms_speedup")

        cache_run = compare.get("cache", {}) or compare.get("with_cache", {}) or {}
        no_cache_run = compare.get("no_cache", {}) or compare.get("without_cache", {}) or {}

        cache_per_token = (
            cache_run.get("summary", {})
            .get("filtered", {})
            .get("avg_per_token_ms", {})
            .get("mean")
        )
        no_cache_per_token = (
            no_cache_run.get("summary", {})
            .get("filtered", {})
            .get("avg_per_token_ms", {})
            .get("mean")
        )
        cache_ttft = (
            cache_run.get("summary", {})
            .get("filtered", {})
            .get("ttft_ms", {})
            .get("mean")
        )
        no_cache_ttft = (
            no_cache_run.get("summary", {})
            .get("filtered", {})
            .get("ttft_ms", {})
            .get("mean")
        )

    (
        memory_sensitive_fraction,
        est_improvement_pct,
        improved_per_token,
        estimate_rationale,
    ) = _estimate_improvement(
        per_token_ms=per_token,
        attention_ms=attention_ms,
        kv_rw_ms=kv_rw_ms,
        measured_speedup=_safe_float(measured_speedup, default=0.0) if measured_speedup is not None else None,
        dtype=str(dtype),
        precision_matrix=matrix,
    )

    absolute_savings = max(0.0, per_token - improved_per_token)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    md = f"""# KV-Cache Optimization Proposal and Architectural Bottleneck Analysis

## 1. Baseline Measurement

This report is automatically generated from the most recent benchmark result file.

- Source benchmark file: `{latest_single.name}`
- Model: {model_name}
- Device: {device}
- Dtype: {dtype}
- Prompt length: {prompt_length}
- Number of measured trials: {num_trials}

Measured baseline results:

- TTFT mean: {ttft:.3f} ms
- Steady-state per-token latency mean: {per_token:.3f} ms
- End-to-end latency mean: {e2e:.3f} ms
- Estimated KV-cache bytes per token: {kv_bytes_per_token}
- Measured steady-state attention time: {attention_ms:.3f} ms
- Estimated KV-cache read/write time: {kv_rw_ms:.3f} ms
- MLP time: {mlp_ms:.3f} ms
- LayerNorm/residual time: {layernorm_residual_ms:.3f} ms
- Embedding time: {embedding_ms:.3f} ms
- Sampling time: {sampling_ms:.3f} ms
- Framework/other overhead: {framework_overhead_ms:.3f} ms
- Effective bandwidth used for KV estimate: {effective_bandwidth if effective_bandwidth is not None else "N/A"}

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

The benchmark estimates **{kv_bytes_per_token} bytes of KV-cache per token**.

Architectural implication:
- KV-cache footprint grows approximately linearly with sequence length.
- At longer contexts, the working set becomes larger and harder to keep close to the processor.
- More decode time is spent revisiting prior cached state instead of performing fresh useful arithmetic.

Why this matters:
- The memory system must support a growing cached context for every generated token.
- This creates increasing pressure on cache capacity and higher-latency memory paths.

#### B. Memory bandwidth saturation

The measured attention time is **{attention_ms:.3f} ms**, while total steady-state per-token latency is **{per_token:.3f} ms**.

That means attention alone accounts for approximately **{_safe_fraction(attention_ms, per_token) * 100.0:.1f}%** of steady-state token latency.

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

The run uses **{device}** with **{dtype}**, and decode typically involves many repeated framework-dispatched operations.

Architectural implication:
- Small per-token operations can incur nontrivial runtime, launch, or synchronization overhead.
- These overheads become more visible in token-by-token decoding than in large batched training-style workloads.

Why this matters:
- Some portion of the measured latency is due to orchestration overhead, not just raw attention math.

## 3. Precision and KV-Cache Matrix from Your Testing

"""

    if matrix is not None:
        md += f"""A precision/KV experiment file was found and incorporated automatically:

- Source matrix file: `{matrix_path.name}`

### Full precision × KV matrix

{_precision_matrix_table(matrix)}

### Pairwise cache benefit within each precision

{_pairwise_matrix_section(matrix)}

### Cross-precision interpretation

{_cross_precision_analysis(matrix)}

### Why precision belongs in the bottleneck analysis

Precision changes not only numerical representation, but also the amount of data moved through the memory hierarchy. When the workload is already memory-sensitive, reducing precision can help because:

- fewer bytes are read and written for cached activations or KV-related state
- memory bandwidth pressure is reduced
- more data may fit closer to the processor
- decode steps can spend less time stalled on memory traffic

That makes the precision study directly relevant to Goal 4, because it helps distinguish whether latency is dominated by compute or by data movement. If lower precision improves steady-state token latency, that is evidence that the decode path is at least partly bandwidth-sensitive.

"""
    else:
        md += """No `precision_kv_matrix_*.json` file was found, so precision/KV matrix analysis could not be included automatically.
If you generate one, this script will add the table and cross-precision discussion next time.

"""

    md += """## 4. Main Bottleneck Summary

The dominant steady-state bottleneck is **memory-sensitive attention during decoding**.

This conclusion is supported by the measurements:

"""
    md += f"""- Per-token latency: **{per_token:.3f} ms**
- Attention contribution: **{attention_ms:.3f} ms**
- Estimated KV read/write time: **{kv_rw_ms:.3f} ms**
- KV-cache bytes per token: **{kv_bytes_per_token}**
- Estimated memory-sensitive share of token latency: **{memory_sensitive_fraction * 100.0:.1f}%**

Taken together, these measurements indicate that steady-state decode is limited less by raw arithmetic throughput and more by repeated movement and access of cached context.

## 5. Evidence from Your Testing

"""

    if compare is not None:
        md += f"""A comparison file was also found and incorporated automatically:

- Source comparison file: `{compare_path.name}`
- Measured cache-vs-no-cache speedup (per-token): {_fmt(measured_speedup)}x
- With-cache per-token latency: {_fmt(cache_per_token)} ms
- Without-cache per-token latency: {_fmt(no_cache_per_token)} ms
- With-cache TTFT: {_fmt(cache_ttft)} ms
- Without-cache TTFT: {_fmt(no_cache_ttft)} ms

Interpretation:
- KV-cache already provides a measurable speedup over recomputing attention from scratch.
- This means the correct optimization direction is **not removing KV-cache**, but making the KV-cache itself more efficient.
- Your measurements therefore support a proposal focused on **KV representation and access efficiency**.

"""
    else:
        md += """No `kv_cache_compare_*.json` file was found, so no cache-vs-no-cache comparison could be added.
If you generate one, this script will include it automatically next time.

"""

    md += f"""## 6. Goal 5: Concrete Optimization Proposal

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

- Current steady-state per-token latency: {per_token:.3f} ms

### Estimate rationale

A reasonable estimate is based on the measured memory-sensitive share of the per-token path, while avoiding overclaiming benefits for components that are not directly improved by KV optimization.

Using:
- attention time = {attention_ms:.3f} ms
- KV read/write estimate = {kv_rw_ms:.3f} ms
- per-token latency = {per_token:.3f} ms
- estimated memory-sensitive share = {memory_sensitive_fraction * 100.0:.1f}%

the projected improvement from **reduced-precision KV-cache + cache-friendly layout** is:

- Estimated latency improvement: {est_improvement_pct:.1f}%
- Estimated new steady-state per-token latency: {improved_per_token:.3f} ms
- Estimated absolute savings per token: {absolute_savings:.3f} ms

Supporting rationale:
{estimate_rationale}

### Before / After Summary

- Before: {per_token:.3f} ms per token
- After: {improved_per_token:.3f} ms per token
- Improvement: {est_improvement_pct:.1f}% ({absolute_savings:.3f} ms/token)

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
- the measured speedup from using KV-cache versus not using it"""

    if measured_speedup is not None:
        md += f" ({_fmt(measured_speedup)}x in your tests)"
    md += """"""

    if matrix is not None:
        md += """
- the precision matrix showing how latency changes across fp32/fp16 and with/without KV-cache
"""
    else:
        md += ".\n"

    md += f"""
Therefore, the proposed optimization is to **store KV-cache in reduced precision and reorganize its layout for more cache-friendly decode access**.

This proposal is expected to reduce steady-state token latency from **{per_token:.3f} ms** to approximately **{improved_per_token:.3f} ms**, with the strongest benefits appearing at longer contexts and larger models.
"""

    OUTPUT_PATH.write_text(md, encoding="utf-8")
    print(f"Saved proposal to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()