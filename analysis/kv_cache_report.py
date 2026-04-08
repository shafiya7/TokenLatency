from __future__ import annotations

from pathlib import Path

from analysis.kv_cache_model import (
    build_spec,
    cache_thresholds,
    compare_precisions,
)


SEQ_LENGTHS = [16, 32, 64, 128, 256, 512]
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def fmt_mb(x: float) -> str:
    return f"{x:.3f} MB"


def fmt_ms(x: float) -> str:
    return f"{x:.4f} ms"


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line1, line2] + body)


def model_section(model_name: str) -> str:
    spec = build_spec(model_name, key_precision="fp16", value_precision="fp16")

    footprint_rows = []
    for row in spec.table(SEQ_LENGTHS, bandwidth_gbps=100.0):
        footprint_rows.append([
            str(row["seq_len"]),
            f"{int(row['kv_bytes_per_token_total'])}",
            fmt_mb(row["kv_total_mb"]),
            fmt_mb(row["decode_read_mb_one_step"]),
            fmt_ms(row["estimated_transfer_ms"]),
        ])

    threshold_rows = []
    for row in cache_thresholds(model_name):
        threshold_rows.append([
            row["cache_size"],
            str(row["threshold_seq_len"]),
        ])

    precision_rows = []
    for row in compare_precisions(model_name, SEQ_LENGTHS, precisions=("fp16", "int8", "int4")):
        precision_rows.append([
            row["precision"],
            str(row["seq_len"]),
            f"{int(row['kv_bytes_per_token_total'])}",
            fmt_mb(row["kv_total_mb"]),
        ])

    section = f"""## Model: {model_name}

### FP16 KV-cache footprint by sequence length

{make_table(
    ["Seq Len", "KV Bytes/Token", "Total KV Size", "Decode Read / Step", "Estimated Transfer @ 100 GB/s"],
    footprint_rows,
)}

### Cache-capacity thresholds

{make_table(
    ["Cache Size", "KV Exceeds Cache At Seq Len"],
    threshold_rows,
)}

### Precision comparison

{make_table(
    ["Precision", "Seq Len", "KV Bytes/Token", "Total KV Size"],
    precision_rows,
)}

### Interpretation
- KV-cache footprint grows **linearly with sequence length**.
- Decode-time KV reads also grow with sequence length, which increases memory traffic.
- Lower-precision KV formats reduce both **storage size** and **per-step read volume**.
- Once KV-cache exceeds smaller on-chip caches, accesses are more likely to spill to slower memory levels.

"""
    return section


def main() -> None:
    report = """# KV-Cache Analytical Report

## Purpose
This report uses an analytical KV-cache model to explain how KV memory footprint scales with:
- model architecture
- sequence length
- KV precision

It supports:
- **Goal 4**: architectural bottleneck analysis
- **Goal 5**: optimization proposal for reduced-precision KV-cache

## Key Equations
For one token, KV storage added across all layers is approximated as:

`KV bytes per token = num_layers × num_kv_heads × head_dim × (bytes(K) + bytes(V))`

For a sequence of length `L`:

`Total KV cache bytes = L × KV bytes per token`

For one decode step, an approximate lower-bound for KV transfer time is:

`transfer time ≈ KV bytes read / memory bandwidth`

"""

    for model in MODELS:
        report += model_section(model)

    report += """## Summary

### Goal 4 connection
This model shows why KV-cache becomes an architectural bottleneck:
- larger sequence lengths increase KV-cache size
- larger KV-cache increases memory traffic during decode
- once cache capacity is exceeded, memory access becomes more expensive
- this contributes to memory-bandwidth saturation and slower decoding

### Goal 5 connection
This model also supports the reduced-precision KV proposal:
- FP16 → INT8 roughly halves KV size
- FP16 → INT4 reduces KV size further
- smaller KV-cache can reduce memory pressure and improve decode latency

### Final takeaway
The analytical KV model gives a concrete explanation for why decode latency grows with sequence length and why reduced-precision KV-cache is a reasonable optimization proposal.

"""

    out_path = Path("analysis/kv_cache_report.md")
    out_path.write_text(report, encoding="utf-8")
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()