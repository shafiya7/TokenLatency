from __future__ import annotations

import json
from pathlib import Path
from typing import Any


COMPONENTS = [
    "embedding_lookup_ms",
    "attention_qkv_ms",
    "attention_softmax_ms",
    "attention_projection_ms",
    "kv_cache_rw_estimated_ms",
    "mlp_total_ms",
    "layernorm_total_ms",
    "residual_ms",
    "lm_head_ms",
    "sampling_decoding_ms",
    "framework_overhead_ms",
    "total_token_ms",
]


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


def load_latest(results_dir: str = "results") -> dict[str, Any]:
    files = sorted(Path(results_dir).glob("single_benchmark_*.json"))
    if not files:
        raise FileNotFoundError("No single_benchmark JSON files found.")
    path = files[-1]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_source_file"] = str(path)
    return data


def avg_component(rows: list[dict[str, float]], key: str) -> float | None:
    vals = [row.get(key, 0.0) for row in rows if isinstance(row, dict)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_report(data: dict[str, Any]) -> str:
    trial = data["trials"][0]
    per_token_rows = trial.get("per_token_breakdown_ms", [])
    steady = trial.get("steady_state_breakdown_avg_ms", {})
    first = trial.get("first_step_breakdown_ms", {})
    cfg = data.get("config", {})
    runtime = data.get("runtime", {})

    lines: list[str] = []
    lines.append("# Goal 2 Latency Decomposition Report")
    lines.append("")
    lines.append(f"**Source file:** `{data.get('_source_file', '-')}`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Model: `{cfg.get('model_name', '-')}`")
    lines.append(f"- Device: `{runtime.get('device', '-')}`")
    lines.append(f"- Dtype: `{runtime.get('dtype', '-')}`")
    lines.append(f"- Generated tokens in example trial: `{trial.get('generated_tokens', '-')}`")
    bw = data.get("measurement_metadata", {}).get("effective_bandwidth_gbps")
    if bw is not None:
        lines.append(f"- Effective bandwidth used for KV estimate: `{bw}` GB/s")
    lines.append("")
    lines.append("## First Token Breakdown")
    lines.append("")
    lines.append("| Component | Time (ms) |")
    lines.append("|---|---:|")
    for key in COMPONENTS:
        lines.append(f"| `{key}` | {_fmt(first.get(key))} |")
    lines.append("")
    lines.append("## Steady-State Average Breakdown")
    lines.append("")
    lines.append("| Component | Avg Time (ms) |")
    lines.append("|---|---:|")
    for key in COMPONENTS:
        lines.append(f"| `{key}` | {_fmt(steady.get(key))} |")
    lines.append("")
    lines.append("## Per-Token Average Recomputed from Stored Token Rows")
    lines.append("")
    lines.append("| Component | Avg Time (ms) |")
    lines.append("|---|---:|")
    for key in COMPONENTS:
        lines.append(f"| `{key}` | {_fmt(avg_component(per_token_rows, key))} |")
    lines.append("")
    lines.append("## Example Per-Token Rows")
    lines.append("")
    lines.append("| Token | Total (ms) | QKV | Softmax | MLP | Residual | Sampling | Overhead |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in per_token_rows[: min(5, len(per_token_rows))]:
        lines.append(
            "| {token} | {total} | {qkv} | {softmax} | {mlp} | {res} | {samp} | {ov} |".format(
                token=_fmt(row.get("token_index")),
                total=_fmt(row.get("total_token_ms")),
                qkv=_fmt(row.get("attention_qkv_ms")),
                softmax=_fmt(row.get("attention_softmax_ms")),
                mlp=_fmt(row.get("mlp_total_ms")),
                res=_fmt(row.get("residual_ms")),
                samp=_fmt(row.get("sampling_decoding_ms")),
                ov=_fmt(row.get("framework_overhead_ms")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    data = load_latest()
    out = Path("docs/goal2_latency_breakdown.md")
    out.write_text(build_report(data), encoding="utf-8")
    print(f"Saved Goal 2 report to: {out}")


if __name__ == "__main__":
    main()
