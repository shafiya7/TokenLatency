from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


def load_latest_benchmark(results_dir: str = "results") -> dict[str, Any]:
    path = Path(results_dir)
    files = sorted(path.glob("single_benchmark_*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark files found in {path.resolve()}")
    latest = files[-1]
    data = json.loads(latest.read_text(encoding="utf-8"))
    data["_source_file"] = str(latest)
    return data


def compute_raw_std(data: dict[str, Any], metric_key: str) -> float | None:
    values = [trial[metric_key] for trial in data.get("trials", []) if metric_key in trial]
    if len(values) <= 1:
        return None
    return statistics.stdev(values)


def compute_filtered_std(data: dict[str, Any], metric_key: str) -> float | None:
    outlier_info = data.get("summary", {}).get("outlier_handling", {})
    bounds_map = outlier_info.get("bounds", {})
    bounds = bounds_map.get(metric_key)

    if not bounds or len(bounds) != 2:
        return None

    low, high = bounds
    values = [
        trial[metric_key]
        for trial in data.get("trials", [])
        if metric_key in trial and low <= trial[metric_key] <= high
    ]

    if len(values) <= 1:
        return None
    return statistics.stdev(values)


def get_trials_kept(data: dict[str, Any], metric_key: str) -> int | None:
    kept_map = (
        data.get("summary", {})
        .get("outlier_handling", {})
        .get("kept_trials", {})
    )

    mapping = {
        "ttft_ms": "ttft",
        "avg_per_token_ms": "avg_per_token",
        "e2e_ms": "e2e",
    }
    return kept_map.get(mapping.get(metric_key, metric_key))


def build_goal1_table(data: dict[str, Any]) -> str:
    raw = data["summary"]["raw"]
    filtered = data["summary"]["filtered"]
    outliers = data["summary"].get("outlier_handling", {})
    cfg = data.get("config", {})
    runtime = data.get("runtime", {})

    lines: list[str] = []
    lines.append("# Goal 1 Benchmark Summary")
    lines.append("")
    lines.append(f"**Source file:** `{data.get('_source_file', '-')}`")
    lines.append("")
    lines.append("## Benchmark Setup")
    lines.append("")
    lines.append(f"- Model: `{cfg.get('model_name', '-')}`")
    lines.append(f"- Device: `{runtime.get('device', '-')}`")
    lines.append(f"- Runtime dtype: `{runtime.get('dtype', '-')}`")
    lines.append(f"- Prompt length (chars): `{len(cfg.get('prompt', ''))}`")
    lines.append(f"- Max new tokens: `{cfg.get('max_new_tokens', '-')}`")
    lines.append(f"- Warm-up runs: `{cfg.get('warmup_runs', '-')}`")
    lines.append(f"- Measured trials: `{cfg.get('measured_trials', '-')}`")
    lines.append(f"- Outlier method: `{outliers.get('method', cfg.get('outlier_method', '-'))}`")
    lines.append(f"- IQR multiplier: `{outliers.get('iqr_multiplier', cfg.get('iqr_multiplier', '-'))}`")
    lines.append(f"- KV cache enabled: `{cfg.get('use_kv_cache', '-')}`")
    lines.append(f"- Sampling enabled: `{cfg.get('do_sample', '-')}`")
    lines.append("")
    lines.append("## Example Result Table")
    lines.append("")
    lines.append("| Metric | Raw Mean (ms) | Raw Std (ms) | Filtered Mean (ms) | Filtered Std (ms) | Trials Kept |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    metric_rows = [
        ("ttft_ms", "First-token latency (TTFT)"),
        ("avg_per_token_ms", "Steady-state per-token latency"),
        ("e2e_ms", "End-to-end response time"),
    ]

    for key, label in metric_rows:
        raw_mean = raw.get(key, {}).get("mean")
        filt_mean = filtered.get(key, {}).get("mean")

        raw_std = compute_raw_std(data, key)
        filt_std = compute_filtered_std(data, key)
        kept = get_trials_kept(data, key)

        lines.append(
            f"| {label} | {_fmt(raw_mean)} | {_fmt(raw_std)} | {_fmt(filt_mean)} | {_fmt(filt_std)} | {_fmt(kept)} |"
        )

    lines.append("")
    lines.append("## Additional Summary Statistics")
    lines.append("")
    lines.append("| Metric | Raw Median (ms) | Raw Min (ms) | Raw Max (ms) | Raw P95 (ms) |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, label in metric_rows:
        entry = raw.get(key, {})
        lines.append(
            f"| {label} | {_fmt(entry.get('median'))} | {_fmt(entry.get('min'))} | {_fmt(entry.get('max'))} | {_fmt(entry.get('p95'))} |"
        )

    lines.append("")
    lines.append("## Outlier Bounds Used")
    lines.append("")
    lines.append("| Metric | Lower Bound (ms) | Upper Bound (ms) |")
    lines.append("|---|---:|---:|")
    bounds_map = outliers.get("bounds", {})
    for key, label in metric_rows:
        bounds = bounds_map.get(key)
        if bounds and len(bounds) == 2:
            low, high = bounds
        else:
            low, high = None, None
        lines.append(f"| {label} | {_fmt(low)} | {_fmt(high)} |")

    lines.append("")
    lines.append("## Measurement Notes")
    lines.append("")
    lines.append("- TTFT is measured from the start of timed model execution to completion of first-token selection.")
    lines.append("- Steady-state per-token latency is computed only for decode steps after the first generated token.")
    lines.append("- End-to-end latency is computed as TTFT plus the sum of steady-state token latencies.")
    lines.append("- Warm-up runs are excluded from reported statistics.")
    lines.append("- Outliers are filtered using IQR-based bounds before computing filtered statistics.")
    lines.append("- Tokenization and host-side prompt preparation are excluded from the timed region.")
    lines.append("- Raw standard deviation is computed from all measured trials.")
    lines.append("- Filtered standard deviation is recomputed from the subset of trials that fall within the recorded IQR bounds.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    data = load_latest_benchmark("results")
    report = build_goal1_table(data)

    out_path = Path("docs/goal1_example_results.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"Saved Goal 1 example table to: {out_path}")


if __name__ == "__main__":
    main()