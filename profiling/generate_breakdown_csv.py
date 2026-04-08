from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    results_dir = Path("results")
    rows = []

    for path in results_dir.glob("single_benchmark_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))

        schema_version = data.get("result_schema_version", "unknown")
        timing_method = data.get("measurement_metadata", {}).get("timing_method", "")
        kv_note = data.get("measurement_metadata", {}).get("kv_cache_measurement_note", "")

        for trial in data.get("trials", []):
            row = {
                "file": path.name,
                "result_schema_version": schema_version,
                "timing_method": timing_method,
                "kv_cache_measurement_note": kv_note,
                "model_name": data["config"]["model_name"],
                "device": data["runtime"]["device"],
                "dtype": data["runtime"]["dtype"],
                "trial_index": trial["trial_index"],
                "prompt_tokens": trial["prompt_tokens"],
                "generated_tokens": trial["generated_tokens"],
                "ttft_ms": trial["ttft_ms"],
                "avg_per_token_ms": trial["avg_per_token_ms"],
                "e2e_ms": trial["e2e_ms"],
                "kv_cache_bytes_per_token": trial.get("kv_cache_bytes_per_token"),
            }

            row.update({f"first_{k}": v for k, v in trial.get("first_step_breakdown_ms", {}).items()})
            row.update({f"steady_{k}": v for k, v in trial.get("steady_state_breakdown_avg_ms", {}).items()})
            rows.append(row)

    if not rows:
        print("No benchmark files found in results/")
        return

    fieldnames = sorted({k for row in rows for k in row})
    out_path = results_dir / "breakdown_summary.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()