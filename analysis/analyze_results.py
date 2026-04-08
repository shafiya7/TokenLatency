from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_dir = Path("results")
    paths = sorted(results_dir.glob("single_benchmark_*.json"))
    if not paths:
        print("No benchmark files found.")
        return

    latest = paths[-1]
    data = json.loads(latest.read_text(encoding="utf-8"))
    summary = data["summary"]["filtered"]

    print("Latest file:", latest.name)
    print("Model:", data["config"]["model_name"])
    print("Device:", data["runtime"]["device"])
    print("TTFT mean (ms):", round(summary["ttft_ms"]["mean"], 3))
    print("Per-token mean (ms):", round(summary["avg_per_token_ms"]["mean"], 3))
    print("E2E mean (ms):", round(summary["e2e_ms"]["mean"], 3))


if __name__ == "__main__":
    main()
