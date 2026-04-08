from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_scaling_file() -> Path | None:
    files = sorted(RESULTS_DIR.glob("scaling_*.json"))
    if not files:
        return None
    return files[-1]


def main() -> None:
    scaling_file = find_latest_scaling_file()
    if scaling_file is None:
        print("No scaling_*.json file found in results/")
        print("Run: python scripts/run_scaling.py")
        return

    with open(scaling_file, "r") as fp:
        payload = json.load(fp)

    rows = payload.get("rows", [])
    if not rows:
        print(f"No rows found in {scaling_file}")
        return

    # group by model + precision
    grouped = {}
    for row in rows:
        key = (row["model_name"], row["precision"])
        grouped.setdefault(key, []).append(row)

    plt.figure(figsize=(9, 5))

    for (model_name, precision), group_rows in grouped.items():
        group_rows = sorted(group_rows, key=lambda r: r["prompt_length_tokens"])
        x = [r["prompt_length_tokens"] for r in group_rows]
        y = [r["ttft_mean_ms"] for r in group_rows]

        label = f"{model_name} ({precision})"
        plt.plot(x, y, marker="o", label=label)

    plt.xlabel("Prompt Length (tokens)")
    plt.ylabel("TTFT Mean (ms)")
    plt.title("TTFT vs Prompt Length")
    plt.grid(True)
    plt.legend(fontsize=8)

    out = PLOTS_DIR / "02_scaling_sequence_length.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")
    print(f"Used scaling file: {scaling_file}")


if __name__ == "__main__":
    main()