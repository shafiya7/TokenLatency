from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    print(f"\n=== Running {script} {' '.join(args)} ===")
    subprocess.run([sys.executable, str(ROOT / script), *args], check=True)


def main() -> None:
    print("\nStarting full benchmark + analysis pipeline\n")

    # Goal 1
    run("scripts/run_single.py")

    # Goal 2 / KV cache / decomposition support
    run("scripts/run_kv_cache_validation.py")
    run("scripts/run_kv_cache_compare.py")

    # Goal 3
    run("scripts/run_scaling.py")
    run("scripts/run_precision_sweep.py")

    # Raw breakdown data
    run("profiling/generate_breakdown_csv.py")

    # Reports and plots
   
    run("analysis/goal1_report_table.py")
    run("analysis/goal2_report.py")
    run("analysis/goal3_report.py")
    run("analysis/kv_cache_model.py")
    # run("analysis/kv_cache_report.py")
    run("analysis/report_builder.py")

    # Goal 5
    run("optimization/optimization_proposal.py")


    run("analysis/plot_latency_pie.py")
    run("analysis/plot_token_latency_trend.py")

    run("scripts/run_scaling.py")
    run("analysis/plot_scaling.py")

    

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    main()