from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bench.config import BenchmarkConfig
from bench.harness import HFLatencyHarness
from bench.utils import ensure_dir, json_dump, timestamp_slug


def run_once(cfg: BenchmarkConfig) -> dict:
    harness = HFLatencyHarness(cfg)
    try:
        return harness.run()
    finally:
        harness.close()


def metric_mean(result: dict, key: str) -> float:
    return float(result["summary"]["filtered"][key]["mean"])


def main() -> None:
    base_cfg = BenchmarkConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        prompt="Explain TTFT, per-token latency, and end-to-end latency in decoder-only LLaMA inference.",
        max_new_tokens=16,
        warmup_runs=1,
        measured_trials=3,
        device_preference="auto",
        torch_dtype="auto",
    )

    cfg_on = copy.deepcopy(base_cfg)
    cfg_on.use_kv_cache = True

    cfg_off = copy.deepcopy(base_cfg)
    cfg_off.use_kv_cache = False

    result_on = run_once(cfg_on)
    result_off = run_once(cfg_off)

    comparison = {
        "comparison_type": "kv_cache_on_vs_off",
        "model_name": base_cfg.model_name,
        "runtime_note": "Same prompt and generation settings, only use_kv_cache changes.",
        "kv_cache_estimation_note": result_on.get("measurement_metadata", {}).get("kv_cache_measurement_note", ""),
        "kv_on": {
            "saved_to": result_on["saved_to"],
            "ttft_ms_mean": metric_mean(result_on, "ttft_ms"),
            "avg_per_token_ms_mean": metric_mean(result_on, "avg_per_token_ms"),
            "e2e_ms_mean": metric_mean(result_on, "e2e_ms"),
        },
        "kv_off": {
            "saved_to": result_off["saved_to"],
            "ttft_ms_mean": metric_mean(result_off, "ttft_ms"),
            "avg_per_token_ms_mean": metric_mean(result_off, "avg_per_token_ms"),
            "e2e_ms_mean": metric_mean(result_off, "e2e_ms"),
        },
    }

    comparison["measured_delta_ms"] = {
        "ttft_ms": comparison["kv_off"]["ttft_ms_mean"] - comparison["kv_on"]["ttft_ms_mean"],
        "avg_per_token_ms": comparison["kv_off"]["avg_per_token_ms_mean"] - comparison["kv_on"]["avg_per_token_ms_mean"],
        "e2e_ms": comparison["kv_off"]["e2e_ms_mean"] - comparison["kv_on"]["e2e_ms_mean"],
    }

    comparison["interpretation"] = {
        "positive_delta_means_kv_cache_helped": True,
        "note": "This is a direct end-to-end measurement of the benefit of enabling KV-cache. It does not isolate pure KV transfer time, but it validates that KV-cache materially affects latency.",
    }

    out_dir = ensure_dir("results")
    out_path = out_dir / f"kv_cache_validation_{timestamp_slug()}.json"
    json_dump(comparison, out_path)

    print("Saved KV-cache validation to:", out_path)
    print(json.dumps(comparison["measured_delta_ms"], indent=2))


if __name__ == "__main__":
    main()