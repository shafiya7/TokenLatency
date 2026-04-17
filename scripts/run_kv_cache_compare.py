from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import argparse
import json
from dataclasses import asdict
from typing import Any

from bench.config import BenchmarkConfig
from bench.harness import HFLatencyHarness
from bench.utils import ensure_dir, json_dump, timestamp_slug


def run_once(cfg: BenchmarkConfig) -> dict[str, Any]:
    harness = HFLatencyHarness(cfg)
    try:
        return harness.run()
    finally:
        harness.close()


def build_compare_payload(with_cache: dict[str, Any], without_cache: dict[str, Any], cfg: BenchmarkConfig) -> dict[str, Any]:
    on_ms = with_cache["summary"]["filtered"]["avg_per_token_ms"]["mean"]
    off_ms = without_cache["summary"]["filtered"]["avg_per_token_ms"]["mean"]
    ttft_on = with_cache["summary"]["filtered"]["ttft_ms"]["mean"]
    ttft_off = without_cache["summary"]["filtered"]["ttft_ms"]["mean"]
    return {
        "experiment": "hf_kv_cache_compare",
        "base_config": asdict(cfg),
        "with_kv_cache": with_cache,
        "without_kv_cache": without_cache,
        "comparison": {
            "avg_per_token_ms_speedup": (off_ms / on_ms) if on_ms and on_ms > 0 else None,
            "avg_per_token_ms_delta": off_ms - on_ms,
            "ttft_ms_delta": ttft_off - ttft_on,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Hugging Face decoding with and without KV cache.")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--prompt", default="Explain how KV cache reduces autoregressive decoding latency in LLaMA.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--save-dir", default="results")
    parser.add_argument("--system-name", default="local_machine")
    parser.add_argument("--estimated-bandwidth-gbps", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = BenchmarkConfig(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        measured_trials=args.measured_trials,
        device_preference=args.device,
        torch_dtype=args.torch_dtype,
        save_dir=args.save_dir,
        system_name=args.system_name,
        estimated_bandwidth_gbps=args.estimated_bandwidth_gbps,
    )

    with_cache = run_once(BenchmarkConfig(**{**asdict(base_cfg), "use_kv_cache": True}))
    without_cache = run_once(BenchmarkConfig(**{**asdict(base_cfg), "use_kv_cache": False}))

    compare = build_compare_payload(with_cache, without_cache, base_cfg)
    out_dir = ensure_dir(args.save_dir)
    out_path = out_dir / f"kv_cache_compare_{timestamp_slug()}.json"
    json_dump(compare, out_path)

    print(json.dumps({
        "saved_to": str(out_path),
        "avg_per_token_ms_with_kv": with_cache["summary"]["filtered"]["avg_per_token_ms"]["mean"],
        "avg_per_token_ms_without_kv": without_cache["summary"]["filtered"]["avg_per_token_ms"]["mean"],
        "speedup": compare["comparison"]["avg_per_token_ms_speedup"],
    }, indent=2))


if __name__ == "__main__":
    main()
