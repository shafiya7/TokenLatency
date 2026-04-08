# from __future__ import annotations

# import sys
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(ROOT))

# from bench.config import BenchmarkConfig
# from bench.harness import HFLatencyHarness


# def main() -> None:
#     cfg = BenchmarkConfig(
#         model_name="meta-llama/Llama-3.2-1B-Instruct",
#         prompt="Explain TTFT, per-token latency, and end-to-end latency in decoder-only LLaMA inference.",
#         max_new_tokens=24,
#         warmup_runs=2,
#         measured_trials=5,
#         device_preference="auto",
#         torch_dtype="auto",
#     )
#     harness = HFLatencyHarness(cfg)
#     result = harness.run()

#     print("Saved to:", result["saved_to"])
#     print("Model:", result["config"]["model_name"])
#     print("Device:", result["runtime"]["device"])
#     print("Filtered TTFT mean:", round(result["summary"]["filtered"]["ttft_ms"]["mean"], 3), "ms")
#     print("Filtered per-token mean:", round(result["summary"]["filtered"]["avg_per_token_ms"]["mean"], 3), "ms")
#     print("Filtered E2E mean:", round(result["summary"]["filtered"]["e2e_ms"]["mean"], 3), "ms")


# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bench.config import BenchmarkConfig
from bench.harness import HFLatencyHarness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--prompt", default="Explain TTFT, per-token latency, and end-to-end latency in decoder-only LLaMA inference.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measured-trials", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--use-kv-cache", action="store_true", default=True)
    parser.add_argument("--no-kv-cache", dest="use_kv_cache", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = BenchmarkConfig(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        measured_trials=args.measured_trials,
        device_preference=args.device,
        torch_dtype=args.dtype,
        use_kv_cache=args.use_kv_cache,
    )

    harness = HFLatencyHarness(cfg)
    try:
        result = harness.run()
    finally:
        harness.close()

    print("Saved to:", result["saved_to"])
    print("Schema version:", result.get("result_schema_version"))
    print("Model:", result["config"]["model_name"])
    print("Device:", result["runtime"]["device"])
    print("Dtype:", result["runtime"]["dtype"])
    print("Filtered TTFT mean:", round(result["summary"]["filtered"]["ttft_ms"]["mean"], 3), "ms")
    print("Filtered per-token mean:", round(result["summary"]["filtered"]["avg_per_token_ms"]["mean"], 3), "ms")
    print("Filtered E2E mean:", round(result["summary"]["filtered"]["e2e_ms"]["mean"], 3), "ms")


if __name__ == "__main__":
    main()