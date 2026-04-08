from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench.config import META_LLAMA_VARIANTS, ScalingConfig
from bench.scaling import run_scaling
from bench.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run precision sweep for Goal 3.")
    parser.add_argument("--model", default=META_LLAMA_VARIANTS[2])
    parser.add_argument("--prompt-lengths", nargs='*', type=int, default=[64, 128, 256])
    parser.add_argument("--precisions", nargs='*', default=None, help="Defaults are chosen from the resolved device.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="results")
    parser.add_argument("--system-name", default="local_machine")
    parser.add_argument("--estimated-bandwidth-gbps", type=float, default=None)
    return parser.parse_args()


def default_precisions(device_name: str) -> list[str]:
    device = resolve_device(device_name)
    if device.type == "cpu":
        return ["float32"]
    return ["float16", "float32"]


def main() -> None:
    args = parse_args()
    precisions = args.precisions or default_precisions(args.device)
    cfg = ScalingConfig(
        model_names=[args.model],
        prompt_lengths=args.prompt_lengths,
        precisions=precisions,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        measured_trials=args.measured_trials,
        device_preference=args.device,
        save_dir=args.save_dir,
        system_name=args.system_name,
        estimated_bandwidth_gbps=args.estimated_bandwidth_gbps,
        use_kv_cache=True,
    )
    payload = run_scaling(cfg)
    print("Saved precision sweep to:", payload["saved_to"])


if __name__ == "__main__":
    main()
