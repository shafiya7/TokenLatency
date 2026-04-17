from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import argparse

from bench.config import META_LLAMA_VARIANTS, ScalingConfig
from bench.scaling import run_scaling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Goal 3 scaling study across prompt length, model size, and precision.')
    parser.add_argument('--models', nargs='*', default=[META_LLAMA_VARIANTS[0], META_LLAMA_VARIANTS[1], META_LLAMA_VARIANTS[2]], help='Model names to benchmark.')
    parser.add_argument('--prompt-lengths', nargs='*', type=int, default=[16, 32, 64, 128, 256], help='Prompt lengths in tokens.')
    parser.add_argument('--precisions', nargs='*', default=['auto'], help='Torch dtypes to sweep, e.g. auto float16 float32 bfloat16.')
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--measured-trials', type=int, default=3)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--system-name', default='local_machine')
    parser.add_argument('--estimated-bandwidth-gbps', type=float, default=None)
    parser.add_argument('--disable-kv-cache', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ScalingConfig(
        model_names=args.models,
        prompt_lengths=args.prompt_lengths,
        precisions=args.precisions,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        measured_trials=args.measured_trials,
        device_preference=args.device,
        save_dir=args.save_dir,
        system_name=args.system_name,
        estimated_bandwidth_gbps=args.estimated_bandwidth_gbps,
        use_kv_cache=not args.disable_kv_cache,
    )
    payload = run_scaling(cfg)
    print('Saved scaling study to:', payload['saved_to'])


if __name__ == '__main__':
    main()
